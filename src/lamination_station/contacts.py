"""Local spatial contacts and sample-balanced region adjacency."""

import torch  # noqa: F401

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import cKDTree

from lamination_station.ring_model import positive_integer


@dataclass(frozen=True, kw_only=True)
class CellContacts:
    """Undirected cell pairs indexed into cell_ids, with explicit sample codes.

    Each pair occurs once. Sample codes index sample_ids, including samples with
    no retained contacts. Cell row alignment is part of the graph's identity.
    """

    left: NDArray[np.int64]
    right: NDArray[np.int64]
    sample: NDArray[np.int64]
    cell_ids: pd.Index
    sample_ids: pd.Index


@dataclass(frozen=True, kw_only=True)
class RegionContacts:
    """Normalized adjacency plus raw per-sample counts and sampling-unit labels."""

    adjacency: pd.DataFrame
    by_sample: NDArray[np.float64]
    counts: NDArray[np.float64]
    sample_ids: pd.Index
    units: pd.Series


def local_cell_contacts(
    *,
    cells: pd.DataFrame,
    x_col: str,
    y_col: str,
    sample_col: str | None,
    piece_col: str | None = None,
    target_ids: pd.Index | None = None,
    neighbors: tuple[int, ...] = (12,),
    spacing_neighbors: int = 6,
    gap_factor: float = 2.5,
    workers: int = 1,
) -> dict[int, CellContacts]:
    """Build mutual kNN contacts in full context, then retain target endpoints.

    Use physical coordinates in the same units on both axes. Edges cannot cross
    samples or supplied pieces. Reject lengths greater than gap_factor times
    the smaller endpoint's median local neighbor distance. Small groups use
    all available neighbors; singleton groups have no edges. Coincident cells
    are allowed, but exact-distance ties need not be invariant to row shuffling.
    Never subset to target cells before finding neighbors: that can bridge
    intervening tissue. Supply piece IDs when even small gaps must be isolated;
    the adaptive distance cap alone cannot guarantee separation of tiny islands.
    """
    if not neighbors or len(set(neighbors)) != len(neighbors):
        raise ValueError("neighbors must be nonempty and unique")
    for k in neighbors:
        positive_integer(value=k, name="neighbors")
    positive_integer(value=spacing_neighbors, name="spacing_neighbors")
    positive_integer(value=workers, name="workers")
    if not np.isfinite(gap_factor) or gap_factor <= 0:
        raise ValueError("gap_factor must be positive and finite")
    columns = [x_col, y_col] + [c for c in (sample_col, piece_col) if c is not None]
    if len(set(columns)) != len(columns):
        raise ValueError("Coordinate, sample, and piece columns must be distinct")
    if not cells.index.is_unique or not cells.columns.is_unique or cells.empty:
        raise ValueError("cells require unique row/column identifiers and at least one row")
    if cells.index.to_frame(index=False).isna().any().any():
        raise ValueError("Cell identifiers must not be missing")
    if set(columns) - set(cells.columns) or cells[columns].isna().any().any():
        raise ValueError("Missing coordinate, sample, or piece values/columns")
    xy = cells[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Coordinates must be finite")
    targets = cells.index.copy() if target_ids is None else target_ids.copy()
    if not targets.is_unique or (cells.index.get_indexer(targets) < 0).any():
        raise ValueError("target_ids must be unique identifiers present in cells")
    target_position = targets.get_indexer(cells.index)
    samples = cells[sample_col] if sample_col is not None else pd.Series(0, index=cells.index)
    sample_ids = pd.Index(sorted(samples.unique()), name=sample_col or "sample")
    sample_codes = sample_ids.get_indexer(samples)
    grouping = pd.DataFrame({"sample": sample_codes}, index=cells.index)
    if piece_col is not None:
        grouping["piece"] = cells[piece_col]
    parts: dict[int, list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]]] = {
        k: [] for k in neighbors
    }
    for positions in grouping.groupby(list(grouping), observed=True, sort=True).indices.values():
        n = len(positions)
        if n < 2:
            continue
        query_k = min(max(max(neighbors), spacing_neighbors) + 1, n)
        distance, neighbor = cKDTree(xy[positions]).query(
            xy[positions], k=query_k, workers=workers
        )
        rows = np.repeat(np.arange(n), query_k)
        cols = neighbor.ravel()
        dist = distance.ravel()
        keep = rows != cols
        rows, cols, dist = rows[keep], cols[keep], dist[keep]
        # Remove self by identity: it need not be the first coincident neighbor.
        starts = np.repeat(
            np.cumsum(np.bincount(rows, minlength=n)) - np.bincount(rows, minlength=n),
            np.bincount(rows, minlength=n),
        )
        rank = np.arange(len(rows)) - starts + 1
        spacing = (
            pd.Series(dist[rank <= spacing_neighbors])
            .groupby(rows[rank <= spacing_neighbors], sort=True)
            .median()
            .reindex(np.arange(n))
            .to_numpy()
        )
        positive = spacing[spacing > 0]
        if len(positive):
            spacing = np.maximum(spacing, np.median(positive) * 1e-6)
        local = dist <= gap_factor * np.minimum(spacing[rows], spacing[cols])
        for k in neighbors:
            use = local & (rank <= k)
            directed = sparse.csr_matrix(
                (np.ones(use.sum(), dtype=np.uint8), (rows[use], cols[use])), shape=(n, n)
            )
            mutual = sparse.triu(directed.multiply(directed.T), k=1).tocoo()
            a = target_position[positions[mutual.row]]
            b = target_position[positions[mutual.col]]
            valid = (a >= 0) & (b >= 0)
            parts[k].append((a[valid], b[valid], np.full(valid.sum(), sample_codes[positions[0]])))
    result = {}
    for k, chunks in parts.items():
        arrays = [
            np.concatenate([c[i] for c in chunks]).astype(np.int64)
            if chunks
            else np.empty(0, dtype=np.int64)
            for i in range(3)
        ]
        result[k] = CellContacts(
            left=arrays[0],
            right=arrays[1],
            sample=arrays[2],
            cell_ids=targets,
            sample_ids=sample_ids,
        )
    return result


def _available_mean(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average observed pairs without interpreting absent pairs as zero contact."""
    count = np.isfinite(array).sum(axis=0)
    result: NDArray[np.float64] = np.divide(
        np.nansum(array, axis=0), count, out=np.full(array.shape[1:], np.nan), where=count > 0
    )
    return result


def balance_contact_graphs(
    *,
    graphs: NDArray[np.float64],
    units: Sequence[object] | pd.Series | NDArray[np.str_],
) -> NDArray[np.float64]:
    """Average available sections within units, then weight available units equally.

    NaN denotes an unsupported pair. Empty input returns a zero adjacency of the
    supplied region shape, useful for leave-one-unit-out when only one unit exists.
    """
    graph = np.asarray(graphs, dtype=float)
    unit = np.asarray(units)
    if graph.ndim != 3 or graph.shape[1] != graph.shape[2] or unit.shape != (len(graph),):
        raise ValueError("graphs must be sample x region x region, aligned with units")
    if pd.isna(unit).any() or np.isinf(graph).any() or (graph < 0).any():
        raise ValueError("Units cannot be missing; graph weights must be nonnegative or NaN")
    if not np.allclose(graph, graph.transpose(0, 2, 1), equal_nan=True):
        raise ValueError("Contact graphs must be symmetric")
    if not len(graph):
        return np.zeros(graph.shape[1:], dtype=float)
    means = np.array([_available_mean(graph[unit == u]) for u in pd.unique(unit)])
    result = np.nan_to_num(_available_mean(means))
    np.fill_diagonal(result, 0)
    return result


def aggregate_region_contacts(
    *,
    contacts: CellContacts,
    labels: pd.Series,
    samples: pd.Series,
    units: pd.Series | None = None,
    region_ids: pd.Index | None = None,
    min_cells: int = 20,
) -> RegionContacts:
    """Normalize cell contacts by incident degree and balance sampling units.

    labels and samples must exactly match contacts.cell_ids in row order; units
    is indexed by sample ID (one unit per section), or defaults to sample IDs.
    Within-region edges contribute twice to degrees but not to final adjacency.
    Pairs with fewer than min_cells in either region/sample are unobserved, not
    zero. Supply region_ids to retain empty fitted components. Region labels can
    be strings or integers; input assignments are never relabeled or mutated.
    """
    positive_integer(value=min_cells, name="min_cells")
    if not labels.index.equals(contacts.cell_ids) or not samples.index.equals(contacts.cell_ids):
        raise ValueError("Label and sample rows must match contact cell_ids exactly")
    if labels.isna().any() or samples.isna().any():
        raise ValueError("Region and sample labels must not be missing")
    ids = pd.Index(pd.unique(labels), name="region") if region_ids is None else region_ids.copy()
    if not ids.is_unique or ids.empty or ids.hasnans:
        raise ValueError("region_ids must be nonempty, unique, and nonmissing")
    codes = ids.get_indexer(labels)
    sample_codes = contacts.sample_ids.get_indexer(samples)
    if (codes < 0).any() or (sample_codes < 0).any():
        raise ValueError("Unknown region or sample IDs")
    a, b, s = contacts.left, contacts.right, contacts.sample
    if any(v.ndim != 1 or v.dtype.kind not in "iu" for v in (a, b, s)) or not (
        len(a) == len(b) == len(s)
    ):
        raise ValueError("Contact endpoints and sample codes must be aligned integer vectors")
    if (
        (a < 0).any()
        or (b < 0).any()
        or (a >= len(labels)).any()
        or (b >= len(labels)).any()
        or (s < 0).any()
        or (s >= len(contacts.sample_ids)).any()
        or (a == b).any()
    ):
        raise ValueError("Invalid contact endpoints or sample codes")
    if not np.array_equal(sample_codes[a], s) or not np.array_equal(sample_codes[b], s):
        raise ValueError("Contacts cross samples or sample labels have changed")
    pairs = np.sort(np.column_stack([a, b]), axis=1)
    if len(np.unique(pairs[:, 0] * len(labels) + pairs[:, 1])) != len(a):
        raise ValueError("Each undirected cell pair must occur exactly once")
    unit_labels = (
        pd.Series(contacts.sample_ids, index=contacts.sample_ids)
        if units is None
        else units.reindex(contacts.sample_ids)
    )
    if not unit_labels.index.is_unique or unit_labels.isna().any():
        raise ValueError("units must identify exactly one nonmissing unit for every sample")
    k = len(ids)
    raw = np.zeros((len(contacts.sample_ids), k, k), dtype=float)
    graphs = np.zeros_like(raw)
    for i in range(len(contacts.sample_ids)):
        use = s == i
        count = (
            np.bincount(codes[a[use]] * k + codes[b[use]], minlength=k * k)
            .reshape(k, k)
            .astype(float)
        )
        count += count.T.copy()
        raw[i] = count
        degree = count.sum(axis=1)
        denom = np.sqrt(degree[:, None] * degree[None, :])
        graph = np.divide(count, denom, out=np.zeros_like(count), where=denom > 0)
        present = np.bincount(codes[sample_codes == i], minlength=k) >= min_cells
        graph[~(present[:, None] & present[None, :])] = np.nan
        np.fill_diagonal(graph, 0)
        graphs[i] = graph
    balanced = balance_contact_graphs(graphs=graphs, units=unit_labels)
    return RegionContacts(
        adjacency=pd.DataFrame(balanced, index=ids, columns=ids),
        by_sample=graphs,
        counts=raw,
        sample_ids=contacts.sample_ids.copy(),
        units=unit_labels.copy(),
    )
