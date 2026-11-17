"""Automatically inferred laminar coordinates and ordering on branching region graphs."""

import torch  # noqa: F401

import json
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr


class OrderMethod(str, Enum):
    """Automatic harmonic ordering or legacy spectral display seriation."""

    SPECTRAL = "spectral"
    HARMONIC = "harmonic"


@dataclass(frozen=True, kw_only=True)
class LaminarOrdering:
    """Display numbers, graph coordinates, contacts, and a graph partial order.

    lamina_number is a unique 1-based display permutation, never a biological
    total order. order_group groups numerically tied coordinates, not homologous
    layers. precedes[a,b] is true only for a strictly increasing path of retained
    contacts; parallel arms need not be comparable. Flat contacts are retained
    in edges but do not impose precedence. NaN coordinates denote isolated
    regions. Never infer V1/V2 identity or laminar homology from this object.
    """

    regions: pd.DataFrame
    adjacency: pd.DataFrame
    edges: pd.DataFrame
    precedes: pd.DataFrame
    method: OrderMethod
    start_regions: tuple[Hashable, ...]
    end_regions: tuple[Hashable, ...]
    min_weight: float
    tie_tolerance: float

    def annotate(self, labels: pd.Series) -> pd.DataFrame:
        """Map ordering fields onto cell labels without changing their assignments."""
        if labels.isna().any() or not labels.isin(self.regions.index).all():
            raise ValueError("Cell labels contain missing or unknown region IDs")
        frame = self.regions.loc[labels.to_numpy()].copy()
        frame.index = labels.index.copy()
        return frame

    def save(self, directory: str | Path) -> None:
        """Write the complete graph and ordering into a new directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.regions.to_csv(path / "regions.csv")
        self.adjacency.to_csv(path / "adjacency.csv")
        self.edges.to_csv(path / "edges.csv", index=False)
        self.precedes.to_csv(path / "precedes.csv")
        manifest = dict(
            schema_version=2,
            method=self.method.value,
            start_regions=[str(v) for v in self.start_regions],
            end_regions=[str(v) for v in self.end_regions],
            endpoint_inference="Maximum effective-resistance pair in each connected component, extended to degree-one terminals by nearer-pole resistance distance; numerical ties use stable region IDs.",
            min_weight=self.min_weight,
            tie_tolerance=self.tie_tolerance,
            reference_order="regions.csv reference_rank; missing when no reference was supplied",
            interpretation="Numbers are display positions. Coordinates and directed paths are conditional on the graph and inferred endpoints; they are not physical depth or anatomical homology.",
            references=[
                "https://epubs.siam.org/doi/abs/10.1137/S0097539795285771",
                "https://leogrady.net/wp-content/uploads/2017/01/grady2006random.pdf",
            ],
        )
        (path / "method.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))


def _checked_graph(adjacency: pd.DataFrame) -> NDArray[np.float64]:
    """Reject misaligned, directed, signed, or nonfinite contact matrices."""
    if (
        adjacency.empty
        or not adjacency.index.is_unique
        or adjacency.index.hasnans
        or not adjacency.index.equals(adjacency.columns)
    ):
        raise ValueError("Adjacency requires nonempty unique matching row/column region IDs")
    graph: NDArray[np.float64] = adjacency.to_numpy(dtype=float, copy=True)
    if not np.isfinite(graph).all() or (graph < 0).any():
        raise ValueError("Adjacency weights must be finite and nonnegative")
    if not np.allclose(graph, graph.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Adjacency must be symmetric")
    if not np.allclose(np.diag(graph), 0, atol=1e-12, rtol=0):
        raise ValueError("Adjacency must have zero diagonal")
    graph = (graph + graph.T) / 2
    np.fill_diagonal(graph, 0)
    return graph


def _spectral_coordinate(
    *,
    graph: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    """Orient the Fiedler vector by prior rank and report a degenerate eigenspace."""
    values, vectors = np.linalg.eigh(np.diag(graph.sum(axis=1)) - graph)
    vector = vectors[:, 1]
    order = np.argsort(vector, kind="stable")
    rho = spearmanr(np.arange(len(order)), prior[order]).statistic
    if np.isfinite(rho) and rho < 0:
        vector = -vector
    coordinate = (vector - vector.min()) / np.ptp(vector)
    ambiguous = len(values) > 2 and values[2] - values[1] <= 1e-8 * max(values[-1], 1e-30)
    return coordinate, bool(ambiguous)


def _infer_harmonic(
    graph: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], int, float, list[str]]:
    """Infer poles and terminal sets, then interpolate on the full connected graph.

    Weights are conductances, so stronger contacts reduce resistance distance.
    Input rows must already be in stable ID order for reproducible tie breaking.
    Endpoint margin is the relative gap between the top two pair distances;
    it is a diagnostic, never a probability or calibrated confidence measure.
    """
    scaled = graph / graph.max()
    lap = np.diag(scaled.sum(axis=1)) - scaled
    eigenvalues, vectors = np.linalg.eigh(lap)
    if eigenvalues[1] <= np.finfo(float).eps * eigenvalues[-1] * len(graph):
        raise ValueError("Contact graph is numerically disconnected; increase min_weight")
    inverse = (vectors[:, 1:] / eigenvalues[1:]) @ vectors[:, 1:].T
    resistance = np.diag(inverse)[:, None] + np.diag(inverse)[None, :] - 2 * inverse
    a, b = np.triu_indices(len(graph), 1)
    distances = resistance[a, b]
    maximum = distances.max()
    candidates = np.flatnonzero(distances >= maximum * (1 - 1e-10))
    left, right = a[candidates[0]], b[candidates[0]]
    top = np.sort(distances)
    margin = float((top[-1] - top[-2]) / maximum) if len(top) > 1 else 1.0
    start = np.zeros(len(graph), dtype=bool)
    end = start.copy()
    start[left] = end[right] = True
    flags = []
    if len(candidates) > 1:
        flags.append("endpoint_pair_ambiguous")
    # Include additional terminal arms rather than leaving them flat at a junction.
    for terminal in np.flatnonzero((graph > 0).sum(axis=1) == 1):
        if terminal in (left, right):
            continue
        difference = resistance[terminal, left] - resistance[terminal, right]
        if abs(difference) <= maximum * 1e-10:
            flags.append("terminal_side_ambiguous")
        if difference <= maximum * 1e-10:
            start[terminal] = True
        else:
            end[terminal] = True
    boundary = start | end
    value = end.astype(float)
    unknown = ~boundary
    if unknown.any():
        value[unknown] = np.linalg.solve(
            lap[np.ix_(unknown, unknown)], -lap[np.ix_(unknown, boundary)] @ value[boundary]
        )
    return np.clip(value, 0, 1), start, end, len(candidates), margin, sorted(set(flags))


def _tied_order(
    *,
    ids: NDArray[np.int64],
    coordinate: NDArray[np.float64],
    prior: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray[np.int64], list[NDArray[np.int64]]]:
    """Sort coordinates, using prior order only to break numerical ties."""
    ordered = ids[np.argsort(coordinate[ids], kind="stable")]
    groups = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and coordinate[ordered[end]] - coordinate[ordered[start]] <= tolerance
        ):
            end += 1
        block = ordered[start:end]
        groups.append(block[np.argsort(prior[block], kind="stable")])
        start = end
    return np.concatenate(groups), groups


def order_regions(
    *,
    adjacency: pd.DataFrame,
    method: OrderMethod = OrderMethod.HARMONIC,
    reference_order: pd.Series | None = None,
    min_weight: float = 0.0,
    tie_tolerance: float = 1e-8,
) -> LaminarOrdering:
    """Order regions while retaining splits, parallel paths, and reconvergence.

    Spectral mode reproduces Fiedler seriation. reference_order sets its sign,
    numerical tie order, and disconnected-component placement only. Without a
    reference, direction is arbitrary and flagged. It is a display heuristic on
    branching graphs, not a depth estimator.

    Harmonic mode automatically selects the maximum effective-resistance pair
    per connected component, extends each side to degree-one terminal branches
    by nearer-pole resistance distance, and solves the graph Dirichlet problem
    with these inferred endpoint sets fixed at 0/1. No labels, anchor regions,
    or prior knowledge of branching are required. All contacts and reconvergent
    paths are retained. Endpoint-pair ties and equidistant terminal choices are
    flagged; stable region IDs break numerical ties. Isolated regions have NaN
    coordinates but still receive display numbers.

    reference_order can only flip the inferred direction, break coordinate ties,
    and place disconnected components. Without it, stable region IDs set direction
    and the result is flagged arbitrary_direction. Never interpret automatically
    inferred start/end regions as anatomical superficial/deep labels. A weakly
    connected outlier can become an endpoint; inspect sample/neighborhood
    sensitivity. This is a naming heuristic, not a depth or homology estimator.

    min_weight removes weights <= that threshold from inference only; every
    original edge remains exported with a used flag. Coordinates depend on
    contact weights, not physical distance. Partial order is graph-conditional;
    weak lateral contacts can change it, so review threshold/endpoint sensitivity.
    """
    if not isinstance(method, OrderMethod):
        raise ValueError("method must be an OrderMethod")
    if not np.isfinite(min_weight) or min_weight < 0:
        raise ValueError("min_weight must be finite and nonnegative")
    if not np.isfinite(tie_tolerance) or not 0 <= tie_tolerance < 1:
        raise ValueError("tie_tolerance must be finite and in [0, 1)")
    graph = _checked_graph(adjacency)
    original = graph.copy()
    graph[graph <= min_weight] = 0
    ids = adjacency.index
    k = len(ids)
    canonical = np.array(sorted(range(k), key=lambda i: (type(ids[i]).__name__, repr(ids[i]))))
    stable_rank = np.empty(k, dtype=float)
    stable_rank[canonical] = np.arange(k)
    prior = stable_rank.copy()
    if reference_order is not None:
        if not reference_order.index.is_unique or set(reference_order.index) != set(ids):
            raise ValueError("reference_order must cover exactly the adjacency region IDs")
        prior = reference_order.reindex(ids).to_numpy(dtype=float)
        if not np.isfinite(prior).all() or len(np.unique(prior)) != k:
            raise ValueError("reference_order must contain finite unique ranks")
    surface = np.zeros(k, dtype=bool)
    bottom = np.zeros(k, dtype=bool)
    candidate_pairs = np.zeros(k, dtype=int)
    endpoint_margin = np.full(k, np.nan)
    n_components, components = connected_components(sparse.csr_matrix(graph > 0), directed=False)
    coordinate = np.full(k, np.nan)
    flags: list[list[str]] = [[] for _ in range(k)]
    pieces = sorted(
        [np.flatnonzero(components == c) for c in range(n_components)],
        key=lambda v: np.median(prior[v]),
    )
    order_parts = []
    group_labels = np.zeros(k, dtype=int)
    next_group = 1
    for members in pieces:
        members = members[np.argsort(stable_rank[members])]
        sub = graph[np.ix_(members, members)]
        component_flags = []
        if n_components > 1:
            component_flags.append("component_placement_is_display_only")
        if len(members) == 1:
            component_flags.append("no_adjacency_support")
        elif method == OrderMethod.SPECTRAL:
            coordinate[members], ambiguous = _spectral_coordinate(graph=sub, prior=prior[members])
            if reference_order is None:
                component_flags.append("arbitrary_direction")
            if ambiguous:
                component_flags.append("degenerate_fiedler_vector")
        if method == OrderMethod.HARMONIC and len(members) > 1:
            value, start, end, n_pairs, margin, inference_flags = _infer_harmonic(sub)
            candidate_pairs[members] = n_pairs
            endpoint_margin[members] = margin
            rho = spearmanr(value, prior[members]).statistic
            if np.isfinite(rho) and rho < 0:
                value = 1 - value
                start, end = end, start
            coordinate[members] = value
            surface[members] = start
            bottom[members] = end
            component_flags.extend(inference_flags)
            if reference_order is None:
                component_flags.append("arbitrary_direction")
        if np.isfinite(coordinate[members]).all():
            ordered, groups = _tied_order(
                ids=members, coordinate=coordinate, prior=prior, tolerance=tie_tolerance
            )
            for block in groups:
                group_labels[block] = next_group
                next_group += 1
                if len(block) > 1:
                    for i in block:
                        flags[i].append("coordinate_tie")
        else:
            ordered = members[np.argsort(prior[members], kind="stable")]
        order_parts.append(ordered)
        for i in members:
            flags[i].extend(component_flags)
    order = np.concatenate(order_parts)
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(1, k + 1)
    delta = coordinate[None, :] - coordinate[:, None]
    directed = (graph > 0) & (delta > tie_tolerance)
    precedes = directed.copy()
    # Boolean transitive closure retains incomparable parallel paths, unlike ranks.
    for i in range(k):
        precedes |= precedes[:, i, None] & precedes[i, None, :]
    edge_records = []
    for a, b in zip(*np.where(np.triu(original > 0, 1)), strict=True):
        used = bool(graph[a, b] > 0)
        relation = (
            "excluded"
            if not used
            else "unoriented"
            if not np.isfinite(delta[a, b])
            else "a_before_b"
            if directed[a, b]
            else "b_before_a"
            if directed[b, a]
            else "flat"
        )
        edge_records.append((ids[a], ids[b], original[a, b], used, relation))
    regions = pd.DataFrame(
        dict(
            lamina_number=rank,
            reference_rank=prior if reference_order is not None else np.full(k, np.nan),
            coordinate=coordinate,
            order_group=pd.array([int(g) if g else pd.NA for g in group_labels], dtype="Int64"),
            graph_component=components,
            contact_degree=graph.sum(axis=1),
            endpoint_role=np.where(surface, "start", np.where(bottom, "end", "interior")),
            endpoint_candidate_pairs=candidate_pairs,
            endpoint_margin=endpoint_margin,
            n_predecessors=directed.sum(axis=0),
            n_successors=directed.sum(axis=1),
            review_flags=[";".join(v) or "none" for v in flags],
        ),
        index=ids.copy().rename("region"),
    )
    return LaminarOrdering(
        regions=regions,
        adjacency=adjacency.copy(deep=True),
        edges=pd.DataFrame(
            edge_records, columns=["region_a", "region_b", "weight", "used", "relation"]
        ),
        precedes=pd.DataFrame(precedes, index=ids, columns=ids),
        method=method,
        start_regions=tuple(ids[surface]),
        end_regions=tuple(ids[bottom]),
        min_weight=min_weight,
        tie_tolerance=tie_tolerance,
    )
