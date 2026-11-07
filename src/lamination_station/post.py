"""Structure co-assignment graphs and explicitly defined graph display scores."""

import torch  # noqa: F401
from collections.abc import Hashable
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.stats import rankdata
from sklearn.manifold import SpectralEmbedding


class DiffusionMethod(str, Enum):
    """Supported ways to compute a source-dependent graph display score."""

    PAGERANK = "pagerank"
    SHORTEST_PATH = "shortest_path"


def _structure_ids(out: pd.DataFrame) -> NDArray[np.int64]:
    """Validate finite integer structure IDs without truncating fractional labels."""
    values = out[["structure_pred1", "structure_pred2"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Structure IDs must be finite integers")
    if (np.abs(values) >= 2**53).any():
        raise ValueError("Structure IDs must be exactly representable integers")
    return np.asarray(values, dtype=np.int64)


def build_structure_graph(out: pd.DataFrame) -> nx.Graph:
    """Build a top-two co-assignment graph, not an anatomical contact graph.

    Weight is the overlap coefficient of cell sets; it is not an edge probability.
    Only observed IDs are retained. Empty input yields an empty graph.
    """
    top2 = _structure_ids(out)
    nodes = np.unique(top2)
    memberships = {int(s): set(np.flatnonzero((top2 == s).any(axis=1))) for s in nodes}
    graph = nx.Graph()
    graph.add_nodes_from(int(s) for s in nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            a, b = memberships[int(u)], memberships[int(v)]
            support = len(a & b)
            if support:
                affinity = support / min(len(a), len(b))
                graph.add_edge(
                    int(u),
                    int(v),
                    weight=affinity,
                    affinity=affinity,
                    length=1.0 / affinity,
                    support=support,
                )
    return graph


def _node_diffusion_shortest_path(
    G: nx.Graph,
    source: Hashable,
    weight: str = "weight",
    *,
    weight_is_distance: bool = False,
) -> dict[Hashable, float]:
    """Score closeness to a source, converting positive affinities to reciprocal lengths.

    Set weight_is_distance=True when passing actual lengths. Unreachable nodes
    receive zero; an isolated source receives one. Scores are display coordinates.
    """
    if source not in G:
        raise ValueError("source must be a node in G")
    lengths = G.copy()
    for _, _, data in lengths.edges(data=True):
        value = float(data.get(weight, 1.0))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Graph edge weights must be finite and strictly positive")
        data["_distance"] = value if weight_is_distance or weight == "length" else 1.0 / value
    distances = nx.single_source_dijkstra_path_length(lengths, source, weight="_distance")
    maximum = max(distances.values())
    if maximum == 0:
        return {n: float(n == source) for n in G}
    return {n: max(0.0, 1.0 - distances[n] / maximum) if n in distances else 0.0 for n in G}


def _node_diffusion_pagerank(
    G: nx.Graph,
    source: Hashable,
    alpha: float = 0.85,
    weight: str = "weight",
) -> dict[Hashable, float]:
    """Return a min-max and square-root transformed personalized PageRank display score."""
    if source not in G:
        raise ValueError("source must be a node in G")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    for _, _, data in G.edges(data=True):
        value = float(data.get(weight, 1.0))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Graph edge weights must be finite and strictly positive")
    personalization = {n: float(n == source) for n in G}
    scores = nx.pagerank(
        G, alpha=alpha, personalization=personalization, weight=weight, max_iter=10000, tol=1e-5
    )
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {n: float(n == source) for n in G}
    return {n: float(np.sqrt((v - lo) / (hi - lo))) for n, v in scores.items()}


def compute_cell_diffusion(
    out: pd.DataFrame,
    G: nx.Graph,
    source_node: int,
    method: DiffusionMethod | str = DiffusionMethod.PAGERANK,
    **diff_kwargs: Any,
) -> pd.DataFrame:
    """Interpolate source-dependent display scores along each cell's predicted pair.

    Unknown node IDs and invalid mixing weights raise ValueError. This score is
    neither a calibrated probability nor an anatomical or developmental time.
    """
    ids = _structure_ids(out)
    missing = set(ids.ravel()) - set(G.nodes)
    if missing:
        raise ValueError(f"Predicted structure IDs are absent from G: {sorted(missing)}")
    phi = out["phi"].to_numpy(dtype=float)
    if not np.isfinite(phi).all() or ((phi < 0) | (phi > 1)).any():
        raise ValueError("phi must contain finite values in [0, 1]")
    method = DiffusionMethod(method)
    if method is DiffusionMethod.SHORTEST_PATH:
        scores = _node_diffusion_shortest_path(G, source_node, **diff_kwargs)
    else:
        scores = _node_diffusion_pagerank(G, source_node, **diff_kwargs)
    first = np.array([scores[int(n)] for n in ids[:, 0]], dtype=float)
    second = np.array([scores[int(n)] for n in ids[:, 1]], dtype=float)
    result = out.copy()
    result["diffusion"] = phi * first + (1.0 - phi) * second
    return result


def _rescale(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale finite coordinates to [0, 1], mapping a constant vector to zero."""
    span = float(np.ptp(values))
    if span == 0:
        return np.zeros_like(values)
    return np.asarray((values - values.min()) / span, dtype=np.float64)


def compute_diffusion_time(
    X: NDArray[np.float64],
    n_neighbors: int = 50,
    use_cuda: bool = False,
    *,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a spectral coordinate and percentile-clipped ranks on a connected graph.

    The historical name is retained for compatibility: this is a single Laplacian
    eigenvector, not rooted diffusion pseudotime. n_neighbors excludes the cell
    itself. Disconnected graphs raise rather than inventing an ordering across gaps.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("X must be finite and two-dimensional with at least three cells")
    n = X.shape[0]
    if not 1 <= n_neighbors < n:
        raise ValueError("n_neighbors must be between 1 and n_cells - 1")
    if use_cuda:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors as cuNN

        gpu = cp.asarray(X)
        neighbors = cuNN(n_neighbors=n_neighbors + 1).fit(gpu)
        indices = cp.asnumpy(neighbors.kneighbors(gpu)[1])
    else:
        from sklearn.neighbors import NearestNeighbors

        neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
        indices = neighbors.kneighbors(X)[1]
    idx = np.stack([row[row != i][:n_neighbors] for i, row in enumerate(indices)])
    rows = np.repeat(np.arange(n), n_neighbors)
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(n, n))
    graph = ((graph + graph.T) * 0.5).tocsr()
    components = sparse.csgraph.connected_components(graph, directed=False, return_labels=False)
    if components != 1:
        raise ValueError("The neighbor graph is disconnected; analyze components separately")
    embedding = SpectralEmbedding(n_components=1, affinity="precomputed", random_state=seed)
    coordinate = np.asarray(embedding.fit_transform(graph).ravel(), dtype=np.float64)
    if coordinate[np.argmax(np.abs(coordinate))] < 0:
        coordinate = -coordinate
    diffusion_time = _rescale(coordinate)
    ranks = np.asarray(rankdata(diffusion_time, method="average"), dtype=np.float64)
    ranks = np.clip(ranks, *np.quantile(ranks, [0.01, 0.99]))
    return diffusion_time, _rescale(ranks)
