"""Physical neighborhood aggregation, separate from composition co-assignment."""

import torch  # noqa: F401
from collections import Counter
from collections.abc import Hashable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def build_spatial_structure_graph(
    cells: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    structure_col: str,
    sample_col: str,
    max_distance: float,
    n_neighbors: int = 6,
) -> nx.Graph:
    """Aggregate distance-capped, mutual spatial kNN edges between hard assignments.

    Coordinates and max_distance must share physical units. All cells, including
    those inside structures, contribute to node degrees. Weight is twice the
    between-structure edge count divided by their combined cell degrees; it is a
    descriptive affinity, never a contact probability. Edges never cross samples
    and disconnected nodes are retained. IDs must be integers. Counts describe
    sampled point proximity, not membrane contact. Recompute after spatial thinning.
    """
    if n_neighbors < 1 or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("n_neighbors and max_distance must be positive and finite")
    if not cells.index.is_unique:
        raise ValueError("Cell identifiers must be unique")
    coordinates = cells[[x_col, y_col]].to_numpy(dtype=float)
    ids = cells[structure_col].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or cells[sample_col].isna().any():
        raise ValueError("Coordinates must be finite and sample identifiers must not be missing")
    if not np.isfinite(ids).all() or (ids != np.floor(ids)).any() or (abs(ids) >= 2**53).any():
        raise ValueError("Structure IDs must be exactly representable finite integers")
    labels = ids.astype(np.int64)
    graph = nx.Graph()
    for label, size in Counter(labels).items():
        graph.add_node(int(label), cells=int(size), cell_degree=0)
    sample_support: dict[tuple[int, int], set[Hashable]] = {}
    for sample, rows in cells.groupby(sample_col, observed=True, sort=False).indices.items():
        if len(rows) < 2:
            continue
        k = min(n_neighbors, len(rows) - 1)
        xy = coordinates[rows]
        distance, index = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
        directed = set()
        for i, (neighbors, lengths) in enumerate(zip(index, distance, strict=True)):
            keep = neighbors != i
            for j, length in zip(neighbors[keep][:k], lengths[keep][:k], strict=True):
                if length <= max_distance:
                    directed.add((i, int(j)))
        for i, j in sorted(directed):
            if i >= j or (j, i) not in directed:
                continue
            u, v = int(labels[rows[i]]), int(labels[rows[j]])
            graph.nodes[u]["cell_degree"] += 1
            graph.nodes[v]["cell_degree"] += 1
            if u == v:
                continue
            edge = (min(u, v), max(u, v))
            sample_support.setdefault(edge, set()).add(sample)
            if graph.has_edge(u, v):
                graph[u][v]["support"] += 1
            else:
                graph.add_edge(u, v, support=1)
    for u, v, data in graph.edges(data=True):
        degree = graph.nodes[u]["cell_degree"] + graph.nodes[v]["cell_degree"]
        affinity = 2 * data["support"] / degree
        data.update(
            weight=affinity,
            affinity=affinity,
            length=1 / affinity,
            sample_support=len(sample_support[(min(u, v), max(u, v))]),
            samples=tuple(sorted(sample_support[(min(u, v), max(u, v))], key=repr)),
        )
    return graph
