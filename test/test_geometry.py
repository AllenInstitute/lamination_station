"""Geometric invariance and graph scoring regressions."""

import torch  # noqa: F401

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from lamination_station import po, pp, pl


def test_rotation_and_input_preservation() -> None:
    """Rigid rotation must preserve counts under the default spatial metric."""
    rng = np.random.default_rng(41)
    xy = rng.normal(size=(80, 2)) * [10, 1]
    frame = pd.DataFrame(xy, columns=["x", "y"])
    frame["type"] = ["A", "B"] * 40
    before = frame.copy(deep=True)
    rotated = frame.copy()
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated[["x", "y"]] = xy @ rotation.T
    a = pp.calculate_comp_grads(frame, "x", "y", "type", near_k=8, far_k=16)
    b = pp.calculate_comp_grads(rotated, "x", "y", "type", near_k=8, far_k=16)
    np.testing.assert_array_equal(a[5], b[5])
    np.testing.assert_allclose(a[2] @ rotation.T, b[2], atol=1e-10)
    pd.testing.assert_frame_equal(frame, before)


def test_duplicate_coordinates_exclude_self() -> None:
    """Tied coordinates must not introduce self counts or invalid type vectors."""
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0],
            "type": pd.Categorical(["A", "B", "C", "D", "E"]),
        }
    )
    with np.errstate(all="raise"):
        result = pp.calculate_comp_grads(frame, "x", "y", "type", near_k=2, far_k=3)
    assert np.equal(np.diag(result[5]), 0).all()
    assert np.isfinite(result[6].to_numpy()).all()


def test_sections_never_share_neighbors() -> None:
    """Identical coordinate frames in different sections must remain disconnected."""
    xy = np.random.default_rng(4).normal(size=(12, 2))
    frame = pd.DataFrame(np.tile(xy, (2, 1)), columns=["x", "y"])
    frame["sample"] = ["first"] * 12 + ["second"] * 12
    frame["type"] = pd.Categorical(["A"] * 12 + ["B"] * 12)
    counts = pp.calculate_comp_grads(
        frame, "x", "y", "type", near_k=3, far_k=5, sample_col="sample"
    )[5]
    assert (counts.iloc[:12].to_numpy() == [3, 0]).all()
    assert (counts.iloc[12:].to_numpy() == [0, 3]).all()


def test_graph_preserves_only_observed_ids_and_empty_inputs() -> None:
    """Sparse numeric IDs must not create phantom nodes."""
    frame = pd.DataFrame({"structure_pred1": [2, 8], "structure_pred2": [8, 2], "phi": [0.9, 0.8]})
    graph = po.build_structure_graph(frame)
    assert set(graph) == {2, 8}
    assert graph[2][8]["length"] == 1
    assert len(po.build_structure_graph(frame.iloc[:0])) == 0
    with pytest.raises(ValueError, match="finite integers"):
        po.build_structure_graph(frame.assign(structure_pred1=2.5))


def test_affinity_lengths_isolates_and_unknown_ids() -> None:
    """Stronger affinity means shorter paths and isolated sources remain valid."""
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 0.9), (0, 2, 0.1)])
    graph.add_node(3)
    scores = po._node_diffusion_shortest_path(graph, 0)
    assert scores[1] > scores[2]
    assert scores[3] == 0
    assert po._node_diffusion_shortest_path(graph, 3) == {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}
    assert po._node_diffusion_pagerank(nx.empty_graph(1), 0) == {0: 1.0}
    frame = pd.DataFrame({"structure_pred1": [0], "structure_pred2": [99], "phi": [0.5]})
    with pytest.raises(ValueError, match="absent"):
        po.compute_cell_diffusion(frame, graph, 0)


def test_spectral_coordinate_rejects_disconnected_graph() -> None:
    """Independent components cannot be ordered by an arbitrary eigenvector."""
    points = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [20.0, 20.0], [20.0, 21.0], [21.0, 20.0]]
    )
    with pytest.raises(ValueError, match="disconnected"):
        po.compute_diffusion_time(points, n_neighbors=2)


def test_plot_seed_and_ylim() -> None:
    """The public seed controls jitter and ylim works without xlim."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {
            "structure_pred1": [0, 1],
            "structure_pred2": [1, 0],
            "phi": [0.9, 0.8],
            "color": [1.0, 2.0],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "gradient_norm": [1.0, 1.0],
        }
    )
    graph = po.build_structure_graph(frame)
    a, _ = pl.plot_cells_on_structure_graph(frame, graph, color_col="color", seed=12)
    b, _ = pl.plot_cells_on_structure_graph(frame, graph, color_col="color", seed=12)
    np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    gradients: NDArray[np.float64] = np.ones((2, 2))
    pl.plot_gradient_arrows(frame, gradients, "structure_pred1", "x", "y", ylim=(-2.0, 3.0))
    np.testing.assert_array_equal(plt.gca().get_ylim(), [-2.0, 3.0])
    plt.close("all")
