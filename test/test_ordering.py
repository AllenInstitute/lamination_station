"""Graph, geometry, and branch regressions for laminar ordering."""

import torch  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lamination_station.contacts import (
    aggregate_region_contacts,
    balance_contact_graphs,
    local_cell_contacts,
)
from lamination_station.ordering import OrderMethod, order_regions


def _branched_graph() -> pd.DataFrame:
    """Shared superficial/deep regions with parallel V1/V2 L4 and L5 paths."""
    ids = ["L1", "L23", "V1_L4", "V2_L4", "V1_L5", "V2_L5", "L6", "WM"]
    graph = pd.DataFrame(0.0, index=ids, columns=ids)
    paths = [
        ["L1", "L23", "V1_L4", "V1_L5", "L6", "WM"],
        ["L1", "L23", "V2_L4", "V2_L5", "L6", "WM"],
    ]
    for path in paths:
        for a, b in zip(path[:-1], path[1:], strict=True):
            graph.loc[a, b] = graph.loc[b, a] = 1
    return graph


def test_parallel_layers_split_and_rejoin(tmp_path: Path) -> None:
    """Alternate layers stay parallel, with both paths joining the shared L6."""
    graph = _branched_graph()
    before = graph.copy()
    result = order_regions(adjacency=graph, method=OrderMethod.HARMONIC)
    z = result.regions.coordinate
    # Two parallel three-edge paths have resistance 1.5 between L23 and L6.
    expected = [0, 2 / 7, 3 / 7, 3 / 7, 4 / 7, 4 / 7, 5 / 7, 1]
    np.testing.assert_allclose(z, expected, atol=1e-12)
    for area in ["V1", "V2"]:
        assert z.L23 < z[f"{area}_L4"] < z[f"{area}_L5"] < z.L6
        assert result.precedes.loc["L23", f"{area}_L5"]
        assert result.precedes.loc[f"{area}_L4", "WM"]
    # A scalar rank alone would invent an order between these non-contacting arms.
    assert not result.precedes.loc["V1_L4", "V2_L5"]
    assert not result.precedes.loc["V2_L4", "V1_L5"]
    assert len(result.edges) == 8  # all cycle edges retained; no spanning-tree reduction
    assert result.regions.loc["L23", "n_successors"] == 2
    assert result.regions.loc["L6", "n_predecessors"] == 2
    assert result.regions.loc["V1_L4", "order_group"] == result.regions.loc["V2_L4", "order_group"]
    assert sorted(result.regions.lamina_number) == list(range(1, 9))
    assert not np.diag(result.precedes).any()
    pd.testing.assert_frame_equal(before, graph)
    result.save(tmp_path / "order")
    saved = pd.read_csv(tmp_path / "order/precedes.csv", index_col=0)
    pd.testing.assert_frame_equal(result.precedes, saved)
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "order")
    mapped = result.annotate(pd.Series(["V2_L4", "L1"], index=["cell_a", "cell_b"]))
    assert mapped.index.tolist() == ["cell_a", "cell_b"]
    assert mapped.coordinate.tolist() == [z.V2_L4, 0]


def test_branch_coordinate_is_id_and_reference_invariant() -> None:
    """Inferred poles survive shuffled IDs; a reference may reverse direction only."""
    graph = _branched_graph()
    baseline = order_regions(adjacency=graph, method=OrderMethod.HARMONIC)
    ids = graph.index[np.random.default_rng(4).permutation(len(graph))]
    changed = order_regions(
        adjacency=graph.loc[ids, ids],
        method=OrderMethod.HARMONIC,
        reference_order=pd.Series(np.arange(8)[::-1], index=ids),
    )
    z = changed.regions.coordinate.reindex(graph.index).to_numpy()
    original = baseline.regions.coordinate.to_numpy()
    reverse = np.allclose(z, 1 - original)
    assert reverse or np.allclose(z, original)
    pd.testing.assert_frame_equal(
        changed.precedes.loc[graph.index, graph.index],
        baseline.precedes.T if reverse else baseline.precedes,
    )
    assert set(changed.start_regions + changed.end_regions) == {"L1", "WM"}


def test_unequal_branch_lengths_and_weights() -> None:
    """Different area-specific subdivisions can coexist without forced pairing."""
    graph = _branched_graph().reindex(
        index=[*_branched_graph().index, "V1_L4b"],
        columns=[*_branched_graph().index, "V1_L4b"],
        fill_value=0,
    )
    graph.loc["V1_L4", "V1_L5"] = graph.loc["V1_L5", "V1_L4"] = 0
    for a, b, weight in [("V1_L4", "V1_L4b", 0.4), ("V1_L4b", "V1_L5", 2)]:
        graph.loc[a, b] = graph.loc[b, a] = weight
    result = order_regions(adjacency=graph, method=OrderMethod.HARMONIC)
    for path in [
        ["L1", "L23", "V1_L4", "V1_L4b", "V1_L5", "L6", "WM"],
        ["L1", "L23", "V2_L4", "V2_L5", "L6", "WM"],
    ]:
        assert (np.diff(result.regions.loc[path, "coordinate"]) > 0).all()
    z = result.regions.coordinate.to_numpy()
    lap = np.diag(graph.sum(axis=1)) - graph.to_numpy()
    np.testing.assert_allclose(
        (lap @ z)[[i for i, name in enumerate(graph.index) if name not in ["L1", "WM"]]],
        0,
        atol=1e-12,
    )
    assert not result.precedes.loc["V1_L4b", "V2_L5"]


def test_disconnected_components_and_terminal_branches() -> None:
    """Infer endpoints for every connected piece, including a non-rejoining fork."""
    graph = pd.DataFrame(0.0, index=range(10), columns=range(10))
    for a, b in [(0, 1), (1, 2), (1, 3), (3, 4), (5, 6), (6, 7)]:
        graph.loc[a, b] = graph.loc[b, a] = 1
    result = order_regions(adjacency=graph)
    assert result.regions.coordinate.loc[[8, 9]].isna().all()
    assert result.regions.review_flags.loc[8].find("no_adjacency_support") >= 0
    assert result.regions.coordinate.loc[:7].notna().all()
    assert set(result.start_regions + result.end_regions) == {0, 2, 4, 5, 7}
    assert result.regions.loc[1, "coordinate"] != result.regions.loc[2, "coordinate"]
    assert not result.precedes.loc[1, 6]
    assert sorted(result.regions.lamina_number) == list(range(1, 11))


def test_symmetric_endpoint_ties_and_weight_scaling() -> None:
    """Symmetric alternatives are deterministic, flagged, and independent of scale."""
    graph = pd.DataFrame(np.ones((4, 4)) - np.eye(4), index=[9, 2, 8, 1], columns=[9, 2, 8, 1])
    result = order_regions(adjacency=graph)
    assert result.regions.review_flags.str.contains("endpoint_pair_ambiguous").all()
    assert (result.regions.endpoint_candidate_pairs == 6).all()
    permuted = graph.loc[[1, 8, 9, 2], [1, 8, 9, 2]]
    changed = order_regions(adjacency=permuted * 1000)
    np.testing.assert_allclose(
        result.regions.coordinate, changed.regions.coordinate.reindex(graph.index), atol=1e-12
    )
    assert result.start_regions == changed.start_regions
    assert result.end_regions == changed.end_regions


def test_spectral_chain_and_degenerate_graph() -> None:
    """Legacy chain ordering is preserved; repeated Fiedler eigenvalues are flagged."""
    permutation = np.random.default_rng(18).permutation(9)
    chain = np.diag(np.ones(8), 1) + np.diag(np.ones(8), -1)
    graph = pd.DataFrame(chain[np.ix_(permutation, permutation)])
    prior = pd.Series(permutation, index=graph.index)
    forward = order_regions(adjacency=graph, reference_order=prior, method=OrderMethod.SPECTRAL)
    reverse = order_regions(adjacency=graph, reference_order=-prior, method=OrderMethod.SPECTRAL)
    np.testing.assert_array_equal(forward.regions.reference_rank, prior)
    np.testing.assert_array_equal(
        permutation[np.argsort(forward.regions.lamina_number)], np.arange(9)
    )
    np.testing.assert_array_equal(
        permutation[np.argsort(reverse.regions.lamina_number)], np.arange(9)[::-1]
    )
    complete = pd.DataFrame(np.ones((4, 4)) - np.eye(4))
    ambiguous = order_regions(adjacency=complete, method=OrderMethod.SPECTRAL)
    assert ambiguous.regions.review_flags.str.contains("degenerate_fiedler").all()
    assert ambiguous.regions.review_flags.str.contains("arbitrary_direction").all()


def test_threshold_retains_raw_edges_and_rejects_bad_inputs() -> None:
    """Thresholding changes inference explicitly without deleting contact evidence."""
    graph = _branched_graph()
    graph.loc["V1_L4", "V2_L5"] = graph.loc["V2_L5", "V1_L4"] = 0.01
    result = order_regions(
        adjacency=graph,
        method=OrderMethod.HARMONIC,
        min_weight=0.02,
    )
    assert (result.edges.relation == "excluded").sum() == 1
    assert result.adjacency.loc["V1_L4", "V2_L5"] == 0.01
    assert not result.precedes.loc["V1_L4", "V2_L5"]
    automatic = order_regions(adjacency=graph)
    assert automatic.regions.coordinate.notna().all()
    with pytest.raises(ValueError, match="matching row/column"):
        order_regions(adjacency=graph.iloc[:, ::-1])
    with pytest.raises(ValueError, match="symmetric"):
        order_regions(adjacency=graph.assign(L1=0))
    with pytest.raises(ValueError, match="finite"):
        order_regions(adjacency=graph * np.nan)


def test_contact_geometry_context_and_small_groups() -> None:
    """No cross-piece edges; context blocks false bridges; tiny/coincident groups work."""
    frame = pd.DataFrame(dict(x=[0.0, 1, 2, 3, 4], y=0.0, section="a"))
    full = local_cell_contacts(
        cells=frame,
        x_col="x",
        y_col="y",
        sample_col="section",
        target_ids=pd.Index([0, 4]),
        neighbors=(1, 2),
    )[2]
    assert len(full.left) == 0
    frame["piece"] = ["one", "one", "two", "two", "three"]
    frame.loc[1, "x"] = 0  # coincident two-cell piece plus singleton
    contacts = local_cell_contacts(
        cells=frame,
        x_col="x",
        y_col="y",
        sample_col="section",
        piece_col="piece",
        neighbors=(1, 6, 18),
    )
    for c in contacts.values():
        assert len(c.left) == 2
        assert (c.left != c.right).all()
        np.testing.assert_array_equal(
            frame.piece.to_numpy()[c.left], frame.piece.to_numpy()[c.right]
        )
    isolated = local_cell_contacts(cells=frame.iloc[:1], x_col="x", y_col="y", sample_col=None)[12]
    assert len(isolated.left) == 0


def test_two_cortical_pieces_from_cells_to_parallel_paths() -> None:
    """Physical contacts recover shared layers with area-specific L4/L5 branches."""
    rng = np.random.default_rng(3)
    x, y = np.meshgrid(np.arange(20), np.arange(36))
    frames = []
    for area, angle in [("V1", 0.4), ("V2", -0.9)]:
        xy = np.column_stack([x.ravel(), y.ravel()]) + rng.normal(0, 0.03, (x.size, 2))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = xy @ rotation + (0 if area == "V1" else 200)
        frame = pd.DataFrame(xy, columns=["x", "y"])
        frame["piece"] = area
        names = np.array(["L1", "L23", f"{area}_L4", f"{area}_L5", "L6", "WM"])
        frame["region"] = names[y.ravel() // 6]
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True).assign(section="slide1")
    contacts = local_cell_contacts(
        cells=cells, x_col="x", y_col="y", sample_col="section", piece_col="piece", neighbors=(6,)
    )[6]
    graph = aggregate_region_contacts(
        contacts=contacts, labels=cells.region, samples=cells.section
    )
    result = order_regions(adjacency=graph.adjacency, method=OrderMethod.HARMONIC)
    for area in ["V1", "V2"]:
        path = ["L1", "L23", f"{area}_L4", f"{area}_L5", "L6", "WM"]
        assert (np.diff(result.regions.loc[path, "coordinate"]) > 0).all()
    assert result.regions.loc["L23", "n_successors"] == 2
    assert result.regions.loc["L6", "n_predecessors"] == 2
    assert not result.precedes.loc["V1_L4", "V2_L5"]
    assert not result.precedes.loc["V2_L4", "V1_L5"]


def test_rotation_sample_isolation_and_gap_pruning() -> None:
    """Independent rotations and translations preserve mutual contacts in sections."""
    rng = np.random.default_rng(18)
    cloud = np.vstack([rng.normal(size=(70, 2)), rng.normal(size=(70, 2)) + [1000, 0]])
    frame = pd.DataFrame(np.vstack([cloud, cloud]), columns=["x", "y"])
    frame["section"] = np.repeat(["a", "b"], len(cloud))
    rotated = frame.copy()
    for sample, angle in [("a", 0.63), ("b", -1.12)]:
        use = frame.section.eq(sample)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated.loc[use, ["x", "y"]] = frame.loc[use, ["x", "y"]].to_numpy() @ rotation + [17, 32]
    a = local_cell_contacts(
        cells=frame, x_col="x", y_col="y", sample_col="section", neighbors=(6, 12, 18)
    )
    b = local_cell_contacts(
        cells=rotated, x_col="x", y_col="y", sample_col="section", neighbors=(6, 12, 18)
    )
    pieces = np.tile(np.repeat([0, 1], 70), 2)
    for k in a:
        np.testing.assert_array_equal(a[k].left, b[k].left)
        np.testing.assert_array_equal(a[k].right, b[k].right)
        np.testing.assert_array_equal(pieces[a[k].left], pieces[a[k].right])
        np.testing.assert_array_equal(
            frame.section.to_numpy()[a[k].left], frame.section.to_numpy()[a[k].right]
        )


def test_incident_normalization_empty_regions_and_balancing() -> None:
    """Within-region contacts affect degrees; missing sample support is not zero."""
    frame = pd.DataFrame(dict(x=[0.0, 1.0, 2.0, 3.0], y=0.0, section="s"))
    edges = local_cell_contacts(
        cells=frame, x_col="x", y_col="y", sample_col="section", neighbors=(2,), gap_factor=10
    )[2]
    labels = pd.Series(["outer", "outer", "inner", "inner"])
    summary = aggregate_region_contacts(
        contacts=edges,
        labels=labels,
        samples=frame.section,
        region_ids=pd.Index(["inner", "outer", "empty"]),
        min_cells=1,
    )
    raw = summary.counts[0]
    expected = raw[0, 1] / np.sqrt(raw[0].sum() * raw[1].sum())
    assert summary.adjacency.loc["inner", "outer"] == expected
    assert (summary.adjacency.loc["empty"] == 0).all()
    assert np.isnan(summary.by_sample[0, 0, 2])
    graphs = np.array([[[0, x], [x, 0]] for x in [1.0, 3.0, 10.0]])
    assert balance_contact_graphs(graphs=graphs, units=["a", "a", "b"])[0, 1] == 6
    graphs[0, 0, 1] = graphs[0, 1, 0] = np.nan
    assert balance_contact_graphs(graphs=graphs, units=["a", "a", "b"])[0, 1] == 6.5
    assert not balance_contact_graphs(graphs=graphs[:0], units=[]).any()
    with pytest.raises(ValueError, match="match contact cell_ids"):
        aggregate_region_contacts(contacts=edges, labels=labels.iloc[::-1], samples=frame.section)
