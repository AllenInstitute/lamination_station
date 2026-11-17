"""Scientific and workflow regressions for the supported gradient-ring caller."""

import torch

import json
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multinomial
from sklearn.metrics import adjusted_rand_score

from lamination_station import regions
from lamination_station.ring_model import FitSettings, RingMixture, fit_ring_mixture


def _contrasting_rings() -> tuple[tuple[pd.DataFrame, ...], NDArray[np.int64]]:
    """Make distinct radial profiles with identical total type compositions."""
    rng = np.random.default_rng(41)
    truth = np.repeat([0, 1], 80)
    a = rng.binomial(15, np.where(truth == 0, 0.95, 0.05))
    inner = pd.DataFrame({"A_count": a, "B_count": 15 - a})
    outer = pd.DataFrame({"A_count": 15 - a, "B_count": a})
    return (inner, outer), truth


@pytest.fixture
def prepared() -> regions.GradientRings:
    """Create a small realistic preparation with enough distinct observations."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    xy = rng.normal(size=(100, 2))
    frame = pd.DataFrame(
        xy, columns=["x", "y"], index=pd.Index([f"cell_{i}" for i in range(100)], name="cell_id")
    )
    frame["type"] = np.where(xy[:, 0] < -0.2, "A", np.where(xy[:, 0] > 0.3, "B", "C"))
    frame["section"] = "one"
    return regions.prepare_gradient_rings(
        cells=frame,
        x_col="x",
        y_col="y",
        celltype_col="type",
        sample_col="section",
        settings=regions.GradientSettings(ring_ends=(4, 8, 12), far_k=18),
    )


def test_ring_likelihood_retains_information_lost_by_pooling() -> None:
    """Opposite inner/outer profiles remain distinct even with identical totals."""
    rings, truth = _contrasting_rings()
    np.testing.assert_array_equal(sum(rings).to_numpy(), np.full((160, 2), 15))
    model = fit_ring_mixture(rings=rings, settings=FitSettings(n_regions=2, n_init=3))
    assert adjusted_rand_score(truth, model.predict_proba(rings).to_numpy().argmax(axis=1)) > 0.98
    assert model.converged
    assert np.diff(model.objective_history).min(initial=0) >= -1e-8
    expected = []
    for k in range(2):
        expected.append(
            np.log(model.weights[k])
            + sum(
                multinomial.logpmf(r.to_numpy(), n=15, p=model.profiles[j].iloc[k].to_numpy())
                for j, r in enumerate(rings)
            )
        )
    np.testing.assert_allclose(model.score_samples(rings), logsumexp(expected, axis=0))
    assert model.starts[model.best_start].objective == max(s.objective for s in model.starts)


def test_single_component_has_correct_map_profiles() -> None:
    """Each ring receives its own Dirichlet pseudocount rather than pooled counts."""
    rings = (
        pd.DataFrame([[4, 0], [4, 0]], columns=["a", "b"]),
        pd.DataFrame([[0, 6], [0, 6]], columns=["a", "b"]),
    )
    model = fit_ring_mixture(rings=rings, settings=FitSettings(n_regions=1, n_init=1))
    np.testing.assert_allclose(model.profiles[0], [[8.5 / 9, 0.5 / 9]])
    np.testing.assert_allclose(model.profiles[1], [[0.5 / 13, 12.5 / 13]])
    np.testing.assert_allclose(model.predict_proba(rings), 1)


def test_groups_and_pieces_never_share_neighbors() -> None:
    """Piece IDs repeat across sections without creating cross-piece type counts."""
    xy = np.random.default_rng(4).normal(size=(20, 2))
    cells = pd.DataFrame(np.tile(xy, (3, 1)), columns=["x", "y"])
    cells["section"] = ["s1"] * 40 + ["s2"] * 20
    cells["piece"] = ["p1"] * 20 + ["p2"] * 20 + ["p1"] * 20
    cells["type"] = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    before = cells.copy(deep=True)
    p = regions.prepare_gradient_rings(
        cells=cells,
        x_col="x",
        y_col="y",
        celltype_col="type",
        sample_col="section",
        piece_col="piece",
        settings=regions.GradientSettings(ring_ends=(3, 6, 9), far_k=12),
    )
    for ring in p.rings:
        np.testing.assert_array_equal(
            ring.to_numpy()[:, :3], np.repeat(np.eye(3, dtype=int) * 3, 20, axis=0)
        )
        assert (ring["__unseen_count"] == 0).all()
    pd.testing.assert_frame_equal(cells, before)


def test_resolution_scan_and_export_roundtrip(
    *, prepared: regions.GradientRings, tmp_path: Path
) -> None:
    """Resolution is selectable; saved models retain exact predictions and metadata."""
    scan = regions.fit_resolutions(features=prepared, n_regions=(2, 4), seeds=(17, 29), n_init=2)
    assert list(scan.summary.index) == [2, 4]
    assert scan.summary.mean_seed_ari.between(-1, 1).all()
    result = scan.best(n_regions=4)
    assert result.memberships.shape == (100, 4)
    assert result.assignments.index.equals(prepared.cells.index)
    assert result.region_summary.n_cells.sum() == 100
    folder = tmp_path / "analysis"
    result.save(folder)
    restored = RingMixture.load(folder / "model.json")
    pd.testing.assert_frame_equal(restored.predict_proba(prepared.rings), result.memberships)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["requested_regions"] == 4
    assert manifest["fit_fingerprint"] == result.fit_fingerprint
    assert manifest["input_fingerprint"] == prepared.input_fingerprint
    saved = pd.read_csv(folder / "assignments.csv", index_col=0)
    pd.testing.assert_index_equal(saved.index, result.assignments.index)
    with pytest.raises(FileExistsError):
        result.save(folder)
    with pytest.raises(ValueError, match="not fitted"):
        scan.best(n_regions=3)


def test_merging_preserves_hard_boundaries(
    *, prepared: regions.GradientRings, tmp_path: Path
) -> None:
    """Summing two losing probabilities must not move a cell out of its fine region."""
    result = regions.fit_regions(features=prepared, n_regions=3, n_init=1)
    q = pd.DataFrame(np.tile([0.4, 0.35, 0.25], (100, 1)), index=prepared.cells.index)
    result = replace(result, memberships=q, assignments=result.assignments.assign(region=0))
    merged = result.merge({2: 1})
    assert (merged.memberships[1] > merged.memberships[0]).all()
    assert (merged.assignments.region == 0).all()
    assert (merged.assignments.fine_region == 0).all()
    np.testing.assert_allclose(merged.memberships.sum(axis=1), 1)
    np.testing.assert_allclose(merged.assignments.membership_assigned, 0.4)
    assert merged.mapping.to_dict() == {0: 0, 1: 1, 2: 1}
    assert merged.source_fingerprint == result.fit_fingerprint
    merged.save(tmp_path / "reviewed")
    with pytest.raises(ValueError, match="absent"):
        result.merge({99: 0})
    with pytest.raises(ValueError, match="nonnegative"):
        result.merge({1: -1})


def test_fitted_result_ordering_preserves_assignments(prepared: regions.GradientRings) -> None:
    """The public ordering step annotates calls without changing fitted regions."""
    result = regions.fit_regions(features=prepared, n_regions=3, n_init=1)
    before = result.assignments.copy(deep=True)
    order = result.order_laminae(min_cells=1)
    assert sorted(order.regions.lamina_number) == [1, 2, 3]
    annotated = order.annotate(result.assignments.region)
    assert annotated.index.equals(result.assignments.index)
    assert annotated.lamina_number.notna().all()
    pd.testing.assert_frame_equal(before, result.assignments)


class _InputColumns(TypedDict):
    """Named preparation columns used in validation cases."""

    x_col: str
    y_col: str
    celltype_col: str
    sample_col: str


def test_vocabulary_and_small_group_validation(prepared: regions.GradientRings) -> None:
    """Absent known types stay explicit; unknown types and undersized pieces fail."""
    args: _InputColumns = dict(x_col="x", y_col="y", celltype_col="type", sample_col="section")
    p = regions.prepare_gradient_rings(
        cells=prepared.cells, **args, cell_types=("A", "B", "C", "D"), settings=prepared.settings
    )
    assert all((r.D_count == 0).all() for r in p.rings)
    with pytest.raises(ValueError, match="absent from"):
        regions.prepare_gradient_rings(
            cells=prepared.cells, **args, cell_types=("A", "B"), settings=prepared.settings
        )
    with pytest.raises(ValueError, match="small group sizes"):
        regions.prepare_gradient_rings(
            cells=prepared.cells.iloc[:10], **args, settings=prepared.settings
        )
    with pytest.raises(ValueError, match="missing"):
        regions.prepare_gradient_rings(
            cells=prepared.cells.assign(section=None), **args, settings=prepared.settings
        )


def test_reordered_or_modified_features_fail(prepared: regions.GradientRings) -> None:
    """Alignment and input fingerprints catch stale feature caches before fitting."""
    model = regions.fit_regions(features=prepared, n_regions=2, n_init=1).model
    with pytest.raises(ValueError, match="match exactly"):
        model.predict_proba((prepared.rings[0].iloc[::-1], *prepared.rings[1:]))
    with pytest.raises(ValueError, match="vocabulary"):
        model.predict_proba(tuple(r.iloc[:, ::-1] for r in prepared.rings))
    changed = prepared.rings[0].copy()
    changed.iloc[0] = changed.iloc[1]
    with pytest.raises(ValueError, match="changed"):
        regions.fit_regions(
            features=replace(prepared, rings=(changed, *prepared.rings[1:])), n_regions=2
        )
    with pytest.raises(ValueError, match="cell order"):
        regions.fit_regions(
            features=replace(prepared, cells=prepared.cells.iloc[::-1]), n_regions=2
        )


def test_nonconvergence_is_visible() -> None:
    """An iteration-limited best fit warns and records its actual stopping status."""
    rings, _ = _contrasting_rings()
    with pytest.warns(RuntimeWarning, match="max_iter"):
        model = fit_ring_mixture(
            rings=rings, settings=FitSettings(n_regions=2, n_init=1, max_iter=1)
        )
    assert not model.converged
    assert model.starts[0].iterations == 1


def test_saved_model_rejects_invalid_probabilities(tmp_path: Path) -> None:
    """A malformed model must fail before it can generate plausible-looking calls."""
    rings, _ = _contrasting_rings()
    model = fit_ring_mixture(rings=rings, settings=FitSettings(n_regions=2, n_init=1))
    file = tmp_path / "model.json"
    model.save(file)
    data = json.loads(file.read_text())
    data["weights"] = [-1, 2]
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="positive"):
        RingMixture.load(file)


def test_repeated_seed_reproduces_assignments(prepared: regions.GradientRings) -> None:
    """The public seed controls every initialization and resulting membership."""
    a = regions.fit_regions(features=prepared, n_regions=3, seed=41, n_init=2)
    b = regions.fit_regions(features=prepared, n_regions=3, seed=41, n_init=2)
    pd.testing.assert_frame_equal(a.memberships, b.memberships)
    assert a.fit_fingerprint == b.fit_fingerprint


def test_invalid_resolution_and_ring_schema() -> None:
    """Reject impossible resolution, fractional counts, and changing ring populations."""
    rings, _ = _contrasting_rings()
    with pytest.raises(ValueError, match="positive integer"):
        FitSettings(n_regions=True)
    with pytest.raises(ValueError, match="increasing|increase"):
        regions.GradientSettings(ring_ends=(5, 4))
    with pytest.raises(ValueError, match="number of cells"):
        fit_ring_mixture(rings=rings, settings=FitSettings(n_regions=200))
    with pytest.raises(ValueError, match="distinct"):
        fit_ring_mixture(rings=(pd.DataFrame([[1, 1]] * 10),), settings=FitSettings(n_regions=2))
    wrong = rings[0].copy()
    wrong.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="constant"):
        fit_ring_mixture(rings=(wrong,))
