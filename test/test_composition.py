"""Likelihood, optimization, prediction and physical-graph regression checks."""

import torch  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multinomial
from sklearn.metrics import adjusted_rand_score

from lamination_station.composition import fit_composition_mixture
from lamination_station.spatial_graph import build_spatial_structure_graph


def test_mixture_recovers_regions_and_matches_exact_pmf() -> None:
    """Separated count components recover held-out regions and score normalized PMFs."""
    rng = np.random.default_rng(13)
    beta = np.array([[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]])
    labels = np.repeat([0, 1], 80)
    x = pd.DataFrame([rng.multinomial(30, beta[i]) for i in labels], columns=list("abc"))
    fitted = fit_composition_mixture(x, n_components=2, n_init=3, seed=4)
    y = pd.DataFrame([rng.multinomial(25, beta[i]) for i in labels], columns=list("abc"))
    q = fitted.predict_proba(y)
    assert adjusted_rand_score(labels, q.to_numpy().argmax(1)) > 0.95
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert fitted.converged
    assert np.all(np.diff(fitted.objective_history) >= -1e-7)
    expected = logsumexp(
        np.stack(
            [
                multinomial.logpmf(y.to_numpy(), n=25, p=row) + np.log(weight)
                for row, weight in zip(fitted.compositions.to_numpy(), fitted.weights, strict=True)
            ]
        ),
        axis=0,
    )
    np.testing.assert_allclose(fitted.score_samples(y), expected)
    with pytest.raises(ValueError, match="vocabulary"):
        fitted.predict_proba(y[["c", "a", "b"]])


def test_single_component_and_unseen_training_type() -> None:
    """The Dirichlet MAP update has the specified mass and handles unseen types."""
    counts = pd.DataFrame([[5, 0], [15, 0]], columns=["a", "b"])
    fit = fit_composition_mixture(counts, n_components=1, composition_prior_mass=2)
    np.testing.assert_allclose(fit.compositions, [[21 / 22, 1 / 22]])
    assert np.isfinite(fit.score_samples(pd.DataFrame([[0, 10]], columns=["a", "b"]))).all()


@pytest.mark.parametrize("values", [[[0, 0]], [[1.5, 2]], [[-1, 2]], [[np.nan, 2]]])
def test_invalid_counts(values: list[list[float]]) -> None:
    """Missing, fractional and empty neighborhood data are rejected."""
    with pytest.raises(ValueError):
        fit_composition_mixture(pd.DataFrame(values), n_components=1)


def test_graph_uses_physical_edges_and_sample_boundaries() -> None:
    """Distant, same-coordinate cross-section and isolated cells cannot invent edges."""
    cells = pd.DataFrame(
        {
            "x": [0.0, 1.0, 10.0, 0.0, 1.0, 20.0],
            "y": [0.0] * 6,
            "sample": ["a"] * 3 + ["b"] * 3,
            "label": [0, 1, 2, 3, 4, 5],
        }
    )
    graph = build_spatial_structure_graph(
        cells,
        x_col="x",
        y_col="y",
        structure_col="label",
        sample_col="sample",
        max_distance=1.1,
        n_neighbors=2,
    )
    assert set(graph.nodes) == set(range(6))
    assert {tuple(sorted(edge)) for edge in graph.edges} == {(0, 1), (3, 4)}
    assert graph[0][1]["weight"] == 1
    assert graph[0][1]["support"] == 1
    assert graph[0][1]["sample_support"] == 1
    assert graph.degree(2) == 0


def test_internal_edges_reduce_contact_affinity() -> None:
    """Within-region edges contribute to degree normalization but not inter-region support."""
    cells = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0] * 4, "sample": ["a"] * 4, "label": [0, 0, 1, 1]}
    )
    graph = build_spatial_structure_graph(
        cells,
        x_col="x",
        y_col="y",
        structure_col="label",
        sample_col="sample",
        max_distance=1.1,
        n_neighbors=3,
    )
    assert graph[0][1]["support"] == 1
    assert graph[0][1]["weight"] == pytest.approx(1 / 3)


def test_distinct_sample_identifiers_keep_distinct_support() -> None:
    """Integer and text sample IDs with the same spelling remain separate replicates."""
    cells = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0] * 4,
            "sample": [1, 1, "1", "1"],
            "label": [0, 1, 0, 1],
        }
    )
    graph = build_spatial_structure_graph(
        cells,
        x_col="x",
        y_col="y",
        structure_col="label",
        sample_col="sample",
        max_distance=1.1,
        n_neighbors=1,
    )
    assert graph[0][1]["support"] == 2
    assert graph[0][1]["sample_support"] == 2
    assert set(graph[0][1]["samples"]) == {1, "1"}


def test_multistart_seed_and_prediction_reproducibility() -> None:
    """Repeated seeds reproduce fitted compositions and objectives."""
    counts = pd.DataFrame(np.random.default_rng(1).multinomial(20, [0.2, 0.3, 0.5], size=80))
    a = fit_composition_mixture(counts, n_components=3, seed=17)
    b = fit_composition_mixture(counts, n_components=3, seed=17)
    np.testing.assert_array_equal(a.compositions, b.compositions)
    np.testing.assert_array_equal(a.objective_history, b.objective_history)
