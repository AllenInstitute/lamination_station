"""Numerical and disconnected-tissue checks for the experimental analysis models."""

import torch

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import gammaln, softmax
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from incremental_models import (
    dm_evidence,
    dm_mean_map,
    fit_dm,
    fit_mixture,
    learn_compatibility,
    log_evidence,
    refine,
    spatial_graphs,
)

torch.set_num_threads(2)
threadpool_limits(2)
rng = np.random.default_rng(123)
profiles = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
truth = np.repeat([0, 1], 100)
rings = [
    np.array([rng.multinomial(n, profiles[k]) for k in truth], dtype=float) for n in [15, 15, 30]
]
fit = fit_mixture(rings, n_components=2, n_init=3)
assert adjusted_rand_score(truth, fit["q"].argmax(1)) > 0.98
assert fit["converged"]

# An incomplete target piece contains just one of the training regions.
single = [
    np.array([rng.multinomial(n, profiles[0]) for _ in range(100)], dtype=float)
    for n in [15, 15, 30]
]
fixed = fit_mixture(single, n_components=2, prior=fit["profiles"], prior_cells=np.inf)
assert len(np.unique(fixed["q"].argmax(1))) == 1
for actual, expected in zip(fixed["profiles"], fit["profiles"], strict=True):
    np.testing.assert_array_equal(actual, expected)
np.testing.assert_allclose(
    fixed["q"],
    softmax(log_evidence(single, fixed["profiles"]) + np.log(fixed["weights"][0]), axis=1),
)

# Pieces can overlap in slide coordinates without introducing inter-piece edges.
xy = rng.normal(size=(100, 2))
counts = sum(rings)
pieces = np.repeat([0, 1], 100)
coordinates = np.vstack([xy, xy])
graphs = spatial_graphs(coordinates, counts, pieces=pieces)
rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
transformed = coordinates.copy()
transformed[100:] = transformed[100:] @ rotation + [1000.0, -300.0]
rotated = spatial_graphs(transformed, counts, pieces=pieces)
for graph, other in zip(graphs, rotated, strict=True):
    np.testing.assert_allclose(graph.toarray(), other.toarray(), atol=1e-12)
    np.testing.assert_allclose(graph.toarray(), graph.T.toarray(), atol=1e-12)
    assert graph[:100, 100:].nnz == 0
physical, uniform, gated = graphs
assert np.all(gated.toarray() <= uniform.toarray() + 1e-12)
zero = refine(fit["unary"], gated, uniform)
np.testing.assert_allclose(zero["q"], fit["q"], atol=1e-12)

# Train an undirected relationship from A-B-C contacts, not label co-assignment.
labels = np.repeat([0, 1, 2], 20)
chain = sparse.diags([np.ones(59), np.ones(59)], [-1, 1], shape=(60, 60)).tocsr()
contact = learn_compatibility([chain], [np.eye(3)[labels]])
assert contact[0, 1] > contact[0, 2] and contact[1, 2] > contact[0, 2]
np.testing.assert_allclose(contact, contact.T)
np.testing.assert_array_equal(np.diag(contact), 0)
permutation = np.array([2, 0, 1])
permuted = learn_compatibility([chain], [np.eye(3)[labels][:, permutation]])
np.testing.assert_allclose(permuted, contact[np.ix_(permutation, permutation)])

# Graph influence must not change a different disconnected component.
refined = refine(
    fit["unary"],
    gated,
    uniform,
    strength=4.0,
    compatibility=np.array([[0.0, -0.2], [-0.2, 0.0]]),
    contact_strength=1.0,
)
changed_unary = fit["unary"].copy()
changed_unary[:100] = rng.normal(size=(100, 2))
changed = refine(
    changed_unary,
    gated,
    uniform,
    strength=4.0,
    compatibility=np.array([[0.0, -0.2], [-0.2, 0.0]]),
    contact_strength=1.0,
)
np.testing.assert_allclose(refined["q"][100:], changed["q"][100:], atol=1e-5)
assert np.diff(refined["history"]).min() >= -1e-7
np.testing.assert_allclose(refined["q"].sum(1), 1.0)

concentration = 30.0
evidence = dm_evidence(rings, fit["profiles"], concentration)
reference = np.zeros_like(evidence)
for x, beta in zip(rings, fit["profiles"], strict=True):
    for component in range(2):
        alpha = concentration * beta[component]
        reference[:, component] += (gammaln(x + alpha) - gammaln(alpha)).sum(1)
np.testing.assert_allclose(evidence, reference, atol=1e-10)
dm = fit_dm(rings, fit, concentration=concentration, max_iter=40)
assert np.diff(dm["history"]).min(initial=0) >= -1e-6
np.testing.assert_allclose(dm["q"].sum(1), 1.0)

tails = np.array([[100.0, 80.0, 20.0], [70.0, 40.0, 5.0], [20.0, 10.0, 0.0]])
mean, success = dm_mean_map(tails, np.full(3, 1 / 3), concentration=30.0, prior=1 / 3)
assert success


def objective(p):
    return -float(np.sum(tails * np.log(30 * p[:, None] + np.arange(3))) + np.log(p).sum() / 3)


independent = minimize(
    objective,
    np.full(3, 1 / 3),
    method="SLSQP",
    bounds=[(1e-10, 1.0)] * 3,
    constraints={"type": "eq", "fun": lambda p: p.sum() - 1},
    options={"ftol": 1e-10, "maxiter": 1000},
)
assert independent.success
np.testing.assert_allclose(mean, independent.x, atol=1e-6)
assert dm["optimizer_failures"] == 0

out = Path("/results/lamination_station_incremental")
out.mkdir(parents=True, exist_ok=True)
checks = {
    "synthetic_compositions_recovered": True,
    "missing_region_allowed": True,
    "fixed_profiles_unchanged": True,
    "prediction_matches_likelihood": True,
    "piece_rotation_translation_invariant": True,
    "no_cross_piece_edges": True,
    "boundary_gate_does_not_amplify_edges": True,
    "zero_strength_recovers_mixture": True,
    "learned_adjacency_recovers_chain": True,
    "adjacency_label_permutation_equivariant": True,
    "disconnected_piece_inference_independent": True,
    "refinement_objective_monotone": True,
    "dm_likelihood_matches_gamma_definition": True,
    "dm_map_objective_monotone": True,
    "dm_newton_matches_independent_constrained_solver": True,
}
(out / "numerical_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
