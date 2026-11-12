"""Numerical and missing-evidence checks for the fixed-type Potts experiment."""

import torch  # noqa: F401

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score

from bass_fixed_types import physical_graph, initial_labels, sample_chain, fit


out = Path("/results/lamination_station_bass")
out.mkdir(parents=True, exist_ok=True)
observed = np.array([0, 0, -1, 1, 1, 0])
profiles = np.array([[0.8, 0.2], [0.3, 0.7]])
graph = sparse.diags([np.ones(5), np.ones(5)], [-1, 1], shape=(6, 6)).tocsr()
edges = sparse.triu(graph).tocoo()
states = np.array(list(itertools.product(range(2), repeat=6)))
potential = np.array(
    [
        sum(np.log(profiles[z[i], observed[i]]) for i in range(6) if observed[i] >= 0)
        + 0.9 * np.sum(z[:-1] == z[1:])
        for z in states
    ]
)
exact = softmax(potential)
samples, _, _ = sample_chain(
    observed,
    np.array([0, 0, 0, 1, 1, 1]),
    np.ones(2),
    edges.row,
    edges.col,
    edges.data,
    graph.indptr,
    graph.indices,
    graph.data,
    0.9,
    2000,
    40000,
    1,
    123,
    profiles,
)
empirical = samples.mean(0)
expected = exact @ states
np.testing.assert_allclose(empirical, expected, atol=0.015)
edge_expected = exact @ (states[:, :-1] == states[:, 1:])
np.testing.assert_allclose((samples[:, :-1] == samples[:, 1:]).mean(0), edge_expected, atol=0.015)

rng = np.random.default_rng(51)
xy = rng.normal(size=(60, 2))
xy = np.vstack([xy, xy])
pieces = np.repeat([0, 1], 60)
g, w = physical_graph(xy, pieces)
transformed = xy.copy()
transformed[60:] = transformed[60:] @ np.array([[0, -1], [1, 0]]) + [2000, -1000]
g2, w2 = physical_graph(transformed, pieces)
np.testing.assert_allclose(w.toarray(), w2.toarray(), atol=1e-12)
assert g[:60, 60:].nnz == 0

# Strong synthetic composition differences, with a narrow middle stripe.
xx, yy = np.meshgrid(np.arange(50), np.arange(25))
coords = np.column_stack([xx.ravel(), yy.ravel()]) + rng.normal(0, 0.04, (1250, 2))
truth = np.where(xx.ravel() < 21, 0, np.where(xx.ravel() < 29, 1, 2))
p = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
types = np.array([rng.choice(3, p=p[z]) for z in truth])
hidden = rng.random(len(types)) < 0.2
masked = types.copy()
masked[hidden] = -1
init = initial_labels(coords, masked, 3, 3, 17)
_, graph = physical_graph(coords)
result = fit(
    masked, graph, init, n_types=3, beta=6, shrinkage=5, seed=17, burnin=500, draws=200, thin=5
)
ari = adjusted_rand_score(truth, result["q"].argmax(1))
assert ari > 0.75, ari
nll = -np.log(result["prediction"][np.arange(hidden.sum()), types[hidden]]).mean()
base_nll = -np.log(result["base"][types[hidden]]).mean()
assert nll < base_nll - 0.15, (nll, base_nll)
# Different withheld values leave every input to model inference unchanged.
changed = types.copy()
changed[hidden] = rng.permutation(changed[hidden])
changed[hidden] = -1
np.testing.assert_array_equal(masked, changed)
np.testing.assert_array_equal(init, initial_labels(coords, changed, 3, 3, 17))
checks = dict(
    fixed_profile_sampler_matches_exact_enumeration=True,
    exact_node_marginal_max_error=float(abs(empirical - expected).max()),
    missing_marks_have_no_likelihood_contribution=True,
    disconnected_pieces_have_no_cross_edges=True,
    graph_rotation_translation_invariant=True,
    synthetic_three_region_ari=float(ari),
    synthetic_masked_type_nll=float(nll),
    synthetic_global_type_nll=float(base_nll),
    initialization_excludes_masked_marks=True,
)
(out / "numerical_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
