"""Numerical checks for neighborhood alignment, geometric invariance, and patch isolation."""

import torch  # noqa: F401

import json

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from context_experiments import OUT, context_counts, get_section, spatial_patches, vocabulary
from context_geometry import tangent_counts

checks = []
s = get_section("spinal_human", "development")
columns = vocabulary([s])
counts = context_counts(s, columns)
for x in counts:
    assert np.isfinite(x).all() and (x >= 0).all()
    assert x[:, -1].sum() == 0 and x[:, :-1].sum() > 0
for a, b in zip(counts[:-1], counts[1:], strict=True):
    assert (b >= a).all()
checks.append(
    "Context columns match original _count vocabulary; observed local types never become unknown; cumulative counts are nested."
)

rng = np.random.default_rng(731)
x = rng.normal(size=(40, 5))
graph = sparse.diags([np.ones(39), np.ones(39)], [-1, 1], shape=(40, 40)).tocsr()
means, mass, labels = spatial_patches(x, graph, 7)
reference = AgglomerativeClustering(n_clusters=7, linkage="ward", connectivity=graph).fit_predict(
    x
)
assert adjusted_rand_score(labels, reference) == 1
assert mass.sum() == len(x)
np.testing.assert_allclose(means, np.stack([x[labels == i].mean(0) for i in range(len(mass))]))
checks.append(
    "Custom spatial Ward merges agree with sklearn on a connected chain, and patch means/masses are exact."
)

graph = sparse.block_diag([np.ones((5, 5)) - np.eye(5), np.ones((6, 6)) - np.eye(6)]).tocsr()
_, mass, labels = spatial_patches(rng.normal(size=(11, 3)), graph, 1)
assert len(mass) == 2 and len(set(labels[:5]) & set(labels[5:])) == 0
checks.append(
    "Patch merging stops at disconnected components even when fewer patches are requested."
)

xy = rng.normal(size=(300, 2))
p = rng.dirichlet(np.ones(4), size=300)
codes = rng.integers(0, 4, size=300)
pieces = np.zeros(300, int)
theta = 0.731
rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
for ratio in [1, 3, 6]:
    original = tangent_counts(xy, p, codes, pieces, ratio)
    transformed = tangent_counts(
        (xy @ rotation) * np.array([-1, 1]) + [51, -27], p, codes, pieces, ratio
    )
    for k, a, b in zip([15, 30, 60], original, transformed, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a.sum(1), k)
checks.append(
    "Jacobian-based neighborhoods are invariant to rotation, reflection, and translation for ratios1/3/6."
)

xy = rng.normal(size=(180, 2))
pieces = np.repeat([0, 1], 90)
codes = pieces.copy()
p = np.eye(2)[codes]
for x in tangent_counts(xy, p, codes, pieces, 6):
    assert x[:90, 1].sum() == 0 and x[90:, 0].sum() == 0
checks.append("New tangent contexts never count cells in a different physical component.")

(OUT / "numerical_checks.json").write_text(json.dumps(checks, indent=2))
print("\n".join(checks))
