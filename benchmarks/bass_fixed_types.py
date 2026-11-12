"""Experimental fixed-type categorical–Potts model, inspired by BASS.

This is an analysis implementation, not a reimplementation of the BASS package.
Only coordinates and observed cell types enter inference. A type code of -1
means missing evidence, not an extra biological type. Region profiles have an
empirical-Bayes Dirichlet prior centered on observed overall type frequencies.
The graph is geometric; no same-cell-type edge filtering is applied.
"""

import torch  # noqa: F401

import numpy as np
from numba import njit
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def physical_graph(xy, pieces=None, neighbors=8):
    """Make a symmetric, distance-capped graph separately inside each tissue piece."""
    xy = np.asarray(xy, dtype=float)
    pieces = np.zeros(len(xy), int) if pieces is None else np.asarray(pieces)
    rows, cols = [], []
    for piece in np.unique(pieces):
        ids = np.flatnonzero(pieces == piece)
        if len(ids) < 2:
            continue
        k = min(neighbors, len(ids) - 1)
        dist, ix = NearestNeighbors(n_neighbors=k + 1).fit(xy[ids]).kneighbors(xy[ids])
        distances = np.array([d[j != i][:k] for i, (d, j) in enumerate(zip(dist, ix))])
        indices = np.array([j[j != i][:k] for i, j in enumerate(ix)])
        keep = distances <= np.quantile(distances[:, -1], 0.95)
        rows.extend(np.repeat(ids, k)[keep.ravel()])
        cols.extend(ids[indices[keep]])
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xy), len(xy)))
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(1)).ravel()
    scale = sparse.diags(1 / np.sqrt(np.maximum(degree, 1)))
    return graph, (scale @ graph @ scale).tocsr()


def initial_labels(xy, observed, n_types, k, seed, pieces=None):
    """Initialize with local compositions computed using observed types only."""
    pieces = np.zeros(len(xy), int) if pieces is None else np.asarray(pieces)
    blocks = np.zeros((len(xy), 2 * n_types))
    base = np.bincount(observed[observed >= 0], minlength=n_types) + 0.5
    base = base / base.sum()
    for piece in np.unique(pieces):
        ids = np.flatnonzero(pieces == piece)
        train = ids[observed[ids] >= 0]
        if len(train) == 0:
            blocks[ids] = np.tile(np.sqrt(base), 2)
            continue
        nn = NearestNeighbors(n_neighbors=min(31, len(train))).fit(xy[train])
        neighbors = train[nn.kneighbors(xy[ids], return_distance=False)]
        # Exclude self where observed; the labels used here are only initialization.
        for row, i in enumerate(ids):
            near = neighbors[row][neighbors[row] != i]
            for j, size in enumerate([15, 30]):
                v = np.bincount(observed[near[:size]], minlength=n_types) + base
                blocks[i, j * n_types : (j + 1) * n_types] = np.sqrt(v / v.sum())
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(blocks)


@njit(cache=True)
def draw_logweights(logp):
    p = np.exp(logp - np.max(logp))
    u = np.random.random() * np.sum(p)
    accum = 0.0
    for j in range(len(p)):
        accum += p[j]
        if u < accum:
            return j
    return len(p) - 1


@njit(cache=True)
def root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def update_labels(z, observed, logprofiles, rr, cc, weight, indptr, indices, values, beta):
    """External-field Swendsen–Wang followed by a random-order Gibbs sweep."""
    n, k = len(z), len(logprofiles)
    parent = np.arange(n)
    size = np.ones(n, np.int64)
    for e in range(len(rr)):
        a, b = rr[e], cc[e]
        if z[a] == z[b] and np.random.random() < -np.expm1(-beta * weight[e]):
            a, b = root(parent, a), root(parent, b)
            if a != b:
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
    fields = np.zeros((n, k))
    for i in range(n):
        a = root(parent, i)
        parent[i] = a
        if observed[i] >= 0:
            fields[a] += logprofiles[:, observed[i]]
    newlabels = np.zeros(n, np.int64)
    for i in range(n):
        if parent[i] == i:
            newlabels[i] = draw_logweights(fields[i])
    for i in range(n):
        z[i] = newlabels[parent[i]]
    for i in np.random.permutation(n):
        local = np.zeros(k)
        if observed[i] >= 0:
            local += logprofiles[:, observed[i]]
        for e in range(indptr[i], indptr[i + 1]):
            local[z[indices[e]]] += beta * values[e]
        z[i] = draw_logweights(local)
    return z


@njit(cache=True)
def sample_chain(
    observed,
    initial,
    prior,
    rr,
    cc,
    weight,
    indptr,
    indices,
    values,
    beta,
    burnin,
    draws,
    thin,
    seed,
    fixed_profiles,
):
    np.random.seed(seed)
    z = initial.copy()
    k = int(np.max(initial)) + 1
    t = len(prior)
    zs = np.zeros((draws, len(z)), np.int16)
    ps = np.zeros((draws, k, t))
    trace = np.zeros((draws, 3))
    profiles = np.zeros((k, t))
    for step in range(burnin + draws * thin):
        old = z.copy()
        counts = np.zeros((k, t))
        for i in range(len(z)):
            if observed[i] >= 0:
                counts[z[i], observed[i]] += 1
        if len(fixed_profiles):
            profiles = fixed_profiles.copy()
        else:
            for r in range(k):
                for c in range(t):
                    profiles[r, c] = max(np.random.gamma(counts[r, c] + prior[c]), 1e-300)
                profiles[r] /= np.sum(profiles[r])
        logs = np.log(profiles)
        z = update_labels(z, observed, logs, rr, cc, weight, indptr, indices, values, beta)
        if step >= burnin and (step - burnin) % thin == 0:
            j = (step - burnin) // thin
            zs[j] = z
            ps[j] = profiles
            for i in range(len(z)):
                if observed[i] >= 0:
                    trace[j, 0] += logs[z[i], observed[i]]
            for e in range(len(rr)):
                trace[j, 1] += weight[e] * (z[rr[e]] == z[cc[e]])
            trace[j, 2] = np.mean(z != old)
    return zs, ps, trace


def fit(
    observed,
    graph,
    initial,
    *,
    n_types,
    beta,
    shrinkage,
    seed,
    burnin=1000,
    draws=200,
    thin=5,
    fixed_profiles=None,
):
    """Sample profiles and labels jointly; retain label-invariant mark predictions."""
    observed = np.asarray(observed, dtype=np.int64)
    initial = np.asarray(initial, dtype=np.int64)
    k = initial.max() + 1
    assert np.array_equal(np.unique(initial), np.arange(k))
    assert (observed >= -1).all() and (observed < n_types).all()
    graph = graph.tocsr()
    assert (graph - graph.T).nnz == 0 and (graph.data >= 0).all()
    edges = sparse.triu(graph, k=1).tocoo()
    base = np.bincount(observed[observed >= 0], minlength=n_types) + 0.5
    base = base / base.sum()
    prior = shrinkage * base
    zs, ps, trace = sample_chain(
        observed,
        initial,
        prior,
        edges.row,
        edges.col,
        edges.data,
        graph.indptr,
        graph.indices,
        graph.data,
        beta,
        burnin,
        draws,
        thin,
        seed,
        np.empty((0, 0)) if fixed_profiles is None else fixed_profiles,
    )
    # Align label names only, without altering partitions or type predictions.
    anchor = zs[len(zs) // 2]
    q = np.zeros((len(initial), k))
    mean_profiles = np.zeros((k, n_types))
    hidden = np.flatnonzero(observed < 0)
    prediction = np.zeros((len(hidden), n_types))
    half_prediction = np.zeros((2, len(hidden), n_types))
    for j, (z, p) in enumerate(zip(zs, ps)):
        overlap = np.bincount(anchor.astype(int) * k + z, minlength=k * k).reshape(k, k)
        r, c = linear_sum_assignment(-overlap)
        mapping = np.empty(k, int)
        mapping[c] = r
        np.add.at(q, (np.arange(len(z)), mapping[z]), 1)
        mean_profiles[mapping] += p
        prediction += p[z[hidden]]
        half_prediction[int(j >= len(zs) // 2)] += p[z[hidden]]
    q /= len(zs)
    mean_profiles /= len(zs)
    prediction /= len(zs)
    half_prediction[0] /= len(zs) // 2
    half_prediction[1] /= len(zs) - len(zs) // 2
    np.testing.assert_allclose(q.sum(1), 1, atol=1e-12)
    return dict(
        q=q,
        profiles=mean_profiles,
        base=base,
        hidden=hidden,
        prediction=prediction,
        half_prediction=half_prediction,
        trace=trace,
        samples=zs[:: max(1, len(zs) // 20)],
        initial=initial,
        beta=beta,
        shrinkage=shrinkage,
        seed=seed,
        burnin=burnin,
        draws=draws,
        thin=thin,
    )
