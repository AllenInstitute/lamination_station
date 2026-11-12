"""Experimental domain models for the incremental analysis, not a public package API.

Inputs are count arrays and physical graphs, never anatomical reference labels.
Neighborhood overlap makes the count likelihood composite; responsibilities are
model scores rather than calibrated anatomical probabilities.
"""

import torch  # noqa: F401

import numpy as np
from scipy import sparse
from scipy.special import logsumexp, softmax, xlogy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def validate_rings(rings):
    """Require aligned, nonnegative count matrices with constant ring sizes."""
    assert len(rings) > 0
    shape = rings[0].shape
    for x in rings:
        assert x.shape == shape and np.isfinite(x).all() and (x >= 0).all()
        assert np.allclose(x, np.round(x)) and np.all(x.sum(1) > 0)
        assert np.allclose(x.sum(1), x.sum(1)[0])


def log_evidence(rings, profiles):
    """Evaluate the multiscale composition evidence, omitting constant coefficients."""
    return sum(x @ np.log(beta).T for x, beta in zip(rings, profiles, strict=True))


def fit_mixture(
    rings,
    *,
    groups=None,
    seed=17,
    n_components=8,
    n_init=5,
    prior=None,
    prior_cells=None,
    max_iter=300,
    case_weights=None,
):
    """Fit shared profiles with sample-specific prevalences or adapt a learned prior.

    prior_cells is the number of pseudo-cells per component per ring. Infinity fixes
    the profiles. Samples contribute equal total weight when groups is supplied.
    Finite adaptation uses a Dirichlet MAP prior centered on training profiles.
    """
    validate_rings(rings)
    n, t = rings[0].shape
    groups = np.zeros(n, int) if groups is None else np.asarray(groups, dtype=int)
    _, group = np.unique(groups, return_inverse=True)
    ng = group.max() + 1
    mass = np.bincount(group)
    row_weight = n / ng / mass[group]
    if case_weights is not None:
        row_weight *= case_weights
    features = np.concatenate([np.sqrt(x / x.sum(1, keepdims=True)) for x in rings], 1)
    rng = np.random.default_rng(seed)
    best = None
    best_objective = -np.inf
    for _start in range(n_init if prior is None else 1):
        if prior is None:
            labels = (
                KMeans(
                    n_clusters=n_components, n_init=1, random_state=int(rng.integers(2**31 - 1))
                )
                .fit(features, sample_weight=row_weight)
                .labels_
            )
            q = np.eye(n_components)[labels]
        else:
            q = softmax(log_evidence(rings, prior), axis=1)
        history = []
        for iteration in range(max_iter):
            weighted = q * row_weight[:, None]
            weights = np.stack([weighted[group == g].sum(0) + 1 / n_components for g in range(ng)])
            weights /= weights.sum(1, keepdims=True)
            profiles = []
            penalty = np.log(weights).sum() / n_components
            for r, x in enumerate(rings):
                if prior is not None and np.isinf(prior_cells):
                    beta = prior[r].copy()
                else:
                    pseudocount = (
                        np.full((n_components, t), 1 / t)
                        if prior is None
                        else prior_cells * x.sum(1)[0] * prior[r] + 1 / t
                    )
                    beta = weighted.T @ x + pseudocount
                    beta /= beta.sum(1, keepdims=True)
                    penalty += np.sum(pseudocount * np.log(beta))
                profiles.append(beta)
            unary = log_evidence(rings, profiles) + np.log(weights[group])
            norm = logsumexp(unary, axis=1)
            objective = float(row_weight @ norm + penalty)
            history.append(objective)
            q = softmax(unary, axis=1)
            if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-7 * (1 + abs(history[-2])):
                break
        assert np.isfinite(q).all()
        assert np.min(np.diff(history), initial=0) > -1e-6 * (1 + abs(history[-1]))
        if objective > best_objective:
            best_objective = objective
            best = dict(
                q=q,
                profiles=profiles,
                weights=weights,
                unary=unary,
                history=np.asarray(history),
                converged=iteration < max_iter - 1,
            )
    return best


def spatial_graphs(xy, composition, *, pieces=None, k=12):
    """Build piece-isolated physical and boundary-gated symmetric graphs.

    Gating is applied after physical degree normalization so weakening an edge is
    not canceled by renormalization. Distance caps and bandwidths are per piece.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    pieces = np.zeros(n, int) if pieces is None else np.asarray(pieces)
    rows, cols = [], []
    gate_rows, gate_cols, gate_values = [], [], []
    hellinger = np.sqrt(composition / composition.sum(1, keepdims=True))
    for piece in np.unique(pieces):
        ids = np.flatnonzero(pieces == piece)
        if len(ids) < 2:
            continue
        use_k = min(k, len(ids) - 1)
        distance, index = NearestNeighbors(n_neighbors=use_k + 1).fit(xy[ids]).kneighbors(xy[ids])
        neighbor = np.array([row[row != i][:use_k] for i, row in enumerate(index)])
        distance = np.array(
            [d[row != i][:use_k] for i, (row, d) in enumerate(zip(index, distance))]
        )
        rr = np.repeat(np.arange(len(ids)), use_k)
        cc = neighbor.ravel()
        keep = distance.ravel() <= np.quantile(distance[:, -1], 0.95)
        graph = sparse.coo_matrix(
            (np.ones(keep.sum()), (rr[keep], cc[keep])), shape=(len(ids), len(ids))
        ).tocsr()
        graph = graph.minimum(graph.T).tocoo()
        gr, gc = ids[graph.row], ids[graph.col]
        d2 = 0.5 * ((hellinger[gr] - hellinger[gc]) ** 2).sum(1)
        bandwidth = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        rows.extend(gr)
        cols.extend(gc)
        gate_rows.extend(gr)
        gate_cols.extend(gc)
        gate_values.extend(np.exp(-d2 / bandwidth))
    physical = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    gated = sparse.coo_matrix((gate_values, (gate_rows, gate_cols)), shape=(n, n)).tocsr()
    degree = np.asarray(physical.sum(1)).ravel()
    scale = sparse.diags(1 / np.sqrt(np.maximum(degree, 1)))
    return physical, scale @ physical @ scale, scale @ gated @ scale


def learn_compatibility(graphs, probabilities, *, pseudocount=20.0, confidence=0.8):
    """Estimate regularized off-diagonal contact enrichment, balancing samples.

    The null uses region abundance at graph endpoints. Only confident assignments
    contribute. Unobserved regions in a sample have neutral entries. This is a fixed
    training-derived potential, not a fitted normalized contact probability.
    """
    estimates, supports = [], []
    for graph, q in zip(graphs, probabilities, strict=True):
        n, k = q.shape
        hard = np.eye(k)[q.argmax(1)] * (q.max(1) >= confidence)[:, None]
        contact = hard.T @ (graph @ hard)
        endpoint = contact.sum(1)
        np.fill_diagonal(contact, 0)
        null = np.outer(endpoint, endpoint)
        np.fill_diagonal(null, 0)
        supported = (endpoint[:, None] > 0) & (endpoint[None, :] > 0)
        if contact.sum() == 0 or null.sum() == 0:
            estimates.append(np.zeros((k, k)))
            supports.append(np.zeros((k, k)))
            continue
        null /= null.sum()
        observed = (contact + pseudocount * null) / (contact.sum() + pseudocount)
        ratio = np.divide(observed, null, out=np.ones_like(null), where=null > 0)
        affinity = np.clip(np.log(np.maximum(ratio, 1e-12)), -2, 2)
        np.fill_diagonal(affinity, 0)
        estimates.append(affinity * supported)
        supports.append(supported)
    support = np.sum(supports, axis=0)
    result = np.divide(
        np.sum(estimates, axis=0), support, out=np.zeros_like(estimates[0]), where=support > 0
    )
    assert np.allclose(result, result.T)
    return result


def refine(
    unary,
    smooth_graph,
    physical_graph,
    *,
    strength=0.0,
    compatibility=None,
    contact_strength=0.0,
    max_iter=200,
):
    """Optimize a fixed-unary mean-field objective with monotone damped updates.

    Same-region smoothing and off-diagonal compatibility have separate graphs and
    strengths. Outputs remain approximate scores, not calibrated probabilities.
    """
    unary = unary - logsumexp(unary, axis=1, keepdims=True)
    q = softmax(unary, axis=1)
    compatibility = np.zeros((q.shape[1], q.shape[1])) if compatibility is None else compatibility
    assert np.allclose(compatibility, compatibility.T)

    def message(v):
        return strength * (smooth_graph @ v) + contact_strength * (
            (physical_graph @ v) @ compatibility
        )

    def energy(v):
        return float(np.sum(v * unary) - xlogy(v, v).sum() + 0.5 * np.sum(v * message(v)))

    history = [energy(q)]
    for iteration in range(max_iter):
        proposal = softmax(unary + message(q), axis=1)
        step = 0.5
        for _ in range(25):
            candidate = (1 - step) * q + step * proposal
            value = energy(candidate)
            if value >= history[-1] - 1e-9:
                break
            step *= 0.5
        change = np.max(np.abs(candidate - q))
        q = candidate
        history.append(value)
        if change < 1e-5:
            break
    assert np.min(np.diff(history), initial=0) >= -1e-7
    return dict(q=q, history=np.array(history), converged=iteration < max_iter - 1)


def dm_evidence(rings, profiles, concentration):
    """Evaluate Dirichlet-multinomial evidence using exact integer-count products.

    Terms constant across components are omitted. Concentration is fixed and shared
    by components, so these constants also do not affect within-fit optimization.
    """
    n = len(rings[0])
    k = len(profiles[0])
    evidence = np.zeros((n, k))
    for x, beta in zip(rings, profiles, strict=True):
        counts = x.astype(int)
        maximum = int(counts.max())
        types = np.arange(x.shape[1])[None, :]
        for component in range(k):
            log_terms = np.log(concentration * beta[component, :, None] + np.arange(maximum))
            lookup = np.column_stack([np.zeros(x.shape[1]), np.cumsum(log_terms, axis=1)])
            evidence[:, component] += lookup[types, counts].sum(1)
    return evidence


def dm_mean_map(tails, initial, *, concentration, prior):
    """Maximize the concave DM mean-profile objective on the probability simplex.

    The diagonal Hessian permits an equality-constrained Newton direction. A
    positivity bound and Armijo line search preserve the simplex and ascent.
    """
    beta = initial.copy()
    offsets = np.arange(tails.shape[1])

    def value(p):
        return float(
            np.sum(tails * np.log(concentration * p[:, None] + offsets)) + prior * np.log(p).sum()
        )

    current = value(beta)
    for iteration in range(200):
        denominator = concentration * beta[:, None] + offsets
        gradient = concentration * np.sum(tails / denominator, axis=1) + prior / beta
        hessian = -(concentration**2) * np.sum(tails / denominator**2, axis=1) - prior / beta**2
        multiplier = np.sum(gradient / hessian) / np.sum(1 / hessian)
        direction = (multiplier - gradient) / hessian
        gain = float(gradient @ direction)
        if gain <= 1e-10 * (1 + abs(current)):
            return beta, True
        negative = direction < 0
        step = (
            min(1.0, float(np.min(-0.99 * beta[negative] / direction[negative])))
            if negative.any()
            else 1.0
        )
        for _ in range(40):
            candidate = beta + step * direction
            candidate /= candidate.sum()
            proposed = value(candidate)
            if proposed >= current + 1e-4 * step * gain - 1e-10:
                break
            step *= 0.5
        beta, current = candidate, proposed
    return beta, False


def fit_dm(rings, initial, *, concentration, max_iter=80):
    """Fit an overdispersed mixture with numerical MAP updates for mean profiles.

    Initialization comes from a separately fitted multinomial mixture. The M-step
    optimizes the actual Dirichlet-multinomial objective, not a multinomial update.
    """
    validate_rings(rings)
    profiles = [beta.copy() for beta in initial["profiles"]]
    q = initial["q"].copy()
    n, k = q.shape
    t = rings[0].shape[1]
    prior = 1 / t
    history = []
    optimizer_failures = 0
    histograms = []
    for x in rings:
        maximum = int(x.max())
        row, kind = np.nonzero(x)
        category = kind * (maximum + 1) + x[row, kind].astype(int)
        indicator = sparse.coo_matrix(
            (np.ones(len(row)), (category, row)), shape=(t * (maximum + 1), n)
        ).tocsr()
        histograms.append((indicator, maximum))
    for iteration in range(max_iter):
        weights = q.sum(0) + 1 / k
        weights /= weights.sum()
        penalty = np.log(weights).sum() / k
        for ring, x in enumerate(rings):
            indicator, maximum = histograms[ring]
            histogram = (indicator @ q).reshape(t, maximum + 1, k)
            all_tails = np.cumsum(histogram[:, ::-1], axis=1)[:, ::-1][:, 1:]
            # Weighted survival counts allow an exact small M-step objective.
            for component in range(k):
                tails = all_tails[:, :, component]

                beta, converged = dm_mean_map(
                    tails, profiles[ring][component], concentration=concentration, prior=prior
                )
                if not converged:
                    optimizer_failures += 1
                profiles[ring][component] = beta
            penalty += prior * np.log(profiles[ring]).sum()
        unary = dm_evidence(rings, profiles, concentration) + np.log(weights)
        objective_value = float(logsumexp(unary, axis=1).sum() + penalty)
        history.append(objective_value)
        q = softmax(unary, axis=1)
        if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-7 * (1 + abs(history[-2])):
            break
    assert np.isfinite(q).all()
    assert np.min(np.diff(history), initial=0) >= -1e-5 * (1 + abs(history[-1]))
    return dict(
        q=q,
        profiles=profiles,
        weights=weights,
        unary=unary,
        history=np.array(history),
        converged=iteration < max_iter - 1,
        optimizer_failures=optimizer_failures,
        concentration=concentration,
    )
