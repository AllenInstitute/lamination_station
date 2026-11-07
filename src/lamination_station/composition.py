"""A single-likelihood multinomial mixture for neighborhood compositions."""

import torch  # noqa: F401
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans


def _counts(counts: pd.DataFrame) -> NDArray[np.float64]:
    """Validate nonempty integer count rows with an unambiguous feature vocabulary."""
    x = counts.to_numpy(dtype=float)
    if not counts.index.is_unique or not counts.columns.is_unique:
        raise ValueError("Count row and column identifiers must be unique")
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("Counts must have at least one row and column")
    if not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError("Counts must be finite nonnegative integers")
    if (x.sum(axis=1) == 0).any():
        raise ValueError("Empty neighborhoods have no composition information")
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class CompositionMixture:
    """Fitted MAP mixture with a fixed count vocabulary and reusable prediction.

    Memberships describe discrete component uncertainty, not physical mixtures.
    Overlapping neighborhoods yield a composite likelihood: never interpret these
    memberships as calibrated anatomical confidence or cells as independent replicates.
    """

    compositions: pd.DataFrame
    weights: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    converged: bool
    seed: int

    def _log_joint(self, counts: pd.DataFrame) -> NDArray[np.float64]:
        """Evaluate component log probabilities with the fitted vocabulary."""
        if not counts.columns.equals(self.compositions.columns):
            raise ValueError("Prediction count columns must match the fitted vocabulary and order")
        x = _counts(counts)
        return np.asarray(x @ np.log(self.compositions.to_numpy()).T + np.log(self.weights))

    def predict_proba(self, counts: pd.DataFrame) -> pd.DataFrame:
        """Return full discrete-component responsibilities in input row order."""
        joint = self._log_joint(counts)
        q = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        return pd.DataFrame(q, index=counts.index, columns=self.compositions.index)

    def score_samples(self, counts: pd.DataFrame) -> NDArray[np.float64]:
        """Return conditional count log PMFs, including multinomial coefficients."""
        joint = self._log_joint(counts)
        x = counts.to_numpy(dtype=float)
        coefficient = gammaln(x.sum(axis=1) + 1) - gammaln(x + 1).sum(axis=1)
        return np.asarray(logsumexp(joint, axis=1) + coefficient, dtype=float)


def fit_composition_mixture(
    counts: pd.DataFrame,
    *,
    n_components: int = 8,
    n_init: int = 5,
    max_iter: int = 500,
    tol: float = 1e-7,
    composition_prior_mass: float = 1.0,
    weight_prior_mass: float = 1.0,
    seed: int = 0,
) -> CompositionMixture:
    """Fit one conditional multinomial observation per neighborhood by MAP EM.

    Dirichlet concentrations are 1 + prior_mass / dimension, giving finite
    positive compositions and mixture weights. Multiple Hellinger-space k-means
    starts are selected by the same penalized likelihood used for convergence.
    No coordinates, reference region labels, gradients, or neural guide enter the
    fit. The component count remains a user choice and can be assessed on separate
    sections with score_samples; training objectives cannot select component count.
    """
    x = _counts(counts)
    n, t = x.shape
    if not 1 <= n_components <= n or min(n_init, max_iter) < 1:
        raise ValueError("Require 1 <= n_components <= n_cells and positive iteration counts")
    settings = np.array([tol, composition_prior_mass, weight_prior_mass])
    if not np.isfinite(settings).all() or (settings <= 0).any():
        raise ValueError("Tolerance and prior masses must be positive and finite")
    a, b = composition_prior_mass / t, weight_prior_mass / n_components
    transformed = np.sqrt(x / x.sum(axis=1, keepdims=True))
    rng = np.random.default_rng(seed)
    best: CompositionMixture | None = None
    best_objective = -np.inf
    for _ in range(n_init):
        labels = KMeans(
            n_clusters=n_components, n_init=1, random_state=int(rng.integers(2**31 - 1))
        ).fit_predict(transformed)
        q = np.eye(n_components)[labels]
        history: list[float] = []
        converged = False
        for _iteration in range(max_iter):
            beta = q.T @ x + a
            beta /= beta.sum(axis=1, keepdims=True)
            weights = q.sum(axis=0) + b
            weights /= weights.sum()
            joint = x @ np.log(beta).T + np.log(weights)
            normalizer = logsumexp(joint, axis=1)
            objective = float(
                normalizer.sum() + a * np.log(beta).sum() + b * np.log(weights).sum()
            )
            history.append(objective)
            q = np.exp(joint - normalizer[:, None])
            if len(history) > 1 and abs(history[-1] - history[-2]) <= tol * (1 + abs(history[-2])):
                converged = True
                break
        if history[-1] > best_objective:
            best_objective = history[-1]
            best = CompositionMixture(
                compositions=pd.DataFrame(beta, columns=counts.columns),
                weights=np.asarray(weights, dtype=float),
                objective_history=np.asarray(history, dtype=float),
                converged=converged,
                seed=seed,
            )
    assert best is not None
    return best
