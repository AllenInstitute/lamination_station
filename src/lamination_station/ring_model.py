"""The benchmarked mixture of disjoint, gradient-adjusted composition rings."""

import torch  # noqa: F401

import json
import warnings
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from lamination_station.composition import _counts


def positive_integer(*, value: int, name: str) -> None:
    """Reject fractional, boolean, and nonpositive count settings."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class FitSettings:
    """Explicit resolution and reproducible MAP optimization settings."""

    n_regions: int = 8
    n_init: int = 5
    max_iter: int = 300
    tol: float = 1e-7
    seed: int = 17

    def __post_init__(self) -> None:
        """Validate settings before any expensive initialization."""
        for name in ("n_regions", "n_init", "max_iter"):
            positive_integer(value=getattr(self, name), name=name)
            object.__setattr__(self, name, int(getattr(self, name)))
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "tol", float(self.tol))


@dataclass(frozen=True, kw_only=True)
class FitStart:
    """Optimization diagnostics for one independently initialized fit."""

    initial_seed: int
    objective: float
    iterations: int
    converged: bool


def validate_rings(rings: tuple[pd.DataFrame, ...]) -> tuple[NDArray[np.float64], ...]:
    """Require aligned integer count rings with constant positive ring sizes."""
    if not rings:
        raise ValueError("At least one disjoint count ring is required")
    arrays = []
    for ring in rings:
        if not ring.index.equals(rings[0].index) or not ring.columns.equals(rings[0].columns):
            raise ValueError(
                "Ring rows and vocabulary columns must match exactly, including order"
            )
        x = _counts(ring)
        if not np.all(x.sum(axis=1) == x.sum(axis=1)[0]):
            raise ValueError("Each ring must have a constant number of neighbors across cells")
        arrays.append(x)
    return tuple(arrays)


class _SavedSettings(TypedDict):
    n_regions: int
    n_init: int
    max_iter: int
    tol: float
    seed: int


class _SavedStart(TypedDict):
    initial_seed: int
    objective: float
    iterations: int
    converged: bool


class _SavedModel(TypedDict):
    schema_version: int
    settings: _SavedSettings
    columns: list[str]
    ring_sizes: list[int]
    profiles: list[list[list[float]]]
    weights: list[float]
    objective_history: list[float]
    starts: list[_SavedStart]
    best_start: int


@dataclass(frozen=True, kw_only=True)
class RingMixture:
    """Fitted product-multinomial mixture with full optimization diagnostics.

    Rings are disjoint within each cell; neighborhoods still overlap between cells.
    Never interpret responsibilities as calibrated anatomical probabilities.
    """

    profiles: tuple[pd.DataFrame, ...]
    weights: NDArray[np.float64]
    ring_sizes: tuple[int, ...]
    objective_history: NDArray[np.float64]
    starts: tuple[FitStart, ...]
    best_start: int
    settings: FitSettings

    @property
    def converged(self) -> bool:
        """Report convergence of the selected start, not merely any start."""
        return self.starts[self.best_start].converged

    def _joint(self, rings: tuple[pd.DataFrame, ...]) -> NDArray[np.float64]:
        """Validate the prediction schema and compute component log scores."""
        arrays = validate_rings(rings)
        if len(arrays) != len(self.profiles):
            raise ValueError("Prediction must use the fitted number of rings")
        if not rings[0].columns.equals(self.profiles[0].columns):
            raise ValueError("Prediction vocabulary and column order must match the fitted model")
        if tuple(int(x[0].sum()) for x in arrays) != self.ring_sizes:
            raise ValueError("Prediction ring sizes must match the fitted neighborhood scales")
        joint = np.tile(np.log(self.weights), (len(rings[0]), 1))
        for x, profile in zip(arrays, self.profiles, strict=True):
            joint += x @ np.log(profile.to_numpy()).T
        return np.asarray(joint, dtype=float)

    def predict_proba(self, rings: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
        """Return responsibilities in the input cell order without dropping types."""
        joint = self._joint(rings)
        q = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        return pd.DataFrame(q, index=rings[0].index, columns=self.profiles[0].index)

    def score_samples(self, rings: tuple[pd.DataFrame, ...]) -> NDArray[np.float64]:
        """Return conditional count log PMFs including every ring's coefficient."""
        scores = logsumexp(self._joint(rings), axis=1)
        for ring in rings:
            x = ring.to_numpy(dtype=float)
            scores += gammaln(x.sum(axis=1) + 1) - gammaln(x + 1).sum(axis=1)
        return np.asarray(scores, dtype=float)

    def save(self, path: str | Path) -> None:
        """Write a versioned JSON model without overwriting an existing file."""
        if not all(isinstance(c, str) for c in self.profiles[0].columns):
            raise ValueError("Saved models require string vocabulary columns")
        data = dict(
            schema_version=1,
            settings=asdict(self.settings),
            columns=self.profiles[0].columns.tolist(),
            ring_sizes=list(self.ring_sizes),
            profiles=[p.to_numpy().tolist() for p in self.profiles],
            weights=self.weights.tolist(),
            objective_history=self.objective_history.tolist(),
            starts=[asdict(s) for s in self.starts],
            best_start=self.best_start,
        )
        with Path(path).open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> "RingMixture":
        """Load and validate a JSON model without executing serialized Python code."""
        try:
            data = cast(_SavedModel, json.loads(Path(path).read_text()))
            if data["schema_version"] != 1:
                raise ValueError("Unsupported model schema version")
            settings = FitSettings(**data["settings"])
            columns = data["columns"]
            if not columns or not all(isinstance(c, str) for c in columns):
                raise ValueError("Invalid vocabulary")
            if len(set(columns)) != len(columns):
                raise ValueError("Duplicate vocabulary columns")
            sizes = tuple(data["ring_sizes"])
            for size in sizes:
                positive_integer(value=size, name="ring size")
            profiles = tuple(np.asarray(p, dtype=float) for p in data["profiles"])
            if not profiles or len(profiles) != len(sizes):
                raise ValueError("Invalid ring schema")
            weights = np.asarray(data["weights"], dtype=float)
            for array in (*profiles, weights):
                if not np.isfinite(array).all() or (array <= 0).any():
                    raise ValueError("Model probabilities must be finite and positive")
                if not np.allclose(array.sum(axis=-1), 1, atol=1e-10, rtol=0):
                    raise ValueError("Model probabilities must sum to one")
            if weights.shape != (settings.n_regions,) or any(
                p.shape != (settings.n_regions, len(columns)) for p in profiles
            ):
                raise ValueError("Invalid model dimensions")
            history = np.asarray(data["objective_history"], dtype=float)
            starts = tuple(FitStart(**s) for s in data["starts"])
            best = data["best_start"]
            if history.ndim != 1 or not len(history) or not np.isfinite(history).all():
                raise ValueError("Invalid optimization history")
            if len(starts) != settings.n_init or not 0 <= best < len(starts):
                raise ValueError("Invalid initialization diagnostics")
            if any(not np.isfinite(s.objective) or s.iterations < 1 for s in starts):
                raise ValueError("Invalid initialization diagnostics")
            if starts[best].iterations != len(history) or starts[best].objective != history[-1]:
                raise ValueError("Optimization diagnostics do not match the fitted model")
            return cls(
                profiles=tuple(pd.DataFrame(p, columns=columns) for p in profiles),
                weights=weights,
                ring_sizes=sizes,
                objective_history=history,
                starts=starts,
                best_start=best,
                settings=settings,
            )
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            raise ValueError(f"Invalid gradient-ring model: {error}") from error


def fit_ring_mixture(
    *, rings: tuple[pd.DataFrame, ...], settings: FitSettings = FitSettings()
) -> RingMixture:
    """Fit the benchmarked MAP objective, keeping all starts and the best fit.

    Each ring uses Dirichlet concentration 1 + 1/n_types, and mixture weights use
    1 + 1/n_regions. Resolution is a requested component count, not a guarantee
    that every component wins a hard assignment. Training likelihood must not be
    used to choose resolution across different component counts.
    """
    arrays = validate_rings(rings)
    n, t = arrays[0].shape
    k = settings.n_regions
    if k > n:
        raise ValueError("n_regions cannot exceed the number of cells")
    transformed = np.concatenate(
        [np.sqrt(x / x.sum(axis=1, keepdims=True)) for x in arrays], axis=1
    )
    if len(np.unique(transformed, axis=0)) < k:
        raise ValueError("n_regions exceeds the number of distinct ring-composition observations")
    rng = np.random.default_rng(settings.seed)
    starts: list[FitStart] = []
    best_objective = -np.inf
    best_history: NDArray[np.float64] = np.empty(0, dtype=float)
    best_profiles: tuple[NDArray[np.float64], ...] = ()
    best_weights = np.empty(0, dtype=float)
    best_start = 0
    for start in range(settings.n_init):
        initial_seed = int(rng.integers(2**31 - 1))
        labels = (
            KMeans(n_clusters=k, n_init=1, random_state=initial_seed)
            .fit(transformed, sample_weight=np.ones(n))
            .labels_
        )
        q = np.eye(k)[labels]
        history: list[float] = []
        converged = False
        for _ in range(settings.max_iter):
            weights = q.sum(axis=0) + 1 / k
            weights /= weights.sum()
            profiles = tuple(q.T @ x + 1 / t for x in arrays)
            profiles = tuple(p / p.sum(axis=1, keepdims=True) for p in profiles)
            joint = np.tile(np.log(weights), (n, 1))
            penalty = np.log(weights).sum() / k
            for x, p in zip(arrays, profiles, strict=True):
                joint += x @ np.log(p).T
                penalty += np.log(p).sum() / t
            normalizer = logsumexp(joint, axis=1)
            objective = float(normalizer.sum() + penalty)
            if not np.isfinite(objective):
                raise FloatingPointError("Nonfinite gradient-ring objective")
            if history and objective < history[-1] - 1e-8 * (1 + abs(history[-1])):
                raise RuntimeError("Gradient-ring EM objective decreased unexpectedly")
            history.append(objective)
            q = np.exp(joint - normalizer[:, None])
            if len(history) > 1 and abs(history[-1] - history[-2]) < settings.tol * (
                1 + abs(history[-2])
            ):
                converged = True
                break
        starts.append(
            FitStart(
                initial_seed=initial_seed,
                objective=objective,
                iterations=len(history),
                converged=converged,
            )
        )
        if objective > best_objective:
            best_objective, best_start = objective, start
            best_history = np.asarray(history, dtype=float)
            best_profiles, best_weights = profiles, weights
    model = RingMixture(
        profiles=tuple(pd.DataFrame(p, columns=rings[0].columns) for p in best_profiles),
        weights=best_weights,
        ring_sizes=tuple(int(x[0].sum()) for x in arrays),
        objective_history=best_history,
        starts=tuple(starts),
        best_start=best_start,
        settings=settings,
    )
    if not model.converged:
        warnings.warn(
            "Selected gradient-ring fit reached max_iter without convergence; inspect diagnostics or increase max_iter.",
            RuntimeWarning,
            stacklevel=2,
        )
    return model
