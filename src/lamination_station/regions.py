"""Prepare, fit, compare, and review gradient-ring region calls."""

import torch  # noqa: F401

import hashlib
import importlib.metadata
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import adjusted_rand_score

from lamination_station.preprocessing import calculate_comp_grads
from lamination_station.ring_model import (
    FitSettings,
    RingMixture,
    fit_ring_mixture,
    positive_integer,
    validate_rings,
)


@dataclass(frozen=True, kw_only=True)
class GradientSettings:
    """Fixed neighborhood scales, independent of clustering resolution."""

    ring_ends: tuple[int, ...] = (15, 30, 60)
    far_k: int = 90
    eps: float = 0.01
    grad_clip: float | None = 0.9
    elev_weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject ambiguous scales before constructing neighborhoods."""
        if not isinstance(self.ring_ends, tuple) or not self.ring_ends:
            raise ValueError("ring_ends must be a nonempty tuple of increasing integers")
        positive_integer(value=self.far_k, name="far_k")
        for end in self.ring_ends:
            positive_integer(value=end, name="ring endpoint")
        if tuple(sorted(set(self.ring_ends))) != self.ring_ends or self.ring_ends[-1] > self.far_k:
            raise ValueError("ring_ends must increase strictly and not exceed far_k")
        if (
            not np.isfinite([self.eps, self.elev_weight]).all()
            or self.eps <= 0
            or self.elev_weight < 0
        ):
            raise ValueError("eps must be positive and elev_weight nonnegative, both finite")
        if self.grad_clip is not None and not 0 <= self.grad_clip <= 1:
            raise ValueError("grad_clip must be None or in [0, 1]")
        object.__setattr__(self, "ring_ends", tuple(int(v) for v in self.ring_ends))
        object.__setattr__(self, "far_k", int(self.far_k))
        object.__setattr__(self, "eps", float(self.eps))
        object.__setattr__(self, "elev_weight", float(self.elev_weight))
        if self.grad_clip is not None:
            object.__setattr__(self, "grad_clip", float(self.grad_clip))


@dataclass(frozen=True, kw_only=True)
class GradientRings:
    """Reusable, row-aligned features and provenance for resolution comparisons."""

    rings: tuple[pd.DataFrame, ...]
    cells: pd.DataFrame
    gradients: pd.DataFrame
    settings: GradientSettings
    cell_types: tuple[str, ...]
    source_columns: tuple[str, ...]
    sample_col: str | None
    piece_col: str | None
    input_fingerprint: str
    feature_fingerprint: str


def _input_digest(
    *,
    cells: pd.DataFrame,
    columns: tuple[str, ...],
    vocabulary: tuple[str, ...],
    settings: GradientSettings,
) -> str:
    """Fingerprint the actual model inputs and geometry settings."""
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(cells[list(columns)], index=True).to_numpy().tobytes()
    )
    digest.update(
        json.dumps(
            dict(columns=columns, vocabulary=vocabulary, settings=asdict(settings)), sort_keys=True
        ).encode()
    )
    return digest.hexdigest()


def _ring_digest(rings: tuple[pd.DataFrame, ...]) -> str:
    """Detect accidental edits to prepared count values or their vocabulary."""
    digest = hashlib.sha256()
    for ring in rings:
        digest.update(pd.util.hash_pandas_object(ring, index=True).to_numpy().tobytes())
        digest.update(json.dumps(ring.columns.tolist()).encode())
    return digest.hexdigest()


def _validate_features(features: GradientRings) -> None:
    """Reject mutated cached features rather than attaching stale provenance."""
    if any(not ring.index.equals(features.cells.index) for ring in features.rings):
        raise ValueError("Prepared feature rows no longer match the source cell order")
    if (
        _ring_digest(features.rings) != features.feature_fingerprint
        or _input_digest(
            cells=features.cells,
            columns=features.source_columns,
            vocabulary=features.cell_types,
            settings=features.settings,
        )
        != features.input_fingerprint
    ):
        raise ValueError(
            "Prepared inputs or counts have changed; prepare the gradient rings again"
        )


def prepare_gradient_rings(
    *,
    cells: pd.DataFrame,
    x_col: str,
    y_col: str,
    celltype_col: str,
    sample_col: str | None,
    piece_col: str | None = None,
    cell_types: tuple[str, ...] | None = None,
    settings: GradientSettings = GradientSettings(),
) -> GradientRings:
    """Compute benchmark-compatible disjoint rings without crossing supplied groups.

    Pass sample_col=None explicitly for a single coordinate frame. piece_col
    isolates multiple tissue pieces inside each section; piece IDs may repeat
    between sections. Every group needs more than far_k cells. Unknown types are
    rejected when a vocabulary is supplied, rather than silently discarded.
    """
    columns = [x_col, y_col, celltype_col]
    columns.extend(c for c in (sample_col, piece_col) if c is not None)
    if len(set(columns)) != len(columns):
        raise ValueError("Coordinate, type, sample, and piece columns must be distinct")
    if not cells.columns.is_unique or not cells.index.is_unique or len(cells) < 2:
        raise ValueError("Cells require unique row/column identifiers and at least two rows")
    if cells.index.to_frame(index=False).isna().any().any():
        raise ValueError("Cell identifiers must not be missing")
    missing = set(columns) - set(cells.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if cells[columns].isna().any().any():
        raise ValueError("Coordinates, cell types, sample IDs, and piece IDs must not be missing")
    if not np.isfinite(cells[[x_col, y_col]].to_numpy(dtype=float)).all():
        raise ValueError("Coordinates must be finite")
    labels = cells[celltype_col]
    if not all(isinstance(value, str) and value for value in labels):
        raise ValueError(
            "Cell-type labels must be nonempty strings; convert numeric IDs explicitly"
        )
    vocabulary = tuple(sorted(set(labels))) if cell_types is None else tuple(cell_types)
    if not vocabulary or not all(isinstance(c, str) and c for c in vocabulary):
        raise ValueError("cell_types must contain nonempty strings")
    if len(set(vocabulary)) != len(vocabulary) or "__unseen" in vocabulary:
        raise ValueError("cell_types must be unique and cannot contain reserved '__unseen'")
    unknown = set(labels) - set(vocabulary)
    if unknown:
        raise ValueError(f"Cell types absent from the supplied vocabulary: {sorted(unknown)}")
    grouping = [c for c in (sample_col, piece_col) if c is not None]
    groups = (
        pd.factorize(pd.MultiIndex.from_frame(cells[grouping]), sort=False)[0]
        if grouping
        else np.zeros(len(cells), dtype=int)
    )
    sizes = np.bincount(groups)
    if np.any(sizes <= settings.far_k):
        small = sizes[sizes <= settings.far_k].tolist()
        raise ValueError(
            f"Every sample/piece needs more than far_k={settings.far_k} cells; small group sizes: {small}. Choose smaller scales explicitly or analyze these pieces separately."
        )
    # Use internal column names to avoid modifying or overwriting source metadata.
    work = pd.DataFrame(
        {
            "x": cells[x_col],
            "y": cells[y_col],
            "type": pd.Categorical(labels, categories=vocabulary),
            "group": groups,
        },
        index=cells.index,
    )
    rings: list[pd.DataFrame] = []
    previous: pd.DataFrame | None = None
    gradients = pd.DataFrame()
    for end in settings.ring_ends:
        prepared = calculate_comp_grads(
            work,
            "x",
            "y",
            "type",
            near_k=end,
            far_k=settings.far_k,
            eps=settings.eps,
            grad_clip=settings.grad_clip,
            elev_weight=settings.elev_weight,
            sample_col="group",
        )
        cumulative = prepared[5].copy()
        # Preserve the benchmark's explicit zero-count vocabulary slot.
        cumulative["__unseen_count"] = 0
        rings.append(cumulative if previous is None else cumulative - previous)
        previous = cumulative
        if gradients.empty:
            gradients = pd.DataFrame(
                prepared[2], index=cells.index, columns=["gradient_x", "gradient_y"]
            )
            gradients["gradient_norm"] = np.linalg.norm(prepared[2], axis=1)
    validate_rings(tuple(rings))
    return GradientRings(
        rings=tuple(rings),
        cells=cells.copy(),
        gradients=gradients,
        settings=settings,
        cell_types=vocabulary,
        source_columns=tuple(columns),
        sample_col=sample_col,
        piece_col=piece_col,
        input_fingerprint=_input_digest(
            cells=cells, columns=tuple(columns), vocabulary=vocabulary, settings=settings
        ),
        feature_fingerprint=_ring_digest(tuple(rings)),
    )


def _assignments(memberships: pd.DataFrame) -> pd.DataFrame:
    """Summarize uncalibrated membership strength without imposing a rejection cutoff."""
    q = memberships.to_numpy()
    order = np.sort(q, axis=1)
    margin = order[:, -1] - order[:, -2] if q.shape[1] > 1 else np.ones(len(q))
    return pd.DataFrame(
        dict(
            region=memberships.columns.to_numpy()[q.argmax(axis=1)],
            membership_max=q.max(axis=1),
            membership_margin=margin,
            membership_entropy=-xlogy(q, q).sum(axis=1),
        ),
        index=memberships.index,
    )


def _versions() -> dict[str, str]:
    """Record package versions alongside an exported analysis."""
    versions = {}
    for name in ("lamination_station", "numpy", "pandas", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@dataclass(frozen=True, kw_only=True)
class MergedRegions:
    """An explicit review mapping that preserves every original fine assignment."""

    assignments: pd.DataFrame
    memberships: pd.DataFrame
    mapping: pd.Series
    source_fingerprint: str

    def save(self, directory: str | Path) -> None:
        """Export the complete fine-to-coarse map without overwriting existing analyses."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.assignments.to_csv(path / "assignments.csv", index=True)
        self.memberships.to_csv(path / "memberships.csv", index=True)
        self.mapping.to_csv(path / "merge_mapping.csv", index=True)
        (path / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    source_fingerprint=self.source_fingerprint,
                    operation="One-pass mapping of fine hard assignments; summed coarse memberships",
                ),
                indent=2,
            )
        )


@dataclass(frozen=True, kw_only=True)
class RegionResult:
    """Region calls, memberships, fitted profiles, and reproducible provenance."""

    features: GradientRings
    model: RingMixture
    memberships: pd.DataFrame
    assignments: pd.DataFrame

    @property
    def fit_fingerprint(self) -> str:
        """Identify the specific fitted profiles behind a review mapping."""
        digest = hashlib.sha256(self.features.input_fingerprint.encode())
        digest.update(self.features.feature_fingerprint.encode())
        digest.update(json.dumps(asdict(self.model.settings), sort_keys=True).encode())
        digest.update(self.model.weights.tobytes())
        for profile in self.model.profiles:
            digest.update(profile.to_numpy().tobytes())
        return digest.hexdigest()

    @property
    def region_summary(self) -> pd.DataFrame:
        """Include empty components so requested and occupied resolution remain visible."""
        ids = self.memberships.columns
        sizes = self.assignments.region.value_counts().reindex(ids, fill_value=0)
        return pd.DataFrame(
            dict(
                n_cells=sizes,
                membership_mass=self.memberships.sum(axis=0),
                mean_assigned_membership=self.assignments.groupby("region")
                .membership_max.mean()
                .reindex(ids),
            ),
            index=ids.rename("region"),
        )

    def merge(self, mapping: Mapping[int, int]) -> MergedRegions:
        """Merge by a one-pass ID mapping while retaining original hard boundaries.

        Unmapped regions retain their IDs. Coarse probabilities are summed, but
        hard labels are mapped from fine labels: never recompute hard labels by
        taking argmax of the summed probabilities, which can move boundaries.
        """
        ids = [int(v) for v in self.memberships.columns]
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 0
            for pair in mapping.items()
            for v in pair
        ):
            raise ValueError("Merge IDs must be nonnegative integers")
        unknown = set(mapping) - set(ids)
        if unknown:
            raise ValueError(f"Merge keys are absent from the fitted model: {sorted(unknown)}")
        complete = pd.Series({i: int(mapping.get(i, i)) for i in ids}, name="region")
        complete.index.name = "fine_region"
        targets = sorted(set(complete.tolist()))
        merged = pd.DataFrame(
            {
                j: self.memberships.loc[:, complete.index[complete == j]].sum(axis=1)
                for j in targets
            },
            index=self.memberships.index,
        )
        fine = self.assignments.region.copy()
        coarse = fine.map(complete)
        positions = pd.Index(targets).get_indexer(coarse)
        assigned = merged.to_numpy()[np.arange(len(merged)), positions]
        assignments = pd.DataFrame(
            dict(
                fine_region=fine,
                region=coarse,
                membership_assigned=assigned,
                membership_entropy=-xlogy(merged.to_numpy(), merged.to_numpy()).sum(axis=1),
            )
        )
        return MergedRegions(
            assignments=assignments,
            memberships=merged,
            mapping=complete,
            source_fingerprint=self.fit_fingerprint,
        )

    def save(self, directory: str | Path) -> None:
        """Export calls, profiles, model, and manifest into a new analysis directory."""
        _validate_features(self.features)
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.features.cells.loc[:, list(self.features.source_columns)].to_csv(path / "cells.csv")
        self.assignments.to_csv(path / "assignments.csv")
        self.memberships.to_csv(path / "memberships.csv")
        self.features.gradients.to_csv(path / "gradients.csv")
        self.region_summary.to_csv(path / "regions.csv")
        pd.DataFrame([asdict(s) for s in self.model.starts]).to_csv(
            path / "starts.csv", index=False
        )
        pd.Series(self.model.objective_history, name="objective").to_csv(path / "objective.csv")
        for i, profile in enumerate(self.model.profiles):
            profile.to_csv(path / f"profiles_ring_{i + 1}.csv", index_label="region")
        self.model.save(path / "model.json")
        manifest = dict(
            schema_version=1,
            method="gradient-ring multinomial MAP mixture",
            input_fingerprint=self.features.input_fingerprint,
            feature_fingerprint=self.features.feature_fingerprint,
            fit_fingerprint=self.fit_fingerprint,
            n_cells=len(self.assignments),
            requested_regions=self.model.settings.n_regions,
            occupied_regions=int(self.assignments.region.nunique()),
            gradient_settings=asdict(self.features.settings),
            fit_settings=asdict(self.model.settings),
            cell_types=self.features.cell_types,
            source_columns=self.features.source_columns,
            sample_col=self.features.sample_col,
            piece_col=self.features.piece_col,
            converged=self.model.converged,
            best_start=self.model.best_start,
            versions=_versions(),
            interpretation="Membership scores are not calibrated anatomical probabilities; component IDs are not named laminae.",
        )
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))


def fit_regions(
    *,
    features: GradientRings,
    n_regions: int = 8,
    seed: int = 17,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-7,
) -> RegionResult:
    """Fit at an explicit resolution while reusing prepared neighborhood features.

    Larger n_regions allows overclustering, but different resolutions are not a
    nested hierarchy. All supplied cells contribute equally to a pooled fit;
    sample/piece boundaries isolate neighbors, not component identities.
    """
    _validate_features(features)
    model = fit_ring_mixture(
        rings=features.rings,
        settings=FitSettings(
            n_regions=n_regions, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
        ),
    )
    q = model.predict_proba(features.rings)
    return RegionResult(features=features, model=model, memberships=q, assignments=_assignments(q))


@dataclass(frozen=True, kw_only=True)
class ResolutionScan:
    """Fits and descriptive stability across explicitly requested resolutions."""

    fits: Mapping[tuple[int, int], RegionResult]
    summary: pd.DataFrame

    def best(self, *, n_regions: int) -> RegionResult:
        """Select by objective among seeds at one resolution, never across resolutions."""
        candidates = [v for (k, _), v in self.fits.items() if k == n_regions]
        if not candidates:
            raise ValueError(f"Resolution {n_regions} was not fitted")
        return max(candidates, key=lambda v: float(v.model.objective_history[-1]))


def fit_resolutions(
    *,
    features: GradientRings,
    n_regions: Sequence[int],
    seeds: Sequence[int] = (17, 29, 43),
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-7,
) -> ResolutionScan:
    """Compare resolutions and seed stability without choosing an anatomical answer."""
    if (
        len(n_regions) == 0
        or len(seeds) == 0
        or len(set(n_regions)) != len(n_regions)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide nonempty, unique resolutions and seeds")
    # Validate every setting before starting an expensive scan.
    for k in n_regions:
        if k > len(features.cells):
            raise ValueError("n_regions cannot exceed the number of cells")
        for seed in seeds:
            FitSettings(n_regions=k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol)
    fits: dict[tuple[int, int], RegionResult] = {}
    rows = []
    for k in n_regions:
        for seed in seeds:
            fits[k, seed] = fit_regions(
                features=features,
                n_regions=k,
                seed=seed,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
            )
        runs = [fits[k, seed] for seed in seeds]
        agreement = [
            adjusted_rand_score(a.assignments.region, b.assignments.region)
            for a, b in itertools.combinations(runs, 2)
        ]
        rows.append(
            dict(
                n_regions=k,
                n_seeds=len(seeds),
                occupied_min=min(r.assignments.region.nunique() for r in runs),
                occupied_max=max(r.assignments.region.nunique() for r in runs),
                converged_runs=sum(r.model.converged for r in runs),
                mean_seed_ari=float(np.mean(agreement)) if agreement else np.nan,
                min_region_cells=min(int(r.region_summary.n_cells.min()) for r in runs),
            )
        )
    return ResolutionScan(fits=fits, summary=pd.DataFrame(rows).set_index("n_regions"))
