"""Diagnose information loss, covariates, annotation geometry, and clustering objectives."""

import torch  # noqa: F401

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    homogeneity_completeness_v_measure,
)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC

from context_experiments import get_section, representation, vocabulary
from incremental_domains import features
from incremental_models import fit_mixture
from lamination_station.preprocessing import calculate_comp_grads

ROOT = Path("/scratch/lamination_station_ceiling")
OUT = Path("/results/lamination_station_ceiling")
ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = ["development", "test", "fresh1", "fresh2", "confirmation1", "confirmation2"]
SPECIES = ["human", "macaque", "mouse"]


def cached(name, function):
    path = ROOT / (name + ".pkl")
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    value = function()
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return value


def nearest(xy, k):
    distance, raw = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    ix = np.stack([row[row != i][:k] for i, row in enumerate(raw)])
    distance = np.stack(
        [d[row != i][:k] for i, (row, d) in enumerate(zip(raw, distance, strict=True))]
    )
    return distance, ix


def adjusted_neighbors(s):
    """Reconstruct original neighbors and verify counts exactly against stored inputs."""
    ref = s["ref"]
    xy = ref[["x", "y"]].to_numpy(dtype=float)
    center = xy - xy.mean(0)
    scaled = center / np.sqrt(np.mean(center**2))
    distance, ix = nearest(scaled, 90)
    prep = calculate_comp_grads(
        ref.copy(), "x", "y", "type", near_k=30, far_k=90, sample_col="section"
    )
    gradient = prep[2]
    norm = np.linalg.norm(gradient, axis=1)
    direction = gradient / (norm[:, None] + 0.01)
    displacement = scaled[ix] - scaled[:, None]
    uv = displacement / (distance[:, :, None] + 0.01)
    penalty = np.abs(np.einsum("nkd,nd->nk", uv, direction)) * norm[:, None]
    order = np.argsort(distance + penalty, axis=1)
    neighbors = np.take_along_axis(ix, order, axis=1)
    types = pd.Categorical(ref.type)
    onehot = np.eye(len(types.categories))[types.codes]
    for k in [15, 30, 60]:
        counts = pd.DataFrame(
            onehot[neighbors[:, :k]].sum(1),
            index=ref.index,
            columns=[t + "_count" for t in types.categories],
        )
        np.testing.assert_array_equal(
            counts.reindex(columns=s["counts"][k].columns, fill_value=0), s["counts"][k]
        )
    return neighbors


def prepare():
    obs = pd.read_parquet("/scratch/lamination_station_fixes/three_species_obs.parquet")
    protocol = dict(
        scope="Diagnostic, exploratory analysis of18 previously examined spinal sections. No new held-out performance claim.",
        truth="Only direct polygon labels; full-section labels used for geometry diagnostics and separate fixed focal labels for comparable ARI.",
        inference="Annotation-aware probes diagnose information/objective limitations; they are not proposed discovery models.",
        density="Log neuronal8/30/60NN density and scale ratios; median-centered within section because units/platform/capture differ. All-cell features use full typed cell pools.",
        boundary="Distance to nearest differently labeled neuron is a point-based boundary proximity surrogate, not a polygon-edge distance.",
        annotation_limit="No independent second annotator or histology adjudication is available in the analyzed cache; model disagreement cannot establish erroneous annotation.",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    manifest = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)

            def compute():
                ref = s["ref"]
                xy = ref[["x", "y"]].to_numpy()
                raw = obs[obs._section == ref.section.iloc[0]].copy()
                assert raw.index.is_unique
                y = (
                    raw.rexed_lamina.astype("string")
                    .where(raw.rexed_lamina_source == "polygon")
                    .reindex(ref.index)
                )
                types = ref.type.astype(str).to_numpy()
                dist, ix = nearest(xy, 90)
                adjusted = adjusted_neighbors(s)
                finite = raw.consensus_manual_group_propagated.notna() & np.isfinite(
                    raw[["_plot_x", "_plot_y"]]
                ).all(axis=1)
                pool = raw[finite]
                poolxy = pool[["_plot_x", "_plot_y"]].to_numpy()
                focal = pool.index.get_indexer(ref.index)
                assert (focal >= 0).all()
                full_dist, full_ix = NearestNeighbors(n_neighbors=101).fit(poolxy).kneighbors(xy)
                full_neighbor = np.stack(
                    [row[row != f][:100] for f, row in zip(focal, full_ix, strict=True)]
                )
                full_distance = np.stack(
                    [
                        d[row != f][:100]
                        for f, row, d in zip(focal, full_ix, full_dist, strict=True)
                    ]
                )
                cats = pd.Categorical(pool.consensus_manual_group_propagated)
                fullcounts = {}
                for k in [30, 100]:
                    rr = np.repeat(np.arange(len(ref)), k)
                    cc = cats.codes[full_neighbor[:, :k]].ravel()
                    matrix = sparse.csr_matrix(
                        (np.ones(len(rr)), (rr, cc)), shape=(len(ref), len(cats.categories))
                    ).toarray()
                    fullcounts[k] = pd.DataFrame(matrix, index=ref.index, columns=cats.categories)
                cov = np.log(
                    np.maximum(np.column_stack([dist[:, 7], dist[:, 29], dist[:, 59]]), 1e-9)
                )
                cov = cov - np.median(cov, axis=0)
                cov = np.column_stack([cov, cov[:, 2] - cov[:, 0]])
                full_cov = np.log(np.maximum(full_distance[:, [29, 99]], 1e-9))
                full_cov -= np.median(full_cov, axis=0)
                codes = pd.Categorical(y).codes
                valid = codes >= 0
                boundary = np.full(len(ref), np.nan)
                for code in np.unique(codes[valid]):
                    inside = codes == code
                    outside = valid & (~inside)
                    boundary[inside] = (
                        NearestNeighbors(n_neighbors=1)
                        .fit(xy[outside])
                        .kneighbors(xy[inside])[0][:, 0]
                    )
                return dict(
                    y=y,
                    types=types,
                    distance=dist,
                    neighbors=ix,
                    adjusted=adjusted,
                    cov=cov,
                    full_cov=full_cov,
                    full_counts=fullcounts,
                    boundary=boundary,
                    n_all=len(raw),
                    n_typed=len(pool),
                    n_neurons=len(ref),
                    neuron_fraction_100=(
                        pool.Neighborhood.to_numpy()[full_neighbor] == "neuron"
                    ).mean(1),
                )

            d = cached("data_" + s["name"], compute)
            manifest.append(
                dict(
                    name=s["name"],
                    species=species,
                    section=str(s["ref"].section.iloc[0]),
                    n_all=d["n_all"],
                    n_typed=d["n_typed"],
                    n_neurons=d["n_neurons"],
                    n_polygon=int(d["y"].notna().sum()),
                )
            )
            print("PREPARED", s["name"], flush=True)
    pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)


def get_data(s):
    return cached("data_" + s["name"], lambda: None)


def baseline(s):
    path = (
        Path("/scratch/lamination_station_context")
        / f"prediction_{s['name']}_8_gradient_ring_seed17.pkl"
    )
    with path.open("rb") as handle:
        return pickle.load(handle).argmax(1)


def mapping(y, pred):
    classes, codes = np.unique(y.astype(str), return_inverse=True)
    clusters = np.unique(pred)
    table = np.zeros((len(classes), len(clusters)), int)
    np.add.at(table, (codes, np.searchsorted(clusters, pred)), 1)
    r, c = linear_sum_assignment(-table)
    return {clusters[j]: classes[i] for i, j in zip(r, c, strict=True)}, table, classes, clusters


def reference_merge(y, k):
    """Achievable annotation-aware coarse partition, not an absolute ARI upper bound."""
    classes, codes = np.unique(y, return_inverse=True)
    groups = [{i} for i in range(len(classes))]
    mass = list(np.bincount(codes))
    while len(groups) > k:
        _, a, b = min(
            (mass[a] * mass[b], a, b)
            for a in range(len(groups))
            for b in range(a + 1, len(groups))
        )
        groups[a] |= groups[b]
        mass[a] += mass[b]
        del groups[b]
        del mass[b]
    assignment = {j: i for i, g in enumerate(groups) for j in g}
    return np.array([assignment[c] for c in codes])


def geometry():
    records = []
    strata = []
    lamina = []
    pairs = []
    cells = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            ref = s["ref"]
            pred = baseline(s)
            meta = s["meta"].dropna(subset=["label"])
            focal = ref.index.get_indexer(meta.index)
            truth = meta.label.astype(str).to_numpy()
            map_, tab, classes, clusters = mapping(truth, pred[focal])
            named = np.array([map_.get(v, "__unmatched") for v in pred])
            codes = pd.Categorical(d["y"]).codes
            valid = codes >= 0
            ratio = d["boundary"] / d["distance"][:, 29]
            h, c, v = homogeneity_completeness_v_measure(truth, pred[focal])
            record = dict(
                name=s["name"],
                species=species,
                ari=adjusted_rand_score(truth, pred[focal]),
                n_classes=len(classes),
                homogeneity=h,
                completeness=c,
                reference_merge8_ari=adjusted_rand_score(truth, reference_merge(truth, 8)),
                median_r30=float(np.median(d["distance"][:, 29])),
                density90_10=float(
                    (
                        np.quantile(d["distance"][:, 29], 0.9)
                        / np.quantile(d["distance"][:, 29], 0.1)
                    )
                    ** 2
                ),
                fraction_focal_labeled=len(focal) / len(ref),
                fraction_r30_crosses_opposite=float(np.mean(ratio[focal] < 1)),
                fraction_original_neurons_labeled=float(valid.mean()),
            )
            for kind, neighbors in [("ordinary", d["neighbors"]), ("gradient", d["adjusted"])]:
                for k in [15, 30, 60]:
                    neigh = codes[neighbors[:, :k]]
                    known = neigh >= 0
                    same = ((neigh == codes[:, None]) & known).sum(1)
                    purity = np.divide(
                        same, known.sum(1), out=np.full(len(ref), np.nan), where=known.sum(1) > 0
                    )
                    record[f"{kind}_purity{k}"] = float(np.nanmean(purity[focal]))
                    if kind == "gradient":
                        cells.append(
                            pd.DataFrame(
                                dict(
                                    name=s["name"],
                                    cell_id=ref.index[focal],
                                    lamina=truth,
                                    k=k,
                                    purity=purity[focal],
                                    boundary_ratio=ratio[focal],
                                    log_density=-2 * d["cov"][focal, 1],
                                    correct=named[focal] == truth,
                                )
                            )
                        )
            records.append(record)
            density = pd.qcut(-2 * d["cov"][focal, 1], 4, labels=False, duplicates="drop")
            for kind, groups in [
                ("density_quartile", density),
                (
                    "boundary_distance_r30",
                    pd.cut(ratio[focal], [-np.inf, 0.5, 1, 2, np.inf], labels=False),
                ),
            ]:
                for group in np.unique(groups):
                    keep = np.asarray(groups) == group
                    strata.append(
                        dict(
                            name=s["name"],
                            species=species,
                            kind=kind,
                            group=int(group),
                            n=int(keep.sum()),
                            ari=adjusted_rand_score(truth[keep], pred[focal][keep]),
                            accuracy=float(np.mean(named[focal][keep] == truth[keep])),
                        )
                    )
            for i, label in enumerate(classes):
                keep = truth == label
                bestf = max(
                    2 * tab[i, j] / max(tab[i].sum() + tab[:, j].sum(), 1)
                    for j in range(len(clusters))
                )
                n = tab[i].sum()
                p = tab[i] / n
                lamina.append(
                    dict(
                        name=s["name"],
                        species=species,
                        lamina=label,
                        n=int(n),
                        best_cluster_f1=bestf,
                        dominant_cluster_fraction=float(p.max()),
                        median_boundary_ratio=float(np.nanmedian(ratio[focal][keep])),
                        median_r30=float(np.median(d["distance"][focal][keep, 29])),
                        fraction_near_boundary=float(np.mean(ratio[focal][keep] < 1)),
                    )
                )
            # Count pairwise merge errors and split errors without sampling cell pairs.
            for i, a in enumerate(classes):
                for j, b in enumerate(classes):
                    if j <= i:
                        continue
                    merged = int(np.sum(tab[i] * tab[j]))
                    possible = int(tab[i].sum() * tab[j].sum())
                    pairs.append(
                        dict(
                            name=s["name"],
                            species=species,
                            a=a,
                            b=b,
                            merged_pairs=merged,
                            total_cross_pairs=possible,
                            merge_rate=merged / possible,
                        )
                    )
            print("GEOMETRY", s["name"], record["ari"], record["gradient_purity30"], flush=True)
    for name, rows in [
        ("geometry", records),
        ("error_strata", strata),
        ("lamina_geometry", lamina),
        ("merge_pairs", pairs),
    ]:
        pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
    pd.concat(cells).to_parquet(OUT / "cell_diagnostics.parquet", index=False)


def enrich_features(s, kind):
    columns = vocabulary([s])
    base = representation(s, columns, "gradient") / np.sqrt(3)
    d = get_data(s)

    def normalize(x):
        x = x - x.mean(0)
        return x / np.sqrt(np.mean(np.sum(x * x, axis=1)) + 1e-10)

    if kind == "base":
        return base
    if kind == "density":
        return np.column_stack([base, 0.5 * normalize(d["cov"])])
    if kind == "fullcell":
        extra = [np.sqrt(f.to_numpy() / k) / np.sqrt(2) for k, f in d["full_counts"].items()]
        return np.column_stack([base] + extra)
    if kind == "fullcell_density":
        x = enrich_features(s, "fullcell")
        return np.column_stack([x, 0.5 * normalize(np.column_stack([d["cov"], d["full_cov"]]))])
    if kind == "presence":
        counts = s["counts"][30].to_numpy()
        binary = (counts > 0).astype(float)
        return np.column_stack([base, 0.5 * normalize(binary)])
    raise ValueError(kind)


def covariates():
    records = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            meta = s["meta"].dropna(subset=["label"])
            ix = s["ref"].index.get_indexer(meta.index)
            y = meta.label.astype(str).to_numpy()
            for kind in ["base", "density", "fullcell", "fullcell_density", "presence"]:
                x = enrich_features(s, kind)
                for k in [8, 13]:
                    for seed in [17, 29, 43]:
                        pred = cached(
                            f"cluster_{s['name']}_{kind}_K{k}_seed{seed}",
                            lambda: KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
                                x
                            ),
                        )
                        records.append(
                            dict(
                                name=s["name"],
                                species=species,
                                kind=kind,
                                k=k,
                                seed=seed,
                                ari=adjusted_rand_score(y, pred[ix]),
                            )
                        )
            # Fit only within the annotated footprint to test capacity spent outside evaluated tissue.
            x = enrich_features(s, "base")
            mask = d["y"].notna().to_numpy()
            for k in [8, 13]:
                pred = cached(
                    f"footprint_{s['name']}_K{k}",
                    lambda: KMeans(n_clusters=k, n_init=10, random_state=17).fit_predict(x[mask]),
                )
                full = np.full(len(x), -1)
                full[mask] = pred
                records.append(
                    dict(
                        name=s["name"],
                        species=species,
                        kind="annotated_footprint_oracle",
                        k=k,
                        seed=17,
                        ari=adjusted_rand_score(y, full[ix]),
                    )
                )
            pd.DataFrame(records).to_csv(OUT / "covariates.csv", index=False)
            print("COVARIATES", s["name"], flush=True)


def within_section():
    """Annotation-aware probes of feature sufficiency; not independent validation."""
    records = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            valid = d["y"].notna().to_numpy()
            ids = np.flatnonzero(valid)
            y = d["y"].iloc[ids].astype(str).to_numpy()
            xy = s["ref"][["x", "y"]].to_numpy()
            base = enrich_features(s, "base")
            # Three interleaved spatial tile folds; local neighborhoods may overlap folds.
            scaled = (xy - xy.min(0)) / np.maximum(np.ptp(xy, axis=0), 1e-9)
            tiles = np.minimum((scaled * 6).astype(int), 5)
            block = (tiles[:, 0] + 2 * tiles[:, 1]) % 3
            random = np.random.default_rng(731).integers(0, 3, len(xy))
            for scheme, folds in [("random_cells", random), ("spatial_tiles", block)]:
                for kind in ["composition", "composition_density", "coordinates_only"]:
                    x = (
                        base
                        if kind == "composition"
                        else enrich_features(s, "density")
                        if kind == "composition_density"
                        else scaled
                    )
                    predicted = np.full(len(y), "__unassigned", dtype=object)
                    for fold in range(3):
                        train = folds[ids] != fold
                        test = ~train
                        if not test.any():
                            continue
                        classifier = (
                            KNeighborsClassifier(n_neighbors=9, weights="distance")
                            if kind == "coordinates_only"
                            else SVC(C=1, kernel="rbf", class_weight="balanced")
                        )
                        classifier.fit(x[ids[train]], y[train])
                        predicted[test] = classifier.predict(x[ids[test]])
                    records.append(
                        dict(
                            name=s["name"],
                            species=species,
                            scheme=scheme,
                            kind=kind,
                            ari=adjusted_rand_score(y, predicted),
                            macro_f1=f1_score(y, predicted, average="macro", zero_division=0),
                            balanced_accuracy=balanced_accuracy_score(y, predicted),
                        )
                    )
            # Compare the same multinomial likelihood at a reference-based initialization and its EM endpoint.
            rings = features(s, vocabulary([s]))
            classes, encoded = np.unique(y, return_inverse=True)
            q = np.eye(len(classes))[encoded]
            profiles = [(q.T @ r[ids] + 1 / r.shape[1]) for r in rings]
            profiles = [p / p.sum(1, keepdims=True) for p in profiles]

            def ll(profiles, weights):
                return float(
                    logsumexp(
                        sum(r @ np.log(p).T for r, p in zip(rings, profiles, strict=True))
                        + np.log(weights),
                        axis=1,
                    ).sum()
                )

            weights = np.bincount(encoded) + 1 / len(classes)
            weights = weights / weights.sum()
            initial_q = sum(
                r @ np.log(p).T for r, p in zip(rings, profiles, strict=True)
            ) + np.log(weights)
            fixed = cached(
                "reference_initialized_em_" + s["name"],
                lambda: fit_mixture(
                    rings, n_components=len(classes), prior=profiles, prior_cells=0, max_iter=1000
                ),
            )
            records.append(
                dict(
                    name=s["name"],
                    species=species,
                    scheme="same_data_reference_diagnostic",
                    kind="reference_initialized_before_em",
                    ari=adjusted_rand_score(y, initial_q[ids].argmax(1)),
                    log_likelihood=ll(profiles, weights),
                )
            )
            records.append(
                dict(
                    name=s["name"],
                    species=species,
                    scheme="same_data_reference_diagnostic",
                    kind="reference_initialized_after_em",
                    ari=adjusted_rand_score(y, fixed["q"][ids].argmax(1)),
                    log_likelihood=ll(fixed["profiles"], fixed["weights"][0]),
                    converged=fixed["converged"],
                )
            )
            pd.DataFrame(records).to_csv(OUT / "information_probes.csv", index=False)
            print("PROBES", s["name"], flush=True)


def density_sweep():
    """Check smaller covariate-family weights using the same fixed seed and cells."""
    records = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            base = enrich_features(s, "base")
            z = d["cov"] - d["cov"].mean(0)
            z = z / np.sqrt(np.mean((z * z).sum(1)))
            meta = s["meta"].dropna(subset=["label"])
            ix = s["ref"].index.get_indexer(meta.index)
            truth = meta.label.astype(str).to_numpy()
            for weight in [0.1, 0.25]:
                for k in [8, 13]:
                    x = np.column_stack([base, weight * z])
                    pred = cached(
                        f"density_sweep_{s['name']}_{weight}_K{k}",
                        lambda: KMeans(n_clusters=k, n_init=10, random_state=17).fit_predict(x),
                    )
                    records.append(
                        dict(
                            name=s["name"],
                            species=species,
                            weight=weight,
                            k=k,
                            ari=adjusted_rand_score(truth, pred[ix]),
                        )
                    )
            pd.DataFrame(records).to_csv(OUT / "density_sweep.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "geometry", "covariates", "probes", "density_sweep"]
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "geometry": geometry,
        "covariates": covariates,
        "probes": within_section,
        "density_sweep": density_sweep,
    }[args.stage]()
