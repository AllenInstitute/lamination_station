"""Counterfactual sampling tests and upstream annotation-quality diagnostics."""

import torch  # noqa: F401

import argparse
import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from ceiling_diagnostics import (
    OUT,
    ROOT,
    SPECIES,
    SPLITS,
    cached,
    get_data,
    get_section,
    baseline,
    mapping,
    nearest,
)
from context_experiments import vocabulary
from incremental_domains import features
from incremental_models import fit_mixture
from lamination_station.preprocessing import calculate_comp_grads


def simulate():
    records = []
    (OUT / "simulation_protocol.json").write_text(
        json.dumps(
            dict(
                scope="Six most recently evaluated spinal sections; exploratory counterfactual diagnosis, not held-out validation.",
                fit="K13, all polygon-labeled reference neurons, same cells and labels in every condition; KMeans10 starts or original ring mixture5 starts; three seeds.",
                actual="Original gradient-adjusted rings restricted to polygon-labeled focal neurons; neighbors include original full neuronal pool.",
                shuffled="Shuffle cell types among neurons within each annotated lamina, preserving type totals and coordinates; unannotated neurons remain unchanged; then recompute original gradient-adjusted neighborhoods.",
                pure60="Draw independent15/15/30 type-count rings from each reference lamina empirical composition; eliminates cross-boundary mixing and within-lamina spatial heterogeneity.",
                pure240="Same as pure60 with60/60/120 type counts; tests finite neighborhood-sampling noise.",
                caution="Pure simulations assume homogeneous, independent cell-type draws conditional on the existing annotations. They are not biological truth, theoretical accuracy bounds, or feasible label-free algorithms.",
            ),
            indent=2,
        )
    )
    for species in SPECIES:
        for split in ["confirmation1", "confirmation2"]:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            valid = d["y"].notna().to_numpy()
            y = d["y"][valid].astype(str).to_numpy()
            columns = vocabulary([s])
            actual = [r[valid] for r in features(s, columns)]
            types = s["ref"].type.astype(str).to_numpy()
            classes = sorted(set(y))
            profile = {}
            lookup = {c.removesuffix("_count"): i for i, c in enumerate(columns[:-1])}
            codes = np.array([lookup[t] for t in types])
            full_labels = d["y"].fillna("__unlabeled").to_numpy()
            for label in classes:
                v = np.bincount(codes[full_labels == label], minlength=len(columns)).astype(float)
                profile[label] = v / v.sum()
            for seed in [17, 29, 43]:

                def make():
                    rng = np.random.default_rng(seed)
                    shuffled = types.copy()
                    for label in classes:
                        ix = np.flatnonzero(full_labels == label)
                        shuffled[ix] = rng.permutation(types[ix])
                    ref = s["ref"].copy()
                    ref["type"] = pd.Categorical(shuffled, categories=s["ref"].type.cat.categories)
                    counts = {
                        k: calculate_comp_grads(
                            ref, "x", "y", "type", near_k=k, far_k=90, sample_col="section"
                        )[5]
                        for k in [15, 30, 60]
                    }
                    tmp = dict(s, ref=ref, counts=counts)
                    arrays = {
                        "actual": actual,
                        "shuffled": [r[valid] for r in features(tmp, columns)],
                    }
                    for key, ns in [("pure60", [15, 15, 30]), ("pure240", [60, 60, 120])]:
                        blocks = []
                        for n in ns:
                            block = np.zeros_like(actual[0])
                            for label in classes:
                                ix = np.flatnonzero(y == label)
                                block[ix] = rng.multinomial(n, profile[label], size=len(ix))
                            blocks.append(block)
                        arrays[key] = blocks
                    return arrays

                arrays = cached(f"simulation_inputs_{s['name']}_seed{seed}", make)
                for kind, rings in arrays.items():
                    x = np.concatenate(
                        [np.sqrt(r / r.sum(1, keepdims=True)) for r in rings], axis=1
                    )
                    for model in ["kmeans", "ring"]:

                        def fit():
                            if model == "kmeans":
                                return KMeans(
                                    n_clusters=13, n_init=10, random_state=seed
                                ).fit_predict(x)
                            return fit_mixture(rings, n_components=13, seed=seed)["q"].argmax(1)

                        pred = cached(
                            f"simulation_prediction_{s['name']}_{kind}_{model}_seed{seed}", fit
                        )
                        records.append(
                            dict(
                                name=s["name"],
                                species=species,
                                condition=kind,
                                model=model,
                                seed=seed,
                                ari=adjusted_rand_score(y, pred),
                            )
                        )
                pd.DataFrame(records).to_csv(OUT / "counterfactuals.csv", index=False)
                print("SIMULATION", s["name"], seed, flush=True)


def quality():
    def read():
        p = "/data/SpinalCord/review_integrated/nnn_clusters/spatial/all_species.allinone.h5ad"
        with h5py.File(p) as f:
            index = pd.MultiIndex.from_arrays(
                [
                    np.asarray(ad.io.read_elem(f["obs/_section"])).astype(str),
                    f["obs/_index"].asstr()[:],
                ]
            )
            fields = [
                "consensus_manual_group_propagated_directly_assigned",
                "consensus_manual_group_propagated_aggregate_probability",
                "consensus_manual_group_propagated_avg_correlation",
            ]
            return pd.DataFrame(
                {
                    key.removeprefix("consensus_manual_group_propagated_"): ad.io.read_elem(
                        f["obs"][key]
                    )
                    for key in fields
                },
                index=index,
            )

    all_qc = cached("source_qc", read)
    records = []
    stratified = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            ref = s["ref"]
            keys = pd.MultiIndex.from_arrays([ref.section.astype(str), ref.index.astype(str)])
            qc = all_qc.reindex(keys).reset_index(drop=True)
            meta = s["meta"].dropna(subset=["label"])
            ix = ref.index.get_indexer(meta.index)
            truth = meta.label.astype(str).to_numpy()
            pred = baseline(s)
            _, contingency, classes, clusters = mapping(truth, pred[ix])
            rr = np.searchsorted(classes, truth)
            cc = np.searchsorted(clusters, pred[ix])
            # Per-cell harmonic mean of B-cubed precision and recall; no label matching.
            correct = 2 * contingency[rr, cc] / (contingency.sum(1)[rr] + contingency.sum(0)[cc])
            molecular_path = (
                ROOT.parent / "lamination_station_context" / f"molecular_{s['name']}.pkl"
            )
            import pickle

            with molecular_path.open("rb") as handle:
                counts = pickle.load(handle)["expression"]
            vals = dict(
                type_probability=qc.aggregate_probability.to_numpy(),
                type_correlation=qc.avg_correlation.to_numpy(),
                log_umi=np.log1p(counts.sum(1)),
                log_density=-2 * d["cov"][:, 1],
                boundary_ratio=d["boundary"] / d["distance"][:, 29],
            )
            records.append(
                dict(
                    name=s["name"],
                    species=species,
                    fraction_direct=float(qc.directly_assigned.mean()),
                    mean_type_probability=float(qc.aggregate_probability.mean()),
                    median_umi=float(np.median(counts.sum(1))),
                )
            )
            for key, x in vals.items():
                values = x[ix]
                for label in sorted(set(truth)):
                    group = (truth == label) & np.isfinite(values)
                    if group.sum() < 20:
                        continue
                    low, high = np.quantile(values[group], [0.25, 0.75])
                    a = group & (values <= low)
                    b = group & (values >= high)
                    if not a.any() or not b.any() or low == high:
                        continue
                    stratified.append(
                        dict(
                            name=s["name"],
                            species=species,
                            lamina=label,
                            covariate=key,
                            n=int(group.sum()),
                            low_score=float(correct[a].mean()),
                            high_score=float(correct[b].mean()),
                            score="cell_bcubed_f1",
                            difference=float(correct[b].mean() - correct[a].mean()),
                        )
                    )
    pd.DataFrame(records).to_csv(OUT / "quality.csv", index=False)
    pd.DataFrame(stratified).to_csv(OUT / "within_lamina_covariates.csv", index=False)
    print("QUALITY", len(records), flush=True)


def boundary_oracle():
    """Prevent cross-label mixing while preserving observed spatial type structure."""
    records = []
    for species in SPECIES:
        for split in ["confirmation1", "confirmation2"]:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            labels = d["y"].fillna("__unlabeled").to_numpy()
            counts = d["y"].value_counts()
            large = counts[counts >= 61].index.to_numpy()
            use = np.isin(labels, large)
            y = labels[use].astype(str)
            xy = s["ref"][["x", "y"]].to_numpy(dtype=float)
            cats = pd.Categorical(s["ref"].type)
            onehot = np.eye(len(cats.categories))[cats.codes]
            oracle = np.zeros((len(xy), 60), int)
            for label in large:
                ix = np.flatnonzero(labels == label)
                _, near = nearest(xy[ix], 60)
                oracle[ix] = ix[near]

            def ring_counts(neighbors):
                return [
                    onehot[neighbors[use, a:b]].sum(1) for a, b in [(0, 15), (15, 30), (30, 60)]
                ]

            arrays = {
                "ordinary": ring_counts(d["neighbors"]),
                "gradient": ring_counts(d["adjusted"]),
                "oracle_same_lamina": ring_counts(oracle),
            }
            for kind, rings in arrays.items():
                x = np.concatenate([np.sqrt(r / r.sum(1, keepdims=True)) for r in rings], axis=1)
                for model in ["kmeans", "ring"]:
                    for seed in [17, 29, 43]:

                        def fit():
                            if model == "kmeans":
                                return KMeans(
                                    n_clusters=len(large), n_init=10, random_state=seed
                                ).fit_predict(x)
                            return fit_mixture(rings, n_components=len(large), seed=seed)[
                                "q"
                            ].argmax(1)

                        pred = cached(
                            f"boundary_oracle_{s['name']}_{kind}_{model}_seed{seed}", fit
                        )
                        records.append(
                            dict(
                                name=s["name"],
                                species=species,
                                condition=kind,
                                model=model,
                                seed=seed,
                                ari=adjusted_rand_score(y, pred),
                                n_classes=len(large),
                                n_cells=len(y),
                                fraction_polygons_used=float(use.sum() / d["y"].notna().sum()),
                            )
                        )
            pd.DataFrame(records).to_csv(OUT / "boundary_oracle.csv", index=False)
            print("BOUNDARY_ORACLE", s["name"], flush=True)


def oracle_null():
    """Measure artificial agreement induced by oracle grouping with randomized types."""
    records = []
    for species in SPECIES:
        for split in ["confirmation1", "confirmation2"]:
            s = get_section("spinal_" + species, split)
            d = get_data(s)
            labels = d["y"].fillna("__unlabeled").to_numpy()
            counts = d["y"].value_counts()
            large = counts[counts >= 61].index.to_numpy()
            use = np.isin(labels, large)
            y = labels[use].astype(str)
            xy = s["ref"][["x", "y"]].to_numpy(dtype=float)
            cats = pd.Categorical(s["ref"].type)
            oracle = np.zeros((len(xy), 60), int)
            for label in large:
                ids = np.flatnonzero(labels == label)
                _, near = nearest(xy[ids], 60)
                oracle[ids] = ids[near]
            for seed in [17, 29, 43]:
                codes = np.random.default_rng(seed).permutation(cats.codes)
                onehot = np.eye(len(cats.categories))[codes]
                rings = [onehot[oracle[use, a:b]].sum(1) for a, b in [(0, 15), (15, 30), (30, 60)]]
                x = np.concatenate([np.sqrt(r / r.sum(1, keepdims=True)) for r in rings], axis=1)
                for model in ["kmeans", "ring"]:

                    def fit():
                        if model == "kmeans":
                            return KMeans(
                                n_clusters=len(large), n_init=10, random_state=seed
                            ).fit_predict(x)
                        return fit_mixture(rings, n_components=len(large), seed=seed)["q"].argmax(
                            1
                        )

                    pred = cached(f"oracle_null_{s['name']}_{model}_seed{seed}", fit)
                    records.append(
                        dict(
                            name=s["name"],
                            species=species,
                            model=model,
                            seed=seed,
                            ari=adjusted_rand_score(y, pred),
                        )
                    )
            pd.DataFrame(records).to_csv(OUT / "oracle_null.csv", index=False)
            print("ORACLE_NULL", s["name"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["simulation", "quality", "boundary", "oracle_null"])
    args = parser.parse_args()
    {
        "simulation": simulate,
        "quality": quality,
        "boundary": boundary_oracle,
        "oracle_null": oracle_null,
    }[args.stage]()
