"""Stagewise, section-separated benchmark of experimental composition domain models."""

import torch

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
)
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/root/capsule/code/lamination_station/src")
from lamination_station.preprocessing import calculate_comp_grads
from incremental_models import fit_mixture, learn_compatibility, refine, spatial_graphs

torch.set_num_threads(2)
threadpool_limits(2)
OLD = Path("/scratch/lamination_station_benchmark")
PREVIOUS = Path("/scratch/lamination_station_investigation")
CACHE = Path("/scratch/lamination_station_incremental")
OUT = Path("/results/lamination_station_incremental")
DATASETS = ["spinal_human", "spinal_macaque", "spinal_mouse", "cortex_human", "cortex_merfish"]
ALPHAS = [20.0, 100.0, float("inf")]
STRENGTHS = [0.0, 1.0, 4.0, 16.0]
CONTACTS = [0.0, 0.5, 2.0, 8.0]
SEEDS = [17, 29, 43]
for directory in [CACHE, OUT, OUT / "predictions"]:
    directory.mkdir(parents=True, exist_ok=True)


def read_pickle(path):
    """Read one locally generated trusted intermediate."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def cached(name, function):
    """Persist expensive fits independently of the report."""
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return read_pickle(path)
    value = function()
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return value


def load_section(dataset, split):
    """Load features with reference labels kept in a separate scoring table."""

    def prepare():
        if split.startswith("fresh"):
            ref, counts, meta = read_pickle(PREVIOUS / f"{dataset}_{split}.pkl")
        else:
            _, meta, ref = read_pickle(OLD / f"{dataset}_{split}.pkl")
            if split == "development":
                counts = {30: read_pickle(PREVIOUS / f"{dataset}_development_full.pkl")[5]}
                counts.update(
                    {
                        k: pd.read_parquet(PREVIOUS / f"{dataset}_development_full_k{k}.parquet")
                        for k in [15, 60]
                    }
                )
            else:
                counts = {
                    k: calculate_comp_grads(
                        ref, "x", "y", "type", near_k=k, far_k=90, sample_col="section"
                    )[5]
                    for k in [15, 30, 60]
                }
        assert all(frame.index.equals(ref.index) for frame in counts.values())
        assert meta.index.isin(ref.index).all()
        return dict(ref=ref, counts=counts, meta=meta, name=f"{dataset}_{split}", dataset=dataset)

    return cached(f"data_{dataset}_{split}", prepare)


def vocabulary(sections):
    """Derive a type vocabulary from training observations, plus an unseen bucket."""
    columns = set()
    for section in sections:
        frame = section["counts"][60]
        columns.update(frame.columns[frame.sum(0) > 0])
    assert "__unseen_count" not in columns
    return sorted(columns) + ["__unseen_count"]


def features(section, columns):
    """Align counts without dropping unknown cell types or changing neighborhoods."""
    counts = {}
    for k, frame in section["counts"].items():
        aligned = frame.reindex(columns=columns, fill_value=0).astype(float)
        unknown = frame.columns.difference(columns)
        aligned["__unseen_count"] = frame[unknown].sum(1)
        counts[k] = aligned.to_numpy()
        np.testing.assert_allclose(counts[k].sum(1), k)
    rings = [counts[15], counts[30] - counts[15], counts[60] - counts[30]]
    return rings


def section_graphs(section, rings):
    """Cache geometry, which does not use anatomical labels."""
    return spatial_graphs(section["ref"][["x", "y"]].to_numpy(), sum(rings))


def score(section, q, graph):
    """Score fixed focal labels and labeled physical edges, without label matching."""
    ref, meta = section["ref"], section["meta"]
    ix = ref.index.get_indexer(meta.index)
    valid = meta.label.notna().to_numpy()
    prediction = q.argmax(1)
    label = meta.label.reindex(ref.index)
    coo = graph.tocoo()
    use = (
        (coo.row < coo.col) & label.notna().to_numpy()[coo.row] & label.notna().to_numpy()[coo.col]
    )
    rr, cc = coo.row[use], coo.col[use]
    truth = label.iloc[rr].to_numpy() != label.iloc[cc].to_numpy()
    probability = np.clip(1 - np.sum(q[rr] * q[cc], axis=1), 0, 1)
    predicted_boundary = prediction[rr] != prediction[cc]
    return dict(
        ari=adjusted_rand_score(meta.label[valid], prediction[ix][valid]),
        boundary_f1=f1_score(truth, predicted_boundary, zero_division=0),
        boundary_brier=brier_score_loss(truth, probability),
        boundary_ap=average_precision_score(truth, probability) if truth.any() else np.nan,
        n_evaluation=int(valid.sum()),
        n_labeled_edges=len(rr),
        boundary_prevalence=float(truth.mean()),
        occupied=len(np.unique(prediction)),
        mean_entropy=float(-xlogy(q, q).sum(1).mean()),
        fraction_confident=float((q.max(1) >= 0.9).mean()),
    )


def make_training(sections, seed, n_components=8):
    """Jointly learn balanced profiles and adjacency potentials from training sections."""
    columns = vocabulary(sections)
    arrays = [features(s, columns) for s in sections]
    rings = [np.concatenate([a[r] for a in arrays]) for r in range(3)]
    group = np.concatenate([np.full(len(a[0]), i) for i, a in enumerate(arrays)])
    fit = fit_mixture(rings, groups=group, seed=seed, n_components=n_components)
    graphs = [section_graphs(s, a)[0] for s, a in zip(sections, arrays, strict=True)]
    probabilities = [fit["q"][group == i] for i in range(len(sections))]
    compatibility = learn_compatibility(graphs, probabilities)
    return dict(
        fit=fit,
        columns=columns,
        compatibility=compatibility,
        train_names=[s["name"] for s in sections],
    )


def adapt(rings, training, alpha, case_weights=None):
    """Infer target profiles/prevalences without access to reference-region labels."""
    return fit_mixture(
        rings,
        n_components=len(training["fit"]["profiles"][0]),
        prior=training["fit"]["profiles"],
        prior_cells=alpha,
        case_weights=case_weights,
    )


def prepare_all():
    """Freeze the analysis protocol and cache counts before any model selection."""
    protocol = {
        "baseline": "Gradient-adjusted disjoint ranks1-15/16-30/31-60, far90, K8, five starts, multinomial MAP mixture",
        "datasets": DATASETS,
        "selection": "Two-fold section cross-validation on historical development/test sections, seed17; average ARI equally over folds and datasets, separately for spinal and cortex",
        "evaluation": "Previously evaluated fresh1/fresh2 sections per dataset, seeds17/29/43; not newly unseen or donor-independent",
        "sharing": "Only within species/platform datasets; separate profiles and sample-specific prevalences; equal section weights; training-only type vocabulary and unseen-type bucket",
        "stage1": "Select target profile prior pseudo-cells per component/ring from20,100,infinity",
        "stage2": "Select gated smoothing strength0,1,4,16 after selecting stage1; also select uniform control and local-fit gated control separately",
        "stage3": "Select training-derived contact potential strength0,.5,2,8 after stage2; also report best nonzero candidate as a diagnostic",
        "spatial": "Mutual12NN capped at per-piece95th percentile of12th distance; symmetric degree normalization; gating exp(-Hellinger_squared/median_positive_edge_distance)",
        "adjacency": "Confidence>=.8 training assignments; off-diagonal physical contacts versus endpoint-abundance null;20 directed-edge pseudo-count mass; log enrichment clipped[-2,2]; equally average supported samples",
        "uncertainty": "Entropy and seed agreement; boundary probabilities are mean-field marginal products, uncalibrated; later spatial-block reweighting diagnostics",
        "reference": "Spinal direct polygon annotations only; cortex composition-derived layer proxies",
        "inference": "Fixed composition unary scores during graph refinement; damped monotone mean-field objective; no anatomical labels or inferred layer ordering in model",
        "scope": "Experimental benchmark implementations; no package-default changes; no expression inputs or forced anatomical template",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    manifest = []
    for dataset in DATASETS:
        for split in ["development", "test", "fresh1", "fresh2"]:
            section = load_section(dataset, split)
            manifest.append(
                dict(
                    name=section["name"],
                    section=str(section["ref"].section.iloc[0]),
                    n_reference=len(section["ref"]),
                    n_evaluation=int(section["meta"].label.notna().sum()),
                )
            )
            print("PREPARED", section["name"], flush=True)
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2))


def tune():
    """Select increments sequentially using only the designated tuning sections."""
    records = []
    contexts = []
    for dataset in DATASETS:
        for train_split, target_split in [("development", "test"), ("test", "development")]:
            train, target = load_section(dataset, train_split), load_section(dataset, target_split)
            key = f"tune_{dataset}_{train_split}"
            training = cached(key + "_training", lambda: make_training([train], 17))
            rings = features(target, training["columns"])
            physical, uniform, gated = section_graphs(target, rings)
            local = cached(key + "_local", lambda: fit_mixture(rings, seed=17))
            ctx = dict(
                key=key,
                training=training,
                rings=rings,
                physical=physical,
                uniform=uniform,
                gated=gated,
                local=local,
                target=target,
                tissue="spinal" if dataset.startswith("spinal") else "cortex",
            )
            contexts.append(ctx)
            for alpha in ALPHAS:
                fit = cached(key + f"_adapt{alpha}", lambda: adapt(rings, training, alpha))
                records.append(
                    dict(
                        name=target["name"],
                        tissue=ctx["tissue"],
                        stage="shared",
                        value=alpha,
                        **score(target, fit["q"], physical),
                    )
                )
            print("TUNED SHARING", key, flush=True)
    selected = {tissue: {} for tissue in ["spinal", "cortex"]}

    def choose(stage, parameter, nonzero=False):
        frame = pd.DataFrame(records)
        for tissue in selected:
            group = frame[(frame.stage == stage) & (frame.tissue == tissue)]
            if nonzero:
                group = group[group.value > 0]
            means = group.groupby("value").ari.mean().sort_values(ascending=False, kind="stable")
            selected[tissue][parameter] = float(means.index[0])

    choose("shared", "prior_cells")
    for ctx in contexts:
        alpha = selected[ctx["tissue"]]["prior_cells"]
        fit = cached(
            ctx["key"] + f"_adapt{alpha}", lambda: adapt(ctx["rings"], ctx["training"], alpha)
        )
        ctx["shared"] = fit
        for stage, unary, graph in [
            ("gated", fit["unary"], ctx["gated"]),
            ("uniform", fit["unary"], ctx["uniform"]),
            ("local_gated", ctx["local"]["unary"], ctx["gated"]),
        ]:
            for strength in STRENGTHS:
                result = cached(
                    ctx["key"] + f"_{stage}_{alpha}_{strength}",
                    lambda: refine(unary, graph, ctx["uniform"], strength=strength),
                )
                records.append(
                    dict(
                        name=ctx["target"]["name"],
                        tissue=ctx["tissue"],
                        stage=stage,
                        value=strength,
                        **score(ctx["target"], result["q"], ctx["physical"]),
                    )
                )
        print("TUNED SMOOTHING", ctx["key"], flush=True)
    for stage in ["gated", "uniform", "local_gated"]:
        choose(stage, stage + "_strength")
    for ctx in contexts:
        alpha = selected[ctx["tissue"]]["prior_cells"]
        strength = selected[ctx["tissue"]]["gated_strength"]
        for rho in CONTACTS:
            result = cached(
                ctx["key"] + f"_contact_{alpha}_{strength}_{rho}",
                lambda: refine(
                    ctx["shared"]["unary"],
                    ctx["gated"],
                    ctx["uniform"],
                    strength=strength,
                    compatibility=ctx["training"]["compatibility"],
                    contact_strength=rho,
                ),
            )
            records.append(
                dict(
                    name=ctx["target"]["name"],
                    tissue=ctx["tissue"],
                    stage="contact",
                    value=rho,
                    **score(ctx["target"], result["q"], ctx["physical"]),
                )
            )
        print("TUNED ADJACENCY", ctx["key"], flush=True)
    choose("contact", "contact_strength")
    choose("contact", "nonzero_contact_strength", nonzero=True)
    pd.DataFrame(records).to_csv(OUT / "tuning.csv", index=False)
    # JSON uses a string for an infinite prior, avoiding nonstandard JSON Infinity.
    serial = {
        t: {k: ("fixed" if np.isinf(v) else v) for k, v in params.items()}
        for t, params in selected.items()
    }
    (OUT / "selected.json").write_text(json.dumps(serial, indent=2))
    shutil.copy2(OUT / "selected.json", CACHE / "selected.json")
    print("SELECTED", serial, flush=True)


def settings():
    """Read the locked development choices before evaluation."""
    result = json.loads((OUT / "selected.json").read_text())
    return {
        t: {k: (float("inf") if v == "fixed" else v) for k, v in p.items()}
        for t, p in result.items()
    }


def evaluate():
    """Fit all reserved comparisons without selecting settings from their scores."""
    selected = settings()
    records = []
    for dataset in DATASETS:
        tissue = "spinal" if dataset.startswith("spinal") else "cortex"
        params = selected[tissue]
        train = [load_section(dataset, split) for split in ["development", "test"]]
        for seed in SEEDS:
            training = cached(
                f"final_{dataset}_training_{seed}", lambda: make_training(train, seed)
            )
            for split in ["fresh1", "fresh2"]:
                section = load_section(dataset, split)
                key = f"eval_{dataset}_{split}_{seed}"
                rings = features(section, training["columns"])
                physical, uniform, gated = section_graphs(section, rings)
                local = cached(key + "_local", lambda: fit_mixture(rings, seed=seed))
                shared = cached(
                    key + "_shared", lambda: adapt(rings, training, params["prior_cells"])
                )
                results = {"local": local, "shared": shared}
                configs = [
                    ("shared_uniform", shared["unary"], uniform, params["uniform_strength"], 0.0),
                    ("shared_gated", shared["unary"], gated, params["gated_strength"], 0.0),
                    (
                        "shared_adjacency",
                        shared["unary"],
                        gated,
                        params["gated_strength"],
                        params["contact_strength"],
                    ),
                    (
                        "shared_adjacency_nonzero",
                        shared["unary"],
                        gated,
                        params["gated_strength"],
                        params["nonzero_contact_strength"],
                    ),
                    ("local_gated", local["unary"], gated, params["local_gated_strength"], 0.0),
                ]
                for method, unary, graph, strength, rho in configs:
                    results[method] = cached(
                        key + "_" + method,
                        lambda: refine(
                            unary,
                            graph,
                            uniform,
                            strength=strength,
                            compatibility=training["compatibility"],
                            contact_strength=rho,
                        ),
                    )
                for method, result in results.items():
                    records.append(
                        dict(
                            name=section["name"],
                            dataset=dataset,
                            tissue=tissue,
                            method=method,
                            seed=seed,
                            converged=result["converged"],
                            **score(section, result["q"], physical),
                        )
                    )
                    np.savez_compressed(
                        OUT / "predictions" / f"{section['name']}_{method}_{seed}.npz",
                        q=result["q"],
                        index=section["ref"].index.astype(str).to_numpy(dtype=str),
                    )
                print(
                    "EVALUATED",
                    key,
                    {
                        m: round(score(section, f["q"], physical)["ari"], 3)
                        for m, f in results.items()
                    },
                    flush=True,
                )
                pd.DataFrame(records).to_csv(OUT / "evaluation.csv", index=False)
    frame = pd.DataFrame(records)
    section_means = (
        frame.groupby(["name", "dataset", "tissue", "method"])
        .mean(numeric_only=True)
        .reset_index()
    )
    section_means.to_csv(OUT / "section_means.csv", index=False)
    summary = section_means.groupby(["tissue", "method"])[
        ["ari", "boundary_f1", "boundary_brier", "boundary_ap", "occupied"]
    ].mean()
    summary.to_csv(OUT / "summary.csv")
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "tune", "evaluate", "all"])
    args = parser.parse_args()
    for name in ["incremental_domains.py", "incremental_models.py"]:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, OUT / name)
    fingerprints = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name("incremental_models.py")]
    }
    (OUT / "source_hashes.json").write_text(json.dumps(fingerprints, indent=2))
    if args.stage in ["all", "prepare"]:
        prepare_all()
    if args.stage in ["all", "tune"]:
        tune()
    if args.stage in ["all", "evaluate"]:
        evaluate()
