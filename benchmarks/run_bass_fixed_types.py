"""Mask-selected benchmarking of the experimental fixed-type BASS-style model."""

import torch  # noqa: F401

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from bass_fixed_types import fit, initial_labels, physical_graph
from context_experiments import get_section, unsupervised
from incremental_domains import DATASETS, features, vocabulary, score
from incremental_models import fit_mixture


ROOT = Path("/scratch/lamination_station_bass")
OUT = Path("/results/lamination_station_bass")
ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BETAS = [2.0, 4.0, 6.0, 8.0, 12.0]
SHRINKAGES = [5.0, 50.0]
SEEDS = [17, 29, 43]
VERSION = "v1"


def cached(key, fn):
    path = ROOT / (VERSION + "_" + key + ".pkl")
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    value = fn()
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)
    return value


def inputs(s, masked=False, fold=0):
    ref = s["ref"]
    xy = ref[["x", "y"]].to_numpy(dtype=float)
    # The vocabulary is the supplied cell-type codebook, not region annotations.
    names, codes = np.unique(ref.type.astype(str), return_inverse=True)
    observed = codes.copy()
    if masked:
        seed = int(hashlib.sha256((s["name"] + str(fold)).encode()).hexdigest()[:8], 16)
        holdout = np.random.default_rng(seed).choice(len(ref), int(0.2 * len(ref)), replace=False)
        observed[holdout] = -1
    graph, weighted = cached("graph_" + s["name"], lambda: physical_graph(xy))
    return dict(xy=xy, names=names, codes=codes, observed=observed, graph=graph, weighted=weighted)


def experiment(
    s,
    k,
    beta,
    shrinkage,
    seed,
    *,
    masked=False,
    fold=0,
    burnin=1000,
    draws=200,
    thin=5,
    init_kind="local",
):
    d = inputs(s, masked, fold)
    init_key = f"init_{s['name']}_K{k}_seed{seed}_mask{masked}_fold{fold}_{init_kind}"

    def initialize():
        if init_kind == "gradient":
            assert not masked
            return baseline(s, k, seed)["q"].argmax(1)
        return initial_labels(d["xy"], d["observed"], len(d["names"]), k, seed)

    initial = cached(init_key, initialize)
    key = f"fit_{init_key}_b{beta}_a{shrinkage}_burn{burnin}_draw{draws}_thin{thin}"
    start = time.monotonic()
    result = cached(
        key,
        lambda: fit(
            d["observed"],
            d["weighted"],
            initial,
            n_types=len(d["names"]),
            beta=beta,
            shrinkage=shrinkage,
            seed=seed,
            burnin=burnin,
            draws=draws,
            thin=thin,
        ),
    )
    return result, d, key, time.monotonic() - start


def baseline(s, k, seed):
    if k == 8:
        return dict(q=unsupervised(s, dict(method="ring", kind="gradient", k=8), seed))
    return cached(
        f"baseline_{s['name']}_K{k}_seed{seed}",
        lambda: fit_mixture(features(s, vocabulary([s])), n_components=k, seed=seed),
    )


def mark_score(result, d):
    hidden = result["hidden"]
    actual = d["codes"][hidden]
    prob = result["prediction"][np.arange(len(hidden)), actual]
    half = result["half_prediction"][:, np.arange(len(hidden)), actual]
    return dict(
        nll=float(-np.log(np.maximum(prob, 1e-300)).mean()),
        base_nll=float(-np.log(result["base"][actual]).mean()),
        half_nll_gap=float(
            abs(
                np.log(np.maximum(half[0], 1e-300)).mean()
                - np.log(np.maximum(half[1], 1e-300)).mean()
            )
        ),
        n_masked=len(hidden),
    )


def knn_scores(s, d):
    train = np.flatnonzero(d["observed"] >= 0)
    hidden = np.flatnonzero(d["observed"] < 0)
    ix = train[
        NearestNeighbors(n_neighbors=min(60, len(train)))
        .fit(d["xy"][train])
        .kneighbors(d["xy"][hidden], return_distance=False)
    ]
    base = np.bincount(d["observed"][train], minlength=len(d["names"])) + 0.5
    base /= base.sum()
    rows = []
    for k in [15, 30, 60]:
        counts = np.zeros((len(hidden), len(base)))
        np.add.at(
            counts,
            (
                np.repeat(np.arange(len(hidden)), min(k, ix.shape[1])),
                d["observed"][ix[:, :k]].ravel(),
            ),
            1,
        )
        p = (counts + 5 * base) / (counts.sum(1, keepdims=True) + 5)
        nll = -np.log(p[np.arange(len(hidden)), d["codes"][hidden]]).mean()
        rows.append(dict(name=s["name"], dataset=s["dataset"], k=k, nll=nll))
    return rows


def protocol():
    p = dict(
        model="Fixed cell-type categorical emissions, jointly sampled regional profiles and Potts labels; not the BASS package.",
        likelihood="Each observed cell type enters once. Missing types have zero emission contribution.",
        profiles="Dirichlet(shrinkage * empirical overall observed type frequencies); frequencies have 0.5 pseudo-count per type. Shrinkage is not a sparse spike-and-slab prior.",
        geometry="Symmetric union 8NN capped at section 95th percentile of 8th distance; symmetric degree normalization; no type-based edge gating.",
        pieces="Explicit piece IDs are supported by model; dataset benchmark supplies sections. Distance cap removes long gap edges but is not a verified tissue mask.",
        sampling="External-field Swendsen-Wang plus random-order single-site Gibbs and conjugate Dirichlet profile draws; fixed beta, not BASS beta inference.",
        initialization="KMeans of observed-only 15/30NN compositions. These counts are used only to initialize. Gradient-ring initialization is a separate sensitivity check.",
        selection="Mask 20% of types on historical development/test sections, seed17; choose beta and shrinkage by mean predictive negative log likelihood, equally weighting sections, separately for spinal/cortex and K. No region annotations used.",
        grid=dict(beta=BETAS, shrinkage=SHRINKAGES, spinal_k=[8, 13], cortex_k=[8]),
        pilot="500 burn-in + 100 retained draws at spacing5; selected configuration rerun longer for evaluation.",
        evaluation="Full-type fits on fresh1/fresh2 and spinal confirmation1/confirmation2, seeds17/29/43; all sections previously evaluated, not fresh holdouts or donor-independent.",
        baseline="Existing gradient-adjusted three-ring multinomial mixture, same fixed focal cells and matched K; separately compare local/global masked-type predictors.",
        references="Spinal direct polygon labels; cortex composition-derived layer proxies, not independent anatomical truth.",
        scope="Exploratory analysis files only; no core package changes.",
    )
    (OUT / "protocol.json").write_text(json.dumps(p, indent=2))


def pilot(dataset):
    records, controls = [], []
    for split in ["development", "test"]:
        s = get_section(dataset, split)
        d = inputs(s, True)
        controls.extend(knn_scores(s, d))
        for k in [8, 13] if dataset.startswith("spinal") else [8]:
            for beta in BETAS:
                for alpha in SHRINKAGES:
                    result, d, key, elapsed = experiment(
                        s, k, beta, alpha, 17, masked=True, burnin=500, draws=100
                    )
                    record = dict(
                        name=s["name"],
                        dataset=dataset,
                        k=k,
                        beta=beta,
                        shrinkage=alpha,
                        **mark_score(result, d),
                        occupied=len(np.unique(result["q"].argmax(1))),
                        cache_key=key,
                        seconds=elapsed,
                    )
                    records.append(record)
                    pd.DataFrame(records).to_csv(OUT / f"pilot_{dataset}.csv", index=False)
                    print(
                        "PILOT",
                        s["name"],
                        k,
                        beta,
                        alpha,
                        round(record["nll"], 4),
                        round(elapsed, 2),
                        flush=True,
                    )
    pd.DataFrame(controls).to_csv(OUT / f"knn_pilot_{dataset}.csv", index=False)


def selection():
    pilot = pd.concat([pd.read_csv(OUT / f"pilot_{d}.csv") for d in DATASETS])
    pilot["tissue"] = pilot.dataset.str.split("_").str[0]
    assert len(pilot) == 160
    groups = pilot.groupby(["tissue", "k", "beta", "shrinkage"]).nll.mean().reset_index()
    chosen = groups.loc[groups.groupby(["tissue", "k"]).nll.idxmin()]
    chosen.to_csv(OUT / "selected.csv", index=False)
    print(chosen.to_string(index=False))


def anatomical_score(s, result, d):
    scored = score(s, result["q"], d["graph"])
    if s["dataset"].startswith("spinal"):
        from ceiling_diagnostics import get_data

        y = get_data(s)["y"].dropna().astype(str)
        ix = s["ref"].index.get_indexer(y.index)
        pred = result["q"].argmax(1)[ix]
        scored["full_polygon_ari"] = adjusted_rand_score(y, pred)
        regions = []
        for label in sorted(y.unique()):
            a = y.to_numpy() == label
            intersections = np.bincount(pred[a], minlength=result["q"].shape[1])
            mass = np.bincount(pred, minlength=result["q"].shape[1])
            union = a.sum() + mass - intersections
            j = np.argmax(intersections / union)
            regions.append(
                dict(
                    lamina=label,
                    n=int(a.sum()),
                    best_iou=float(intersections[j] / union[j]),
                    recall=float(intersections[j] / a.sum()),
                    precision=float(intersections[j] / max(mass[j], 1)),
                )
            )
        scored["thin_best_iou"] = np.mean(
            [r["best_iou"] for r in regions if r["lamina"] in ["2i", "3"]]
        )
    else:
        regions = []
    return scored, regions


def evaluate(dataset, diagnostic=False):
    chosen = pd.read_csv(OUT / "selected.csv")
    chosen = chosen[chosen.tissue == dataset.split("_")[0]]
    records, regional = [], []
    splits = ["fresh1", "fresh2"] + (
        ["confirmation1", "confirmation2"] if dataset.startswith("spinal") else []
    )
    if diagnostic:
        splits = ["confirmation1"] if dataset.startswith("spinal") else ["fresh1"]
    for split in splits:
        s = get_section(dataset, split)
        for _, config in chosen.iterrows():
            k = int(config.k)
            seeds = [17] if diagnostic else SEEDS
            for seed in seeds:
                kind = "gradient" if diagnostic else "local"
                result, d, key, elapsed = experiment(
                    s, k, config.beta, config.shrinkage, seed, init_kind=kind
                )
                for method, value in [("bass", result), ("gradient_ring", baseline(s, k, seed))]:
                    scored, regs = anatomical_score(s, value, d)
                    row = dict(
                        name=s["name"],
                        dataset=dataset,
                        k=k,
                        seed=seed,
                        method=method,
                        beta=config.beta,
                        shrinkage=config.shrinkage,
                        init_kind=kind,
                        cache_key=key if method == "bass" else "",
                        **scored,
                    )
                    if method == "bass":
                        row["mean_step_change"] = result["trace"][:, 2].mean()
                        row["sample_ari_to_modal"] = np.mean(
                            [
                                adjusted_rand_score(result["q"].argmax(1), z)
                                for z in result["samples"]
                            ]
                        )
                        row["initial_ari"] = score(s, np.eye(k)[result["initial"]], d["graph"])[
                            "ari"
                        ]
                    records.append(row)
                    regional.extend(
                        [
                            dict(
                                name=s["name"], dataset=dataset, k=k, seed=seed, method=method, **r
                            )
                            for r in regs
                        ]
                    )
                    print("EVAL", s["name"], k, seed, method, round(scored["ari"], 4), flush=True)
                suffix = "_gradientinit" if diagnostic else ""
                pd.DataFrame(records).to_csv(
                    OUT / f"evaluation_{dataset}{suffix}.csv", index=False
                )
                pd.DataFrame(regional).to_csv(OUT / f"regions_{dataset}{suffix}.csv", index=False)


def validation(dataset):
    selected = pd.read_csv(OUT / "selected.csv")
    selected = selected[selected.tissue == dataset.split("_")[0]]
    records, controls = [], []
    splits = (
        ["confirmation1", "confirmation2"]
        if dataset.startswith("spinal")
        else ["fresh1", "fresh2"]
    )
    for split in splits:
        s = get_section(dataset, split)
        d = inputs(s, True, fold=1)
        controls.extend(knn_scores(s, d))
        for _, config in selected.iterrows():
            for seed in SEEDS:
                result, d, key, elapsed = experiment(
                    s, int(config.k), config.beta, config.shrinkage, seed, masked=True, fold=1
                )
                records.append(
                    dict(
                        name=s["name"],
                        dataset=dataset,
                        k=int(config.k),
                        seed=seed,
                        **mark_score(result, d),
                        cache_key=key,
                    )
                )
                pd.DataFrame(records).to_csv(OUT / f"validation_{dataset}.csv", index=False)
                print("VALIDATE", s["name"], int(config.k), seed, records[-1]["nll"], flush=True)
    pd.DataFrame(controls).to_csv(OUT / f"knn_validation_{dataset}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["protocol", "pilot", "select", "evaluate", "gradientinit", "validate"]
    )
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.stage == "protocol":
        protocol()
    elif args.stage == "pilot":
        pilot(args.dataset)
    elif args.stage == "select":
        selection()
    elif args.stage == "evaluate":
        evaluate(args.dataset)
    elif args.stage == "gradientinit":
        evaluate(args.dataset, True)
    elif args.stage == "validate":
        validation(args.dataset)
