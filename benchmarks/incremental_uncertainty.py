"""Conditional block-reweighting stability and a paired cell-thinning stress test."""

import torch  # noqa: F401

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from incremental_domains import (
    CACHE,
    DATASETS,
    OUT,
    adapt,
    cached,
    features,
    load_section,
    read_pickle,
    score,
    section_graphs,
    settings,
)
from incremental_models import fit_mixture, refine
from lamination_station.preprocessing import calculate_comp_grads

shutil.copy2(__file__, OUT / Path(__file__).name)
(OUT / "uncertainty_protocol.json").write_text(
    json.dumps(
        {
            "block_reweighting": "Five exponential-weight perturbations of a6x6 spatial tiling; weights normalized to mean1; fixed neighborhoods and fixed training profiles; target profiles/prevalences refitted",
            "interpretation": "Conditional sensitivity diagnostic, not posterior intervals or calibrated uncertainty; does not resample training sections",
            "thinning": "One80%-retained cell sample per section, seed991; recompute gradient-adjusted neighborhoods and all graphs; compare only the same retained cells",
            "methods": "Local multinomial and development-selected shared+gated+adjacency pipeline, seed17",
            "alignment": "Hungarian match to original assignments for disagreement; matching never uses anatomical labels",
        },
        indent=2,
    )
)
selected = settings()
records, thinning_records = [], []
for dataset in DATASETS:
    tissue = "spinal" if dataset.startswith("spinal") else "cortex"
    params = selected[tissue]
    training = read_pickle(CACHE / f"final_{dataset}_training_17.pkl")
    for split in ["fresh1", "fresh2"]:
        section = load_section(dataset, split)
        key = f"eval_{dataset}_{split}_17"
        rings = features(section, training["columns"])
        physical, uniform, gated = section_graphs(section, rings)
        local = read_pickle(CACHE / f"{key}_local.pkl")
        shared_base = read_pickle(CACHE / f"{key}_shared_adjacency.pkl")
        xy = section["ref"][["x", "y"]].to_numpy(dtype=float)
        tiles = np.minimum(
            5, np.floor(6 * (xy - xy.min(0)) / (np.ptp(xy, axis=0) + 1e-12))
        ).astype(int)
        _, block = np.unique(tiles[:, 0] * 6 + tiles[:, 1], return_inverse=True)
        for method, base in [("local", local), ("shared_adjacency", shared_base)]:
            base_labels = base["q"].argmax(1)
            agreement = []
            aris = []
            for replicate in range(5):
                rng = np.random.default_rng(400 + replicate)
                weight = rng.exponential(size=block.max() + 1)[block]
                weight /= weight.mean()

                def perturb():
                    if method == "local":
                        fitted = fit_mixture(
                            rings, prior=local["profiles"], prior_cells=0.0, case_weights=weight
                        )
                        return fitted
                    fitted = adapt(rings, training, params["prior_cells"], case_weights=weight)
                    return refine(
                        fitted["unary"],
                        gated,
                        uniform,
                        strength=params["gated_strength"],
                        compatibility=training["compatibility"],
                        contact_strength=params["contact_strength"],
                    )

                result = cached(key + f"_block_{method}_{replicate}", perturb)
                labels = result["q"].argmax(1)
                confusion = np.bincount(8 * base_labels + labels, minlength=64).reshape(8, 8)
                rr, cc = linear_sum_assignment(-confusion)
                mapping = np.empty(8, int)
                mapping[cc] = rr
                agreement.append(mapping[labels] == base_labels)
                aris.append(adjusted_rand_score(base_labels, labels))
            stability = np.mean(agreement, axis=0)
            confidence = base["q"].max(1)
            entropy = -np.sum(base["q"] * np.log(np.maximum(base["q"], 1e-300)), axis=1)
            correlation = (
                spearmanr(entropy, 1 - stability).statistic if np.std(stability) > 0 else np.nan
            )
            records.append(
                dict(
                    name=section["name"],
                    tissue=tissue,
                    method=method,
                    mean_assignment_agreement=float(stability.mean()),
                    mean_replicate_ari=float(np.mean(aris)),
                    unstable_confident_fraction=float(
                        ((stability < 0.8) & (confidence >= 0.9)).mean()
                    ),
                    entropy_instability_spearman=float(correlation),
                )
            )
            pd.DataFrame(
                dict(
                    index=section["ref"].index,
                    confidence=confidence,
                    entropy=entropy,
                    block_assignment_agreement=stability,
                )
            ).to_csv(
                OUT / "predictions" / f"{section['name']}_{method}_uncertainty.csv.gz", index=False
            )

        def prepare_thinned():
            rng = np.random.default_rng(991)
            keep = np.sort(rng.choice(len(xy), int(0.8 * len(xy)), replace=False))
            ref = section["ref"].iloc[keep].copy()
            counts = {
                k: calculate_comp_grads(
                    ref, "x", "y", "type", near_k=k, far_k=90, sample_col="section"
                )[5]
                for k in [15, 30, 60]
            }
            meta = section["meta"].loc[section["meta"].index.isin(ref.index)]
            return dict(ref=ref, counts=counts, meta=meta, keep=keep, name=section["name"])

        thin = cached(key + "_thin_data", prepare_thinned)
        thin_rings = features(thin, training["columns"])
        thin_physical, thin_uniform, thin_gated = section_graphs(thin, thin_rings)
        for method, base in [("local", local), ("shared_adjacency", shared_base)]:

            def fit_thinned():
                if method == "local":
                    return fit_mixture(thin_rings, seed=17)
                fitted = adapt(thin_rings, training, params["prior_cells"])
                return refine(
                    fitted["unary"],
                    thin_gated,
                    thin_uniform,
                    strength=params["gated_strength"],
                    compatibility=training["compatibility"],
                    contact_strength=params["contact_strength"],
                )

            result = cached(key + f"_thin_{method}", fit_thinned)
            before = base["q"][thin["keep"]]
            original_score = score(thin, before, thin_physical)
            after_score = score(thin, result["q"], thin_physical)
            thinning_records.append(
                dict(
                    name=section["name"],
                    tissue=tissue,
                    method=method,
                    original_ari_same_cells=original_score["ari"],
                    thinned_ari=after_score["ari"],
                    delta_ari=after_score["ari"] - original_score["ari"],
                    assignment_ari=adjusted_rand_score(before.argmax(1), result["q"].argmax(1)),
                )
            )
        pd.DataFrame(records).to_csv(OUT / "uncertainty.csv", index=False)
        pd.DataFrame(thinning_records).to_csv(OUT / "thinning.csv", index=False)
        print("UNCERTAINTY CHECKED", section["name"], flush=True)
