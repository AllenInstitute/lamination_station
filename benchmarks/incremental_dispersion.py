"""Separate overdispersion ablation; settings are selected without evaluation labels."""

import torch  # noqa: F401

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from incremental_domains import (
    CACHE,
    DATASETS,
    OUT,
    SEEDS,
    cached,
    features,
    load_section,
    make_training,
    score,
    section_graphs,
)
from incremental_models import fit_dm, fit_mixture

CONCENTRATIONS = [30.0, 100.0]


def tune():
    """Select fixed concentration on the same two historical sections per dataset."""
    (OUT / "dispersion_protocol.json").write_text(
        json.dumps(
            {
                "comparison": "Independent per-section Dirichlet-multinomial versus independent multinomial; no sharing or graph refinement",
                "selection": "Original development/test sections, equal section/dataset ARI, separately spinal/cortex; seed17",
                "concentrations": CONCENTRATIONS,
                "inference": "Initialize from best of five multinomial starts; exact fixed-concentration DM MAP M-step via constrained Newton ascent; at most80 EM iterations",
                "evaluation": "Same fresh1/fresh2 sections, seeds17/29/43; concentration locked before scoring",
                "limitation": "One DM initialization per multinomial fit; composite likelihood remains due to overlapping neighborhoods",
            },
            indent=2,
        )
    )
    records = []
    for dataset in DATASETS:
        tissue = "spinal" if dataset.startswith("spinal") else "cortex"
        for train_split, target_split in [("development", "test"), ("test", "development")]:
            target = load_section(dataset, target_split)
            key = f"tune_{dataset}_{train_split}"
            training = cached(
                key + "_training", lambda: make_training([load_section(dataset, train_split)], 17)
            )
            rings = features(target, training["columns"])
            graph = section_graphs(target, rings)[0]
            initial = cached(key + "_local", lambda: fit_mixture(rings, seed=17))
            for concentration in CONCENTRATIONS:
                result = cached(
                    key + f"_dm{concentration}",
                    lambda: fit_dm(rings, initial, concentration=concentration),
                )
                records.append(
                    dict(
                        name=target["name"],
                        tissue=tissue,
                        concentration=concentration,
                        converged=result["converged"],
                        optimizer_failures=result["optimizer_failures"],
                        **score(target, result["q"], graph),
                    )
                )
                pd.DataFrame(records).to_csv(OUT / "dispersion_tuning.csv", index=False)
                print(
                    "DM TUNED",
                    target["name"],
                    concentration,
                    records[-1]["ari"],
                    result["converged"],
                    flush=True,
                )
    frame = pd.DataFrame(records)
    selected = {}
    for tissue in ["spinal", "cortex"]:
        mean = frame[frame.tissue == tissue].groupby("concentration").ari.mean()
        selected[tissue] = float(mean.idxmax())
    (OUT / "dispersion_selected.json").write_text(json.dumps(selected, indent=2))
    shutil.copy2(OUT / "dispersion_selected.json", CACHE / "dispersion_selected.json")
    print("DM SELECTED", selected, flush=True)


def evaluate():
    """Run the locked overdispersion contrast on reserved sections."""
    selected = json.loads((OUT / "dispersion_selected.json").read_text())
    records = []
    for dataset in DATASETS:
        tissue = "spinal" if dataset.startswith("spinal") else "cortex"
        concentration = selected[tissue]
        for seed in SEEDS:
            training = cached(
                f"final_{dataset}_training_{seed}",
                lambda: make_training(
                    [load_section(dataset, split) for split in ["development", "test"]], seed
                ),
            )
            for split in ["fresh1", "fresh2"]:
                section = load_section(dataset, split)
                key = f"eval_{dataset}_{split}_{seed}"
                rings = features(section, training["columns"])
                graph = section_graphs(section, rings)[0]
                initial = cached(key + "_local", lambda: fit_mixture(rings, seed=seed))
                result = cached(
                    key + f"_dm{concentration}",
                    lambda: fit_dm(rings, initial, concentration=concentration),
                )
                records.append(
                    dict(
                        name=section["name"],
                        dataset=dataset,
                        tissue=tissue,
                        method="local_dm",
                        seed=seed,
                        concentration=concentration,
                        converged=result["converged"],
                        optimizer_failures=result["optimizer_failures"],
                        **score(section, result["q"], graph),
                    )
                )
                np.savez_compressed(
                    OUT / "predictions" / f"{section['name']}_local_dm_{seed}.npz",
                    q=result["q"],
                    index=section["ref"].index.astype(str).to_numpy(dtype=str),
                )
                pd.DataFrame(records).to_csv(OUT / "dispersion_evaluation.csv", index=False)
                print("DM EVALUATED", key, records[-1]["ari"], result["converged"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tune", "evaluate", "all"])
    args = parser.parse_args()
    shutil.copy2(__file__, OUT / Path(__file__).name)
    if args.stage in ["tune", "all"]:
        tune()
    if args.stage in ["evaluate", "all"]:
        evaluate()
