"""Exploratory spinal dictionary-capacity check after inspection of K8 failures."""

import torch  # noqa: F401

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from incremental_domains import (
    DATASETS,
    OUT,
    SEEDS,
    adapt,
    cached,
    features,
    load_section,
    make_training,
    score,
    section_graphs,
)
from incremental_models import fit_mixture

shutil.copy2(__file__, OUT / Path(__file__).name)
(OUT / "capacity_protocol.json").write_text(
    json.dumps(
        {
            "status": "Exploratory follow-up after K8 evaluation showed merged spinal regions; not an independent confirmation",
            "settings": "K12, seeds17/29/43; compare local versus shared; retain100-pseudo-cell prior selected for K8; no spatial terms or further parameter tuning",
            "scope": "Three spinal datasets, same historical training sections and same six evaluation sections",
        },
        indent=2,
    )
)
records = []
for dataset in [name for name in DATASETS if name.startswith("spinal")]:
    train = [load_section(dataset, split) for split in ["development", "test"]]
    for seed in SEEDS:
        training = cached(
            f"capacity_{dataset}_training_{seed}",
            lambda: make_training(train, seed, n_components=12),
        )
        for split in ["fresh1", "fresh2"]:
            section = load_section(dataset, split)
            rings = features(section, training["columns"])
            graph = section_graphs(section, rings)[0]
            key = f"capacity_{dataset}_{split}_{seed}"
            local = cached(key + "_local", lambda: fit_mixture(rings, seed=seed, n_components=12))
            shared = cached(key + "_shared", lambda: adapt(rings, training, 100.0))
            for method, fitted in [("local", local), ("shared", shared)]:
                records.append(
                    dict(
                        name=section["name"],
                        method=method,
                        seed=seed,
                        k=12,
                        **score(section, fitted["q"], graph),
                    )
                )
                np.savez_compressed(
                    OUT / "predictions" / f"{section['name']}_{method}_K12_{seed}.npz",
                    q=fitted["q"],
                    index=section["ref"].index.astype(str).to_numpy(dtype=str),
                )
            pd.DataFrame(records).to_csv(OUT / "capacity_evaluation.csv", index=False)
            print(
                "CAPACITY",
                key,
                {
                    method: round(score(section, fitted["q"], graph)["ari"], 3)
                    for method, fitted in [("local", local), ("shared", shared)]
                },
                flush=True,
            )
print(pd.DataFrame(records).groupby("method")[["ari", "boundary_f1", "occupied"]].mean())
