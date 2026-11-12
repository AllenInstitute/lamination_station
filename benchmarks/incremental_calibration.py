"""Calibrate edge-boundary scores on development sections, then evaluate separately."""

import torch  # noqa: F401

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss

from incremental_domains import (
    CACHE,
    DATASETS,
    OUT,
    SEEDS,
    features,
    load_section,
    read_pickle,
    section_graphs,
    settings,
)

shutil.copy2(__file__, OUT / Path(__file__).name)
selected = settings()
dm_selected = json.loads((OUT / "dispersion_selected.json").read_text())
METHODS = ["local", "shared_adjacency", "local_dm"]


def edge_scores(section, q, graph):
    """Return mapping-invariant scores on edges with two reference annotations."""
    labels = section["meta"].label.reindex(section["ref"].index)
    valid = labels.notna().to_numpy()
    graph = graph.tocoo()
    use = (graph.row < graph.col) & valid[graph.row] & valid[graph.col]
    rr, cc = graph.row[use], graph.col[use]
    y = (labels.iloc[rr].to_numpy() != labels.iloc[cc].to_numpy()).astype(int)
    p = np.clip(1 - np.sum(q[rr] * q[cc], axis=1), 1e-6, 1 - 1e-6)
    return p, y


training_rows = {t: {m: [] for m in METHODS} for t in ["spinal", "cortex"]}
for dataset in DATASETS:
    tissue = "spinal" if dataset.startswith("spinal") else "cortex"
    params = selected[tissue]
    for train_split, target_split in [("development", "test"), ("test", "development")]:
        section = load_section(dataset, target_split)
        key = f"tune_{dataset}_{train_split}"
        training = read_pickle(CACHE / f"{key}_training.pkl")
        graph = section_graphs(section, features(section, training["columns"]))[0]
        files = {
            "local": f"{key}_local",
            "shared_adjacency": f"{key}_contact_{params['prior_cells']}_{params['gated_strength']}_{params['contact_strength']}",
            "local_dm": f"{key}_dm{dm_selected[tissue]}",
        }
        for method, filename in files.items():
            result = read_pickle(CACHE / f"{filename}.pkl")
            training_rows[tissue][method].append(edge_scores(section, result["q"], graph))

calibrators = {}
for tissue, methods in training_rows.items():
    calibrators[tissue] = {}
    for method, observations in methods.items():
        p = np.concatenate([obs[0] for obs in observations])
        y = np.concatenate([obs[1] for obs in observations])
        weights = np.concatenate([np.full(len(obs[0]), 1 / len(obs[0])) for obs in observations])
        weights /= weights.sum()
        x = logit(p)

        def objective(parameters):
            eta = parameters[0] * x + parameters[1]
            value = np.sum(weights * (np.logaddexp(0, eta) - y * eta))
            residual = weights * (expit(eta) - y)
            gradient = np.array([residual @ x, residual.sum()])
            return value, gradient

        result = minimize(
            objective,
            [1.0, 0.0],
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None), (None, None)],
            options={"ftol": 1e-12},
        )
        assert result.success
        calibrators[tissue][method] = dict(
            slope=float(result.x[0]),
            intercept=float(result.x[1]),
            constant_prevalence=float(weights @ y),
        )
(OUT / "boundary_calibrators.json").write_text(json.dumps(calibrators, indent=2))

records = []
for dataset in DATASETS:
    tissue = "spinal" if dataset.startswith("spinal") else "cortex"
    training = read_pickle(CACHE / f"final_{dataset}_training_17.pkl")
    for split in ["fresh1", "fresh2"]:
        section = load_section(dataset, split)
        graph = section_graphs(section, features(section, training["columns"]))[0]
        for method in METHODS:
            fitted = calibrators[tissue][method]
            for seed in SEEDS:
                prediction = np.load(
                    OUT / "predictions" / f"{section['name']}_{method}_{seed}.npz"
                )
                p, y = edge_scores(section, prediction["q"], graph)
                calibrated = expit(fitted["slope"] * logit(p) + fitted["intercept"])
                constant = np.full(len(y), fitted["constant_prevalence"])
                records.append(
                    dict(
                        name=section["name"],
                        tissue=tissue,
                        method=method,
                        seed=seed,
                        raw_brier=brier_score_loss(y, p),
                        calibrated_brier=brier_score_loss(y, calibrated),
                        constant_brier=brier_score_loss(y, constant),
                        raw_logloss=log_loss(y, p),
                        calibrated_logloss=log_loss(y, calibrated),
                        n_edges=len(y),
                    )
                )
pd.DataFrame(records).to_csv(OUT / "boundary_calibration.csv", index=False)
print(
    pd.DataFrame(records)
    .groupby(["tissue", "method"])[["raw_brier", "calibrated_brier", "constant_brier"]]
    .mean()
)
