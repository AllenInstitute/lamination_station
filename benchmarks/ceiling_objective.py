"""Does optimizing the clustering objective move away from an anatomical initialization?"""

import torch  # noqa: F401

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from ceiling_diagnostics import OUT, SPECIES, SPLITS, cached, get_section, get_data
from context_experiments import vocabulary
from incremental_domains import features


def em(rings, profiles, weights, y, case_weights, uniform=False):
    k = len(weights)
    t = rings[0].shape[1]
    history = []
    for iteration in range(1001):
        unary = sum(r @ np.log(p).T for r, p in zip(rings, profiles, strict=True)) + np.log(
            weights
        )
        norm = logsumexp(unary, axis=1)
        q = np.exp(unary - norm[:, None])
        objective = float(
            case_weights @ norm
            + sum(np.log(p).sum() / t for p in profiles)
            + np.log(weights).sum() / k
        )
        history.append(
            dict(iteration=iteration, objective=objective, ari=adjusted_rand_score(y, q.argmax(1)))
        )
        if len(history) > 1:
            assert objective >= history[-2]["objective"] - 1e-6
            if abs(objective - history[-2]["objective"]) < 1e-7 * (1 + abs(objective)):
                break
        weighted = q * case_weights[:, None]
        weights = np.ones(k) / k if uniform else weighted.sum(0) + 1 / k
        weights = weights / weights.sum()
        profiles = [weighted.T @ r + 1 / t for r in rings]
        profiles = [p / p.sum(1, keepdims=True) for p in profiles]
    return dict(history=pd.DataFrame(history), prediction=q.argmax(1), converged=iteration < 1000)


records = []
histories = []
for species in SPECIES:
    for split in SPLITS:
        s = get_section("spinal_" + species, split)
        d = get_data(s)
        valid = d["y"].notna().to_numpy()
        y = d["y"][valid].astype(str).to_numpy()
        classes, encoded = np.unique(y, return_inverse=True)
        onehot = np.eye(len(classes))[encoded]
        rings = [r[valid] for r in features(s, vocabulary([s]))]
        profiles = [onehot.T @ r + 1 / r.shape[1] for r in rings]
        profiles = [p / p.sum(1, keepdims=True) for p in profiles]
        weights = np.bincount(encoded) + 1 / len(classes)
        weights = weights / weights.sum()
        area = d["distance"][valid, 29] ** 2
        area = np.clip(area, np.quantile(area, 0.05), np.quantile(area, 0.95))
        modes = {
            "unweighted": np.ones(len(y)),
            "uniform_component_prior": np.ones(len(y)),
            "reference_balanced": 1 / np.bincount(encoded)[encoded],
            "inverse_density": area,
        }
        for mode, case in modes.items():
            case = case / case.mean()
            initial_weights = (
                np.ones(len(classes)) / len(classes)
                if mode == "uniform_component_prior"
                else weights
            )
            fit = cached(
                "objective_" + s["name"] + "_" + mode,
                lambda: em(
                    rings,
                    [p.copy() for p in profiles],
                    initial_weights.copy(),
                    y,
                    case,
                    mode == "uniform_component_prior",
                ),
            )
            h = fit["history"].copy()
            h["name"] = s["name"]
            h["species"] = species
            h["mode"] = mode
            histories.append(h)
            records.append(
                dict(
                    name=s["name"],
                    species=species,
                    model="ring",
                    mode=mode,
                    initial_ari=h.ari.iloc[0],
                    final_ari=h.ari.iloc[-1],
                    objective_gain=h.objective.iloc[-1] - h.objective.iloc[0],
                    n_iter=len(h) - 1,
                    converged=fit["converged"],
                )
            )
        x = np.concatenate([np.sqrt(r / r.sum(1, keepdims=True)) for r in rings], axis=1)
        centers = np.stack([x[encoded == i].mean(0) for i in range(len(classes))])
        dist = ((x[:, None] - centers[None]) ** 2).sum(2)
        initial = dist.argmin(1)
        model = KMeans(n_clusters=len(classes), init=centers, n_init=1, random_state=17).fit(x)
        records.append(
            dict(
                name=s["name"],
                species=species,
                model="kmeans",
                mode="unweighted",
                initial_ari=adjusted_rand_score(y, initial),
                final_ari=adjusted_rand_score(y, model.labels_),
                objective_gain=dist.min(1).sum() - model.inertia_,
                n_iter=model.n_iter_,
                converged=True,
            )
        )
        assert records[-1]["objective_gain"] >= -1e-7
        pd.DataFrame(records).to_csv(OUT / "objective_drift.csv", index=False)
        pd.concat(histories).to_parquet(OUT / "objective_histories.parquet", index=False)
        print("OBJECTIVE", s["name"], flush=True)
