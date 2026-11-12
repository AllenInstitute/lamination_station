"""Local composition-Jacobian neighborhoods: follow layers without a global axis."""

import torch  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from context_experiments import (
    DATASETS,
    OUT,
    SEEDS,
    cached,
    evaluate,
    get_section,
    representation,
    tag,
    vocabulary,
)
from incremental_domains import features, section_graphs


def tangent_counts(xy, composition, codes, pieces, ratio):
    """Estimate signed per-type derivatives and penalize travel across their normal."""
    neighbor_count = min(256, len(xy) - 1)
    distance, raw = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(xy).kneighbors(xy)
    ix = np.array([row[row != i][:neighbor_count] for i, row in enumerate(raw)])
    displacement = xy[ix] - xy[:, None]
    same = pieces[ix] == pieces[:, None]
    local = min(40, neighbor_count)
    d = displacement[:, :local]
    scale = np.maximum(np.linalg.norm(d[:, -1], axis=1), 1e-12)
    d = d / scale[:, None, None]
    weights = np.exp(-np.sum(d * d, axis=2)) * same[:, :local]
    diff = composition[ix[:, :local]] - composition[:, None]
    xx = np.einsum("nki,nk,nkj->nij", d, weights, d) + np.eye(2)[None] * 1e-3
    xy_cross = np.einsum("nki,nk,nkt->nit", d, weights, diff)
    jacobian = np.linalg.solve(xx, xy_cross)
    tensor = jacobian @ jacobian.transpose(0, 2, 1)
    eigenvalue, eigenvector = np.linalg.eigh(tensor)
    normal = eigenvector[:, :, -1]
    confidence = (eigenvalue[:, -1] - eigenvalue[:, 0]) / (eigenvalue.sum(1) + 1e-10)
    crossing = np.einsum("nkd,nd->nk", displacement, normal)
    metric = np.sum(displacement**2, axis=2) + (ratio**2 - 1) * confidence[:, None] * crossing**2
    order = np.argsort(np.where(same, metric, np.inf), axis=1)
    selected = np.take_along_axis(ix, order, axis=1)
    valid = np.take_along_axis(same, order, axis=1)
    result = []
    for k in [15, 30, 60]:
        count = np.zeros((len(xy), codes.max() + 1), float)
        use = min(k, neighbor_count)
        np.add.at(
            count,
            (
                np.repeat(np.arange(len(xy)), use)[valid[:, :use].ravel()],
                codes[selected[:, :use]][valid[:, :use]],
            ),
            1,
        )
        result.append(count)
    return result


def predict(s, spec, seed):
    def compute():
        def prepare():
            columns = vocabulary([s])
            rings = features(s, columns)
            graph = section_graphs(s, rings)[0]
            _, pieces = connected_components(graph, directed=False)
            p = s["counts"][30].to_numpy(dtype=float) / 30
            codes = pd.Categorical(s["ref"].type).codes
            counts = tangent_counts(
                s["ref"][["x", "y"]].to_numpy(), p, codes, pieces, spec["ratio"]
            )
            return np.concatenate(
                [representation(s, columns, "gradient") / np.sqrt(3)]
                + [
                    np.sqrt(x / np.maximum(x.sum(1, keepdims=True), 1)) / np.sqrt(3)
                    for x in counts
                ],
                axis=1,
            )

        x = cached("tangent_features_" + s["name"] + f"_ratio{spec['ratio']}", prepare)
        labels = KMeans(n_clusters=8, n_init=10, random_state=seed).fit_predict(x)
        return np.eye(8)[labels]

    return cached("prediction_" + s["name"] + "_" + tag(spec) + f"_seed{seed}", compute)


def run(stage):
    records = []
    if stage == "screen":
        candidates = [dict(method="tangent", kind="jacobian", ratio=r, k=8) for r in [3, 6]]
        splits = ["development", "test"]
        seeds = [17]
        (OUT / "geometry_protocol.json").write_text(
            json.dumps(
                dict(
                    candidates=candidates,
                    description="Weighted local least-squares per-type derivatives of existing30-neighbor composition; principal eigenvector of J J^T gives unsigned normal; eigengap controls anisotropy; select15/30/60 neighbors from256-candidate pool by elliptic metric; retain original gradient features as equal-weight family.",
                    selection="Separate spinal/cortex mean development/test ARI; freeze before confirmation.",
                    invariance="Euclidean rigid transforms; disconnected physical components are never mixed in new neighborhoods.",
                ),
                indent=2,
            )
        )
    else:
        selection = json.loads((OUT / "geometry_selection.json").read_text())
        splits = (
            ["fresh1", "fresh2"] if stage == "replication" else ["confirmation1", "confirmation2"]
        )
        seeds = SEEDS
    for dataset in DATASETS if stage != "confirmation" else DATASETS[:3]:
        if stage != "screen":
            candidates = [selection[dataset.split("_")[0]]]
        for split in splits:
            s = get_section(dataset, split)
            for spec in candidates:
                for seed in seeds:
                    records.append(evaluate(s, predict(s, spec, seed), spec, seed, stage))
            pd.DataFrame(records).to_csv(OUT / f"geometry_{stage}.csv", index=False)
    if stage == "screen":
        frame = pd.DataFrame(records)
        selection = {}
        for tissue in ["spinal", "cortex"]:
            winner = frame[frame.tissue == tissue].groupby("candidate").ari.mean().idxmax()
            selection[tissue] = next(s for s in candidates if tag(s) == winner)
        (OUT / "geometry_selection.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "replication", "confirmation"])
    args = parser.parse_args()
    run(args.stage)
