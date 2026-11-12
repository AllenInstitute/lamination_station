"""Use the complete cortical neighborhood pool while retaining fixed evaluation cells."""

import torch  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from context_experiments import (
    OUT,
    SEEDS,
    cached,
    evaluate,
    get_section,
    representation,
    tag,
    vocabulary,
)


def source(dataset):
    if dataset == "cortex_merfish":
        d = pd.read_parquet(
            "/scratch/lamination_station_benchmark/cortex_merfish_obs.parquet",
            columns=["batch", "cell_type", "center_x", "center_y"],
        )
        d = d.rename(
            columns={"batch": "section", "cell_type": "type", "center_x": "x", "center_y": "y"}
        )
    else:
        d = pd.read_csv(
            "/data/hmba_xs_v1/spatial/human_merfish.tsv",
            sep="\t",
            usecols=["center_x", "center_y", "section", "roi", "supervised_subclass_2_level_2"],
        )
        d.index = [f"human_{i}" for i in d.index]
        d = d[d.roi == "V1"].rename(
            columns={"supervised_subclass_2_level_2": "type", "center_x": "x", "center_y": "y"}
        )
    d["section"] = d.section.astype(str)
    return d[d.type.notna() & np.isfinite(d[["x", "y"]]).all(axis=1)]


def full_features(s, pool):
    def compute():
        d = pool[pool.section == str(s["ref"].section.iloc[0])]
        assert d.index.is_unique
        focal = d.index.get_indexer(s["ref"].index)
        assert (focal >= 0).all()
        xy = d[["x", "y"]].to_numpy()
        np.testing.assert_allclose(xy[focal], s["ref"][["x", "y"]])
        tree = NearestNeighbors(n_neighbors=13).fit(xy)
        distances, indices = tree.kneighbors(xy)
        # Full-resolution physical graph establishes disconnected tissue components.
        rr = np.repeat(np.arange(len(d)), 12)
        cc = indices[:, 1:].ravel()
        keep = distances[:, 1:].ravel() <= np.quantile(distances[:, -1], 0.95)
        graph = sparse.csr_matrix(
            (np.ones(keep.sum()), (rr[keep], cc[keep])), shape=(len(d), len(d))
        )
        graph = graph.minimum(graph.T)
        _, pieces = sparse.csgraph.connected_components(graph, directed=False)
        raw = tree.kneighbors(xy[focal], n_neighbors=513, return_distance=False)
        ix = np.array([row[row != f][:512] for f, row in zip(focal, raw, strict=True)])
        valid = pieces[ix] == pieces[focal, None]
        codes = pd.Categorical(d.type).codes
        result = {}
        for k in [8, 32, 128, 512]:
            x = np.zeros((len(focal), codes.max() + 1))
            np.add.at(
                x,
                (
                    np.repeat(np.arange(len(focal)), k)[valid[:, :k].ravel()],
                    codes[ix[:, :k]][valid[:, :k]],
                ),
                1,
            )
            result[k] = np.sqrt(x / np.maximum(x.sum(1, keepdims=True), 1))
        return dict(
            blocks=result, n_pool=len(d), n_focal=len(focal), n_components=int(pieces.max() + 1)
        )

    return cached("full_density_" + s["name"], compute)


def predict(s, spec, seed, pool):
    def compute():
        base = representation(s, vocabulary([s]), "gradient") / np.sqrt(3)
        extra = full_features(s, pool)["blocks"]
        scales = [8, 32, 128] if spec["kind"] == "micro" else [32, 128, 512]
        x = np.concatenate([base] + [extra[k] / np.sqrt(3) for k in scales], axis=1)
        labels = KMeans(n_clusters=8, n_init=10, random_state=seed).fit_predict(x)
        return np.eye(8)[labels]

    return cached("prediction_" + s["name"] + "_" + tag(spec) + f"_seed{seed}", compute)


def run(stage):
    records = []
    if stage == "screen":
        candidates = [dict(method="density", kind=k, k=8) for k in ["micro", "meso"]]
        splits = ["development", "test"]
        seeds = [17]
        (OUT / "density_protocol.json").write_text(
            json.dumps(
                dict(
                    candidates=candidates,
                    selection="Mean ARI on initial development/test sections, both cortical datasets; freeze before historical replication.",
                    description="Retain gradient-adjusted features, add complete-pool context8/32/128 or32/128/512 at original sampled reference cells; component-isolated physical neighborhoods; KMeans8.",
                    caveat="All cortical labels are composition-derived proxies; later sections were previously evaluated.",
                ),
                indent=2,
            )
        )
    else:
        candidates = [json.loads((OUT / "density_selection.json").read_text())]
        splits = ["fresh1", "fresh2"]
        seeds = SEEDS
    for dataset in ["cortex_human", "cortex_merfish"]:
        pool = source(dataset)
        for split in splits:
            s = get_section(dataset, split)
            for spec in candidates:
                for seed in seeds:
                    records.append(evaluate(s, predict(s, spec, seed, pool), spec, seed, stage))
            pd.DataFrame(records).to_csv(OUT / f"density_{stage}.csv", index=False)
    if stage == "screen":
        best = pd.DataFrame(records).groupby("candidate").ari.mean().idxmax()
        (OUT / "density_selection.json").write_text(
            json.dumps(next(s for s in candidates if tag(s) == best), indent=2)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "replication"])
    args = parser.parse_args()
    run(args.stage)
