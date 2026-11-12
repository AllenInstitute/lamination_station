"""Check information discarded by coarse cell-type composition, without transformers."""

import torch  # noqa: F401

import argparse
import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


def extract():
    """Stream expression columns, selecting only benchmark neurons; omit region metadata."""
    sections = [
        get_section("spinal_" + species, split)
        for species in ["human", "macaque", "mouse"]
        for split in ["development", "test", "fresh1", "fresh2", "confirmation1", "confirmation2"]
    ]
    ids = pd.MultiIndex.from_tuples(
        [(str(row.section), str(cell)) for s in sections for cell, row in s["ref"].iterrows()]
    )
    assert ids.is_unique
    source = "/data/SpinalCord/review_integrated/nnn_clusters/spatial/all_species.allinone.h5ad"
    with h5py.File(source) as f:
        source_ids = pd.MultiIndex.from_arrays(
            [
                np.asarray(ad.io.read_elem(f["obs/_section"])).astype(str),
                f["obs/_index"].asstr()[:],
            ]
        )
        assert source_ids.is_unique and ids.isin(source_ids).all()
        mapping = ids.get_indexer(source_ids)
        xx = f["X"]
        n_genes = xx.attrs["shape"][1]
        expression = np.zeros((len(ids), n_genes), np.float32)
        ptr = xx["indptr"][:]
        for g in range(n_genes):
            rows = mapping[xx["indices"][ptr[g] : ptr[g + 1]]]
            values = xx["data"][ptr[g] : ptr[g + 1]]
            keep = rows >= 0
            expression[rows[keep], g] = values[keep]
        fine = ad.io.read_elem(f["obs/consensus_final_cluster"])
        fine = (
            pd.Series(fine, index=source_ids)
            .reindex(ids)
            .astype("string")
            .fillna("__unknown_fine")
        )
        genes = f["var/_index"].asstr()[:]
    for s in sections:
        keys = pd.MultiIndex.from_arrays(
            [s["ref"].section.astype(str), s["ref"].index.astype(str)]
        )
        ix = ids.get_indexer(keys)
        cached(
            "molecular_" + s["name"],
            lambda: dict(
                expression=expression[ix],
                fine=fine.iloc[ix].to_numpy(),
                genes=genes,
                index=s["ref"].index,
            ),
        )
        print("EXTRACTED", s["name"], expression[ix].shape, flush=True)


def extra_features(s, kind):
    def compute():
        data = cached("molecular_" + s["name"], lambda: None)
        assert data["index"].equals(s["ref"].index)
        xy = s["ref"][["x", "y"]].to_numpy()
        raw = NearestNeighbors(n_neighbors=129).fit(xy).kneighbors(xy, return_distance=False)
        ix = np.array([row[row != i][:128] for i, row in enumerate(raw)])
        from incremental_domains import features, section_graphs
        from scipy.sparse.csgraph import connected_components

        graph = section_graphs(s, features(s, vocabulary([s])))[0]
        _, piece = connected_components(graph, directed=False)
        same_piece = piece[ix] == piece[:, None]
        if kind == "fine":
            labels, codes = np.unique(data["fine"].astype(str), return_inverse=True)
            matrix = np.eye(len(labels))[codes]
            blocks = []
            for k in [16, 32, 64, 128]:
                avg = np.zeros_like(matrix)
                for rank in range(k):
                    avg += matrix[ix[:, rank]] * same_piece[:, rank, None]
                avg /= np.maximum(same_piece[:, :k].sum(1, keepdims=True), 1)
                blocks.append(np.sqrt(avg) / 2)
            return np.concatenate(blocks, axis=1)
        x = data["expression"].astype(float)
        assert (x >= 0).all()
        x = np.log1p(1e4 * x / np.maximum(x.sum(1, keepdims=True), 1))
        x = x[:, (x > 0).mean(0) >= 0.02]
        if kind == "within_type":
            codes = pd.Categorical(s["ref"].type).codes
            for c in np.unique(codes):
                keep = codes == c
                x[keep] -= x[keep].mean(0)
        x = (x - x.mean(0)) / np.maximum(x.std(0), 0.1)
        blocks = []
        for k in [30, 90]:
            # Average only within the same disconnected physical component.
            valid = same_piece[:, :k]
            avg = np.zeros_like(x)
            for rank in range(k):
                avg += x[ix[:, rank]] * valid[:, rank, None]
            avg /= np.maximum(valid.sum(1, keepdims=True), 1)
            z = PCA(n_components=min(20, avg.shape[1]), random_state=17).fit_transform(avg)
            z /= np.sqrt(np.mean(np.sum(z * z, axis=1)) + 1e-8)
            blocks.append(z / np.sqrt(2))
        return np.concatenate(blocks, axis=1)

    return cached("extra_" + s["name"] + "_" + kind, compute)


def specs():
    return [
        dict(method="molecular", kind=k, weight=w, k=8)
        for k in ["fine", "expression", "within_type"]
        for w in [0.3, 1.0]
    ]


def prediction(s, spec, seed):
    def compute():
        base = representation(s, vocabulary([s]), "gradient") / np.sqrt(3)
        extra = extra_features(s, spec["kind"])
        x = np.concatenate([base, extra * spec["weight"]], axis=1)
        labels = KMeans(n_clusters=spec["k"], n_init=10, random_state=seed).fit_predict(x)
        return np.eye(spec["k"])[labels]

    return cached("prediction_" + s["name"] + "_" + tag(spec) + f"_seed{seed}", compute)


def run(stage):
    records = []
    if stage == "screen":
        candidates = specs()
        splits = ["development", "test"]
        seeds = [17]
        (OUT / "molecular_protocol.json").write_text(
            json.dumps(
                dict(
                    candidates=candidates,
                    selection="Mean ARI on initial development/test sections only; freeze one candidate before confirmation.",
                    inputs="Raw counts normalized per cell to10000 and log1p, genes detected>=2%; within-type variant subtracts section-specific cell-type mean; scale genes then average30/90 physical neighbors; PCA20 and normalized variance per scale.",
                    distinction="Cheap information-content tests inspired by CellTransformer; not implementations of CellTransformer or evidence about its performance.",
                    scope="Spinal only; same neuron subset and evaluation cells as baseline.",
                ),
                indent=2,
            )
        )
    else:
        candidates = [json.loads((OUT / "molecular_selection.json").read_text())]
        splits = (
            ["fresh1", "fresh2"] if stage == "replication" else ["confirmation1", "confirmation2"]
        )
        seeds = SEEDS
    for species in ["human", "macaque", "mouse"]:
        for split in splits:
            s = get_section("spinal_" + species, split)
            for spec in candidates:
                for seed in seeds:
                    records.append(evaluate(s, prediction(s, spec, seed), spec, seed, stage))
            pd.DataFrame(records).to_csv(OUT / f"molecular_{stage}.csv", index=False)
    if stage == "screen":
        best = pd.DataFrame(records).groupby("candidate").ari.mean().idxmax()
        selected = next(s for s in candidates if tag(s) == best)
        (OUT / "molecular_selection.json").write_text(json.dumps(selected, indent=2))
        print("FROZEN", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "screen", "replication", "confirmation"])
    args = parser.parse_args()
    if args.stage == "extract":
        extract()
    else:
        run(args.stage)
