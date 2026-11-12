"""Development-only selection and separate-section validation of HRCHY and BANKSY."""

import torch
import contextlib
import importlib.metadata
import json
import pickle
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from hrchy_cytocommunity.models.HRCHYCytoCommunity import HRCHYCytoCommunityGrand
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy.main import LeidenPartition

torch.set_num_threads(2)
root = Path("/scratch/lamination_station_external")
out = Path("/results/lamination_station_external")
old = Path("/scratch/lamination_station_benchmark")
previous = Path("/scratch/lamination_station_investigation")
method, phase = sys.argv[1:3]
subset = sys.argv[3] if len(sys.argv) > 3 else ""
suffix = f"_{subset}" if subset else ""
resolutions = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def score(labels, *, ref, meta, name, method, seed, settings, seconds):
    """Evaluate the same fixed focal cells used by previous comparisons."""
    ix = ref.index.get_indexer(meta.index)
    assert (ix >= 0).all() and len(labels) == len(ref)
    mask = meta.label.notna().to_numpy()
    labels = np.asarray(labels)[ix]
    return dict(
        name=name,
        dataset=name.rsplit("_", 1)[0],
        method=method,
        seed=seed,
        settings=json.dumps(settings, sort_keys=True),
        seconds=seconds,
        ari=adjusted_rand_score(meta.label[mask], labels[mask]),
        ami=adjusted_mutual_info_score(meta.label[mask], labels[mask]),
        occupied=len(np.unique(labels)),
        n_reference=len(ref),
        n_scored=int(mask.sum()),
    )


def hrchy(ref, *, name, seed, settings):
    """Use the released full consistency-regularized model without anatomical labels."""
    target = root / f"{name}_hrchy_{settings['near']}_c{settings['coarse']}_seed{seed}.pkl"
    if target.exists():
        return pickle.load(target.open("rb"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    near = int(np.sqrt(len(ref))) if settings["near"] == "sqrt" else int(settings["near"])
    g = kneighbors_graph(ref[["x", "y"]], n_neighbors=near, include_self=False)
    g = g.maximum(g.T).tocoo()
    features = pd.get_dummies(ref.type).to_numpy(dtype=np.float32)
    data = Data(
        x=torch.from_numpy(features),
        edge_index=torch.tensor(np.vstack([g.row, g.col]), dtype=torch.long),
    )
    directory = root / target.stem
    start = time.perf_counter()
    fit = HRCHYCytoCommunityGrand(
        data,
        num_tcn1=8,
        num_tcn2=settings["coarse"],
        cell_meta=ref[["x", "y", "type"]].copy(),
        num_epoch=1500,
        device="cuda",
        gt_coarse=False,
        gt_fine=False,
    )
    fit.train(str(directory), output=False, vis_while_training=False)
    q, labels, _, _ = fit.predict(save=False)
    assert np.isfinite(q).all()
    result = {
        "labels": labels.ravel(),
        "seconds": time.perf_counter() - start,
        "near_actual": near,
    }
    pickle.dump(result, target.open("wb"))
    del fit, data
    torch.cuda.empty_cache()
    return result


def banksy(ref, *, name, seed, near, lambdas):
    """Apply official BANKSY spatial/AGF features to one-hot cell-type indicators."""
    embeddings = {}
    missing = [lam for lam in lambdas if not (root / f"{name}_banksy_n{near}_l{lam}.npy").exists()]
    if missing:
        a = ad.AnnData(
            X=pd.get_dummies(ref.type).to_numpy(dtype=float), obs=ref[["x", "y", "type"]].copy()
        )
        a.obsm["spatial"] = ref[["x", "y"]].to_numpy()
        with (
            (root / f"{name}_banksy_n{near}_features.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            b = initialize_banksy(
                a,
                coord_keys=("x", "y", "spatial"),
                num_neighbours=near,
                max_m=1,
                plt_edge_hist=False,
                plt_nbr_weights=False,
                plt_agf_angles=False,
                plt_theta=False,
            )
            b, _ = generate_banksy_matrix(a, b, lambda_list=missing, max_m=1, verbose=False)
            for lam in missing:
                x = b["scaled_gaussian"][lam]["adata"].X
                # BANKSY's PCA helper replaces undefined zero-variance feature z-scores with zero.
                x = np.nan_to_num(x, nan=0.0)
                assert np.isfinite(x).all()
                embedding = PCA(
                    n_components=min(20, x.shape[1], len(ref) - 1), svd_solver="full"
                ).fit_transform(x)
                np.save(root / f"{name}_banksy_n{near}_l{lam}.npy", embedding)
    for lam in lambdas:
        embeddings[lam] = np.load(root / f"{name}_banksy_n{near}_l{lam}.npy")
    results = []
    for lam, embedding in embeddings.items():
        target = root / f"{name}_banksy_native_n{near}_l{lam}_seed{seed}.pkl"
        if target.exists():
            results.extend(pickle.load(target.open("rb")))
            continue
        start = time.perf_counter()
        predictions = [
            (
                "banksy_kmeans",
                None,
                KMeans(n_clusters=8, n_init=20, random_state=seed).fit_predict(embedding),
            )
        ]
        with (root / f"{target.stem}.log").open("w") as log, contextlib.redirect_stdout(log):
            partitioner = LeidenPartition(embedding, num_nn=50, verbose=False)
            for resolution in resolutions:
                labels, _ = partitioner.partition(
                    resolution=resolution, n_iterations=-1, seed=seed
                )
                predictions.append(("banksy_leiden", resolution, labels.dense.copy()))
        batch = [
            dict(
                method=m,
                labels=labels,
                settings=dict(near=near, lam=lam, resolution=res),
                seconds=time.perf_counter() - start,
            )
            for m, res, labels in predictions
        ]
        pickle.dump(batch, target.open("wb"))
        results.extend(batch)
    return results


records = []
if phase == "select":
    frames = [
        pd.read_csv(out / f"hrchy_development_spinal_{s}.csv")
        for s in ["human", "macaque", "mouse"]
    ]
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 6 and frame.name.nunique() == 3
    frame.to_csv(out / "hrchy_development.csv", index=False)
    selected = frame.groupby("settings").ari.mean().idxmax()
    (out / "hrchy_selected.json").write_text(json.dumps({"hrchy": json.loads(selected)}, indent=2))
    print(selected)
    sys.exit(0)
if phase == "development":
    paths = sorted(old.glob("spinal_*_development.pkl"))
    seeds = [17]
    configs = [dict(near=k, coarse=2) for k in [30, "sqrt"]]
    banksy_near = [15, 30, 60]
    lambdas = [0.8, 0.95, 1.0]
else:
    paths = sorted(previous.glob("spinal_*_fresh[12].pkl"))
    seeds = [17, 29, 43]
    selections = json.loads((out / f"{method}_selected.json").read_text())
    configs = [selections["hrchy"]] if method == "hrchy" else []
    banksy_near = sorted(set(v["near"] for v in selections.values())) if method == "banksy" else []
    lambdas = sorted(set(v["lam"] for v in selections.values())) if method == "banksy" else []

for path in [p for p in paths if p.stem.startswith(subset)]:
    if phase == "development":
        _, meta, ref = pickle.load(path.open("rb"))
    else:
        ref, _, meta = pickle.load(path.open("rb"))
    name = path.stem
    for seed in seeds:
        if method == "hrchy":
            for config in configs:
                result = hrchy(ref, name=name, seed=seed, settings=config)
                records.append(
                    score(
                        result["labels"],
                        ref=ref,
                        meta=meta,
                        name=name,
                        method=method,
                        seed=seed,
                        settings=config,
                        seconds=result["seconds"],
                    )
                )
                print(records[-1], flush=True)
        else:
            for near in banksy_near:
                for result in banksy(ref, name=name, seed=seed, near=near, lambdas=lambdas):
                    if (
                        phase != "development"
                        and result["settings"] != selections[result["method"]]
                    ):
                        continue
                    records.append(
                        score(
                            result["labels"],
                            ref=ref,
                            meta=meta,
                            name=name,
                            method=result["method"],
                            seed=seed,
                            settings=result["settings"],
                            seconds=result["seconds"],
                        )
                    )
        pd.DataFrame(records).to_csv(out / f"{method}_{phase}{suffix}.csv", index=False)
        print(name, seed, "complete", flush=True)

if phase == "development":
    selections = {}
    for m, frame in pd.DataFrame(records).groupby("method"):
        means = frame.groupby("settings").ari.mean().sort_values(ascending=False)
        selections[m] = json.loads(means.index[0])
    (out / f"{method}_selected{suffix}.json").write_text(json.dumps(selections, indent=2))
    print(selections, flush=True)
(out / f"{method}_{phase}_versions.json").write_text(
    json.dumps(
        {
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "torch-geometric",
                "hrchy-cytocommunity",
                "pybanksy",
                "numpy",
                "scipy",
                "scanpy",
                "anndata",
                "scikit-learn",
                "leidenalg",
            ]
        },
        indent=2,
    )
)
