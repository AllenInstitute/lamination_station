"""Broader-context, patch segmentation, and explicitly supervised transfer pilots."""

import torch  # noqa: F401

import argparse
import heapq
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

from incremental_domains import DATASETS, features, load_section, score, section_graphs, vocabulary
from incremental_models import fit_mixture

ROOT = Path("/scratch/lamination_station_context")
OUT = Path("/results/lamination_station_context")
ROOT.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)
SEEDS = [17, 29, 43]
SCALES = [8, 16, 32, 64, 128, 256]


def cached(name, function):
    path = ROOT / (name + ".pkl")
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    result = function()
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle)
    temporary.replace(path)
    return result


def prepare_confirmation():
    """Reserve sections five and six; truth is stored separately from model inputs."""
    from lamination_station.preprocessing import calculate_comp_grads

    obs = pd.read_parquet("/scratch/lamination_station_fixes/three_species_obs.parquet")
    manifest = []
    for species in ["human", "macaque", "mouse"]:
        d = obs[(obs.species == species) & (obs.Neighborhood == "neuron")].copy()
        d["label"] = d.rexed_lamina.astype("string").where(d.rexed_lamina_source == "polygon")
        d = d.rename(
            columns={
                "_plot_x": "x",
                "_plot_y": "y",
                "_section": "section",
                "consensus_manual_group_propagated": "type",
            }
        )
        eligible = d[d.label.notna()].groupby("section", observed=True).size()
        for j, section in enumerate(sorted(eligible[eligible >= 500].index)[4:6]):
            name = f"spinal_{species}_confirmation{j + 1}"
            ref = d.loc[
                (d.section == section) & d.type.notna(), ["x", "y", "type", "section", "label"]
            ].copy()
            ref = ref[np.isfinite(ref[["x", "y"]]).all(axis=1)]
            ref["type"] = ref.type.astype("category").cat.remove_unused_categories()
            select = np.sort(
                np.random.default_rng(2027).choice(len(ref), min(2048, len(ref)), replace=False)
            )
            meta = ref[["label"]].iloc[select].copy()
            ref = ref.drop(columns="label")

            def compute():
                counts = {
                    k: calculate_comp_grads(
                        ref, "x", "y", "type", near_k=k, far_k=90, sample_col="section"
                    )[5]
                    for k in [15, 30, 60]
                }
                return dict(
                    ref=ref, counts=counts, meta=meta, name=name, dataset=f"spinal_{species}"
                )

            s = cached("data_" + name, compute)
            manifest.append(
                dict(name=name, section=str(section), n_reference=len(ref), n_evaluation=len(meta))
            )
            print("PREPARED", s["name"], flush=True)
    (OUT / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2))


def get_section(dataset, split):
    if split.startswith("confirmation"):
        with (ROOT / f"data_{dataset}_{split}.pkl").open("rb") as handle:
            return pickle.load(handle)
    return load_section(dataset, split)


def context_counts(section, columns):
    """Count neighbor types at several ranks without using anatomical annotations."""

    def compute():
        ref = section["ref"]
        xy = ref[["x", "y"]].to_numpy()
        distance, ix = (
            NearestNeighbors(n_neighbors=min(len(ref), max(SCALES) + 1)).fit(xy).kneighbors(xy)
        )
        ix = np.array([row[row != i][: max(SCALES)] for i, row in enumerate(ix)])
        # Isolate disconnected physical components before taking long-range contexts.
        physical = section_graphs(section, features(section, vocabulary([section])))[0]
        _, pieces = sparse.csgraph.connected_components(physical, directed=False)
        types = ref.type.astype(str).to_numpy()
        all_types = sorted(set(types))
        lookup = {v: i for i, v in enumerate(all_types)}
        codes = np.array([lookup[t] for t in types])
        result = {}
        for k in SCALES:
            neighbors = ix[:, : min(k, ix.shape[1])]
            valid = pieces[neighbors] == pieces[:, None]
            counts = np.zeros((len(ref), len(all_types)), dtype=np.float32)
            np.add.at(
                counts,
                (
                    np.repeat(np.arange(len(ref)), neighbors.shape[1])[valid.ravel()],
                    codes[neighbors][valid],
                ),
                1,
            )
            result[k] = pd.DataFrame(counts, index=ref.index, columns=all_types)
        return result

    frames = cached("context_" + section["name"], compute)
    aligned = []
    for k in SCALES:
        f = frames[k].rename(columns=lambda c: c + "_count")
        x = f.reindex(columns=columns, fill_value=0).to_numpy(dtype=float)
        x[:, -1] = f[f.columns.difference(columns)].sum(1)
        aligned.append(x)
    return aligned


def representation(section, columns, kind):
    rings = features(section, columns)
    grad = np.concatenate([np.sqrt(x / x.sum(1, keepdims=True)) for x in rings], axis=1)
    if kind == "gradient":
        return grad
    counts = context_counts(section, columns)
    if kind == "small":
        counts = counts[:3]
    blocks = []
    for x in counts:
        n = x.sum(1, keepdims=True)
        if kind == "residual":
            p = (x.sum(0, keepdims=True) + 0.5) / (x.sum() + 0.5 * x.shape[1])
            b = np.clip((x - n * p) / np.sqrt(n * p * (1 - p) + 1), -5, 5)
            b /= np.sqrt(np.mean(np.sum((b - b.mean(0)) ** 2, axis=1)) + 1e-8)
        else:
            b = np.sqrt(x / np.maximum(n, 1))
        blocks.append(b)
    # Equal total block weight for original gradients and new context family.
    return np.concatenate([grad / np.sqrt(3)] + [b / np.sqrt(len(blocks)) for b in blocks], axis=1)


def spatial_patches(x, graph, n_patches):
    """Ward merges only adjacent patches; disconnected components stay separate."""
    n = len(x)
    sums = np.zeros((2 * n, x.shape[1]), dtype=float)
    sums[:n] = x
    size = np.zeros(2 * n, dtype=int)
    size[:n] = 1
    active = np.zeros(2 * n, dtype=bool)
    active[:n] = True
    parent = np.arange(2 * n)
    neighbors = [set(graph.indices[graph.indptr[i] : graph.indptr[i + 1]]) for i in range(n)]
    neighbors.extend(set() for _ in range(n))
    heap = []

    def cost(a, b):
        return (
            size[a]
            * size[b]
            / (size[a] + size[b])
            * np.sum((sums[a] / size[a] - sums[b] / size[b]) ** 2)
        )

    for a in range(n):
        for b in neighbors[a]:
            if a < b:
                heapq.heappush(heap, (cost(a, b), a, b))
    count, new = n, n
    while count > n_patches and heap:
        _, a, b = heapq.heappop(heap)
        if not active[a] or not active[b]:
            continue
        active[a] = active[b] = False
        active[new] = True
        parent[a] = parent[b] = new
        size[new] = size[a] + size[b]
        sums[new] = sums[a] + sums[b]
        neighbors[new] = (neighbors[a] | neighbors[b]) - {a, b}
        for c in neighbors[new]:
            neighbors[c].discard(a)
            neighbors[c].discard(b)
            neighbors[c].add(new)
            heapq.heappush(heap, (cost(new, c), c, new))
        new += 1
        count -= 1
    for i in range(new - 1, -1, -1):
        parent[i] = parent[parent[i]]
    ids, inverse = np.unique(parent[:n], return_inverse=True)
    return sums[ids] / size[ids, None], size[ids], inverse


def candidate_specs():
    return (
        [dict(method="ring", kind="gradient", k=8)]
        + [
            dict(method="kmeans", kind=kind, k=k)
            for kind in ["gradient", "small", "broad", "residual"]
            for k in [8, 12]
        ]
        + [
            dict(method="patch", kind=kind, k=8, patches=p)
            for kind in ["gradient", "broad"]
            for p in [64, 192]
        ]
    )


def tag(spec):
    return "_".join(str(spec[k]) for k in sorted(spec))


def unsupervised(section, spec, seed):
    def compute():
        columns = vocabulary([section])
        rings = features(section, columns)
        if spec["method"] == "ring":
            return fit_mixture(rings, seed=seed)["q"]
        x = representation(section, columns, spec["kind"])
        if spec["method"] == "patch":
            patches, weights, ix = cached(
                "patches_" + section["name"] + "_" + tag(spec),
                lambda: spatial_patches(x, section_graphs(section, rings)[0], spec["patches"]),
            )
            labels = (
                KMeans(n_clusters=spec["k"], n_init=10, random_state=seed)
                .fit(patches, sample_weight=weights)
                .labels_[ix]
            )
        else:
            labels = KMeans(n_clusters=spec["k"], n_init=10, random_state=seed).fit_predict(x)
        return np.eye(spec["k"])[labels]

    return cached("prediction_" + section["name"] + "_" + tag(spec) + f"_seed{seed}", compute)


def transfer(training, target, kind, leaf, seed):
    """Supervised anatomical transfer: training section labels are explicitly required."""
    columns = vocabulary(training)

    def train():
        xx, yy, ww = [], [], []
        for s in training:
            x = representation(s, columns, kind)
            meta = s["meta"].dropna(subset=["label"])
            ix = s["ref"].index.get_indexer(meta.index)
            xx.append(x[ix])
            yy.append(meta.label.astype(str).to_numpy())
            ww.append(np.full(len(ix), 1 / len(ix)))
        labels = np.concatenate(yy)
        classes, encoded = np.unique(labels, return_inverse=True)
        weights = np.concatenate(ww) * sum(map(len, yy)) / len(training)
        mass = np.bincount(encoded, weights=weights)
        weights *= weights.sum() / (len(classes) * mass[encoded])
        model = ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features=0.7, n_jobs=2, random_state=seed
        )
        model.fit(np.concatenate(xx), encoded, sample_weight=weights)
        return model, classes

    key = (
        "forest_model_" + "_".join(s["name"] for s in training) + f"_{kind}_leaf{leaf}_seed{seed}"
    )
    model, classes = cached(key, train)
    q = model.predict_proba(representation(target, columns, kind))
    return dict(q=q, classes=classes[model.classes_], training=[s["name"] for s in training])


def evaluate(s, q, spec, seed, stage, classes=None):
    graph = section_graphs(s, features(s, vocabulary([s])))[0]
    result = score(s, q, graph)
    result.update(
        name=s["name"],
        dataset=s["dataset"],
        tissue=s["dataset"].split("_")[0],
        candidate=tag(spec),
        seed=seed,
        stage=stage,
        **spec,
    )
    if classes is not None:
        meta = s["meta"].dropna(subset=["label"])
        pred = classes[q.argmax(1)][s["ref"].index.get_indexer(meta.index)]
        result["macro_f1"] = f1_score(
            meta.label.astype(str), pred, average="macro", zero_division=0
        )
        result["balanced_accuracy"] = balanced_accuracy_score(meta.label.astype(str), pred)
        result["known_class_fraction"] = float(meta.label.isin(classes).mean())
    print(stage, s["name"], tag(spec), seed, round(result["ari"], 4), flush=True)
    return result


def screen():
    protocol = dict(
        selection="Mean ARI over historical development/test sections; separate spinal/cortex; seed17. These labels choose parameters only.",
        followup="Historical fresh1/fresh2 sections are exploratory replication, not new holdouts.",
        confirmation="Previously unscored fifth/sixth eligible sections per spinal species; no model changes after selection is frozen.",
        training="Supervised arm trains on other initial section during screening and both initial development/test sections for all later targets; never target labels.",
        baseline="Original gradient-adjusted ring mixture K8; unsupervised candidates K8 or K12; supervised predicts anatomical classes seen during training.",
        context="Existing gradient-adjusted 15/30/60 counts retained; new ordinary rank contexts8/16/32/64/128/256 restricted to mutual12NN physical components.",
        candidates=candidate_specs(),
        supervised=[
            dict(method="transfer", kind=k, leaf=leaf)
            for k in ["gradient", "broad", "residual"]
            for leaf in [3, 15]
        ],
        caveats=[
            "Cortex truth is composition-derived proxy.",
            "Mouse sections may be the same animal.",
            "Supervised transfer uses manual labels and is not a fair unsupervised-method competition.",
            "MENDER and SCGP inspired alternatives here are custom prototypes, not runs of those packages.",
        ],
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    for dataset in DATASETS:
        for split, other in [("development", "test"), ("test", "development")]:
            s = get_section(dataset, split)
            for spec in candidate_specs():
                q = unsupervised(s, spec, 17)
                records.append(evaluate(s, q, spec, 17, "screen"))
            for spec in protocol["supervised"]:
                fit = cached(
                    "transfer_" + s["name"] + "_" + tag(spec) + "_seed17",
                    lambda: transfer(
                        [get_section(dataset, other)], s, spec["kind"], spec["leaf"], 17
                    ),
                )
                records.append(evaluate(s, fit["q"], spec, 17, "screen", fit["classes"]))
            pd.DataFrame(records).to_csv(OUT / "screen.csv", index=False)
    table = pd.DataFrame(records)
    selection = {}
    for tissue in ["spinal", "cortex"]:
        selection[tissue] = {}
        for family, methods in [
            ("context", ["kmeans"]),
            ("patch", ["patch"]),
            ("transfer", ["transfer"]),
        ]:
            group = table[(table.tissue == tissue) & table.method.isin(methods)]
            winner = group.groupby("candidate").ari.mean().idxmax()
            row = group[group.candidate == winner].iloc[0]
            keys = (
                ["method", "kind"]
                + (["leaf"] if family == "transfer" else ["k"])
                + (["patches"] if family == "patch" else [])
            )
            selection[tissue][family] = {
                k: int(row[k]) if k in ["leaf", "k", "patches"] else row[k] for k in keys
            }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))
    print("FROZEN", selection, flush=True)


def freeze_spinal():
    """Record or verify all spinal selections before opening confirmation scores."""
    frozen = dict(
        main=json.loads((OUT / "selection.json").read_text())["spinal"],
        structured=json.loads((OUT / "transfer_selection.json").read_text())["spinal"],
        molecular=json.loads((OUT / "molecular_selection.json").read_text()),
        geometry=json.loads((OUT / "geometry_selection.json").read_text())["spinal"],
        training=["development", "test"],
        seeds=SEEDS,
    )
    path = OUT / "spinal_frozen_before_confirmation.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    else:
        assert not (OUT / "confirmation.csv").exists()
        path.write_text(json.dumps(frozen, indent=2))
    return frozen


def confirm(stage):
    if stage == "confirmation":
        freeze_spinal()
    selection = json.loads((OUT / "selection.json").read_text())
    datasets = DATASETS if stage == "replication" else DATASETS[:3]
    splits = ["fresh1", "fresh2"] if stage == "replication" else ["confirmation1", "confirmation2"]
    records = []
    for dataset in datasets:
        training = [get_section(dataset, split) for split in ["development", "test"]]
        specs = [dict(method="ring", kind="gradient", k=8)] + list(
            selection[dataset.split("_")[0]].values()
        )
        for split in splits:
            s = get_section(dataset, split)
            for spec in specs:
                for seed in SEEDS:
                    if spec["method"] == "transfer":
                        fit = cached(
                            "transfer_" + s["name"] + "_" + tag(spec) + f"_seed{seed}",
                            lambda: transfer(training, s, spec["kind"], spec["leaf"], seed),
                        )
                        records.append(evaluate(s, fit["q"], spec, seed, stage, fit["classes"]))
                    else:
                        q = unsupervised(s, spec, seed)
                        records.append(evaluate(s, q, spec, seed, stage))
            pd.DataFrame(records).to_csv(OUT / (stage + ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["screen", "prepare", "freeze", "replication", "confirmation"]
    )
    args = parser.parse_args()
    if args.stage == "screen":
        screen()
    elif args.stage == "prepare":
        prepare_confirmation()
    elif args.stage == "freeze":
        freeze_spinal()
    else:
        confirm(args.stage)
