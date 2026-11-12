"""Alternative supervised objectives and spatial refinement of anatomical predictions."""

import torch  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from context_experiments import (
    DATASETS,
    OUT,
    SEEDS,
    cached,
    evaluate,
    get_section,
    representation,
    tag,
    transfer,
    vocabulary,
)
from incremental_domains import features, section_graphs
from incremental_models import refine


def specs():
    return [
        dict(method="svm", kind=k, c=c) for k in ["gradient", "broad"] for c in [1.0, 10.0]
    ] + [dict(method="forest_spatial", kind="broad", strength=b) for b in [4.0, 16.0]]


def fit(training, target, spec, seed):
    def compute():
        if spec["method"] == "forest_spatial":
            base = dict(method="transfer", kind="broad", leaf=3)
            model = cached(
                "transfer_" + target["name"] + "_" + tag(base) + f"_seed{seed}",
                lambda: transfer(training, target, "broad", 3, seed),
            )
            rings = features(target, vocabulary(training))
            _, uniform, _ = section_graphs(target, rings)
            refined = refine(
                np.log(np.maximum(model["q"], 1e-8)),
                uniform,
                uniform,
                strength=spec["strength"],
                max_iter=2000,
            )
            assert refined["converged"]
            return dict(q=refined["q"], classes=model["classes"], training=model["training"])
        columns = vocabulary(training)
        xx, yy, ww = [], [], []
        for s in training:
            x = representation(s, columns, spec["kind"])
            meta = s["meta"].dropna(subset=["label"])
            ix = s["ref"].index.get_indexer(meta.index)
            xx.append(x[ix])
            yy.append(meta.label.astype(str).to_numpy())
            ww.append(np.ones(len(ix)) / len(ix))
        classes, encoded = np.unique(np.concatenate(yy), return_inverse=True)
        weights = np.concatenate(ww) * len(encoded) / len(training)
        mass = np.bincount(encoded, weights=weights)
        weights *= weights.sum() / (len(classes) * mass[encoded])
        model = SVC(
            C=spec["c"],
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
            cache_size=512,
        )
        model.fit(np.concatenate(xx), encoded, sample_weight=weights)
        q = model.predict_proba(representation(target, columns, spec["kind"]))
        return dict(q=q, classes=classes[model.classes_], training=[s["name"] for s in training])

    return cached("structured_" + target["name"] + "_" + tag(spec) + f"_seed{seed}", compute)


def run(stage):
    records = []
    if stage == "screen":
        candidates = specs()
        splits = ["development", "test"]
        seeds = [17]
        (OUT / "transfer_protocol.json").write_text(
            json.dumps(
                dict(
                    candidates=candidates,
                    selection="Separate cortex/spinal mean ARI on historical development/test sections; same section-level splits as original transfer arm.",
                    description="RBF support-vector classifier C1/10 with training-only probability fitting; or broad-context ExtraTrees leaf3 followed by fixed-unary Potts mean-field refinement on normalized physical graph with strength4/16.",
                    warning="Explicitly supervised; anatomy labels on training sections are required. Outputs are not calibrated on target sections.",
                ),
                indent=2,
            )
        )
    else:
        selection = json.loads((OUT / "transfer_selection.json").read_text())
        splits = (
            ["fresh1", "fresh2"] if stage == "replication" else ["confirmation1", "confirmation2"]
        )
        seeds = SEEDS
    for dataset in DATASETS if stage != "confirmation" else DATASETS[:3]:
        if stage != "screen":
            candidates = [selection[dataset.split("_")[0]]]
        for split in splits:
            training = (
                [get_section(dataset, {"test": "development", "development": "test"}[split])]
                if stage == "screen"
                else [get_section(dataset, x) for x in ["development", "test"]]
            )
            s = get_section(dataset, split)
            for spec in candidates:
                for seed in seeds:
                    model = fit(training, s, spec, seed)
                    records.append(evaluate(s, model["q"], spec, seed, stage, model["classes"]))
            pd.DataFrame(records).to_csv(OUT / f"transfer_{stage}.csv", index=False)
    if stage == "screen":
        frame = pd.DataFrame(records)
        selection = {}
        for tissue in ["spinal", "cortex"]:
            winner = frame[frame.tissue == tissue].groupby("candidate").ari.mean().idxmax()
            selection[tissue] = next(s for s in candidates if tag(s) == winner)
        (OUT / "transfer_selection.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "replication", "confirmation"])
    args = parser.parse_args()
    run(args.stage)
