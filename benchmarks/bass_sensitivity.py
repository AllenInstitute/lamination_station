"""Longer-chain and explicitly annotation-aware spatial-strength diagnostics."""

import torch  # noqa: F401

import argparse

import pandas as pd

from run_bass_fixed_types import OUT, get_section, experiment, anatomical_score


def run(dataset, kind):
    if kind == "select":
        frames = [
            pd.read_csv(OUT / f"strength_spinal_{species}.csv")
            for species in ["human", "macaque", "mouse"]
        ]
        summary = pd.concat(frames).groupby("beta")[["ari", "thin_best_iou"]].mean()
        summary.to_csv(OUT / "strength_selection.csv")
        print(summary.to_string())
        return
    chosen = pd.read_csv(OUT / "selected.csv")
    config = chosen[(chosen.tissue == dataset.split("_")[0]) & (chosen.k == 8)].iloc[0]
    rows = []
    if kind == "strength":
        settings = [
            (split, beta, 1000, 200)
            for split in ["development", "test"]
            for beta in [2.0, 4.0, 6.0, 8.0, 12.0]
        ]
    elif kind == "annotated_tuning":
        strengths = pd.read_csv(OUT / "strength_selection.csv")
        beta = float(strengths.loc[strengths.ari.idxmax(), "beta"])
        settings = [
            (split, beta, 1000, 200)
            for split in ["fresh1", "fresh2", "confirmation1", "confirmation2"]
        ]
    else:
        split = "confirmation1" if dataset.startswith("spinal") else "fresh1"
        settings = [(split, config.beta, 5000, 500)]
    for split, beta, burnin, draws in settings:
        s = get_section(dataset, split)
        result, d, key, elapsed = experiment(
            s, 8, beta, config.shrinkage, 17, burnin=burnin, draws=draws
        )
        scored, _ = anatomical_score(s, result, d)
        rows.append(
            dict(name=s["name"], dataset=dataset, beta=beta, kind=kind, cache_key=key, **scored)
        )
        pd.DataFrame(rows).to_csv(OUT / f"{kind}_{dataset}.csv", index=False)
        print(kind, s["name"], beta, scored["ari"], elapsed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("kind", choices=["long", "strength", "select", "annotated_tuning"])
    args = parser.parse_args()
    run(args.dataset, args.kind)
