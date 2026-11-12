"""Secondary test: share fixed-type region profiles across disconnected sections."""

import torch  # noqa: F401

import argparse

import numpy as np
import pandas as pd
from scipy import sparse

from bass_fixed_types import fit, initial_labels
from run_bass_fixed_types import (
    OUT,
    SEEDS,
    cached,
    inputs,
    get_section,
    anatomical_score,
)


def run(dataset, masked=False):
    splits = ["development", "test", "fresh1", "fresh2"]
    if dataset.startswith("spinal"):
        splits += ["confirmation1", "confirmation2"]
    sections = [get_section(dataset, s) for s in splits]
    data = [inputs(s, masked, fold=1) for s in sections]
    names = sorted(set().union(*(set(d["names"]) for d in data)))
    lookup = {name: i for i, name in enumerate(names)}
    observed = []
    codes = []
    for d in data:
        mapped = np.array([lookup[d["names"][c]] for c in d["codes"]])
        codes.append(mapped)
        mapped = mapped.copy()
        mapped[d["observed"] < 0] = -1
        observed.append(mapped)
    observed = np.concatenate(observed)
    codes = np.concatenate(codes)
    graph = sparse.block_diag([d["weighted"] for d in data], format="csr")
    xy = np.concatenate([d["xy"] for d in data])
    groups = np.concatenate([np.full(len(d["xy"]), i) for i, d in enumerate(data)])
    offsets = np.r_[0, np.cumsum([len(d["xy"]) for d in data])]
    chosen = pd.read_csv(OUT / "selected.csv")
    chosen = chosen[chosen.tissue == dataset.split("_")[0]]
    records, regions = [], []
    for _, config in chosen.iterrows():
        k = int(config.k)
        for seed in [17] if masked else SEEDS:
            init_key = f"joint_init_{dataset}_K{k}_seed{seed}_mask{masked}"
            init = cached(
                init_key, lambda: initial_labels(xy, observed, len(names), k, seed, groups)
            )
            key = (
                f"joint_{dataset}_K{k}_seed{seed}_mask{masked}_b{config.beta}_a{config.shrinkage}"
            )
            result = cached(
                key,
                lambda: fit(
                    observed,
                    graph,
                    init,
                    n_types=len(names),
                    beta=config.beta,
                    shrinkage=config.shrinkage,
                    seed=seed,
                    burnin=1000,
                    draws=200,
                    thin=5,
                ),
            )
            for i, (s, d) in enumerate(zip(sections, data)):
                # The first two sections contribute unsupervised information but not scores.
                if i < 2:
                    continue
                if masked:
                    hidden = result["hidden"]
                    use = (hidden >= offsets[i]) & (hidden < offsets[i + 1])
                    probability = result["prediction"][use, codes[hidden[use]]]
                    score = dict(
                        nll=float(-np.log(np.maximum(probability, 1e-300)).mean()),
                        n_masked=int(use.sum()),
                    )
                else:
                    sub = dict(q=result["q"][offsets[i] : offsets[i + 1]])
                    score, regs = anatomical_score(s, sub, d)
                    regions.extend(
                        [
                            dict(
                                name=s["name"],
                                dataset=dataset,
                                k=k,
                                seed=seed,
                                method="bass_shared",
                                **r,
                            )
                            for r in regs
                        ]
                    )
                records.append(
                    dict(
                        name=s["name"],
                        dataset=dataset,
                        k=k,
                        seed=seed,
                        method="bass_shared",
                        beta=config.beta,
                        shrinkage=config.shrinkage,
                        cache_key=key,
                        **score,
                    )
                )
                print("SHARED", s["name"], k, seed, score, flush=True)
            suffix = "_masked" if masked else ""
            pd.DataFrame(records).to_csv(OUT / f"shared_{dataset}{suffix}.csv", index=False)
            if not masked:
                pd.DataFrame(regions).to_csv(OUT / f"shared_regions_{dataset}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--masked", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.masked)
