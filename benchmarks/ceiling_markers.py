"""Audit spatially cohesive and region-enriched cell types without inventing new labels."""

import torch  # noqa: F401

import argparse

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.metrics import roc_auc_score, average_precision_score

from ceiling_diagnostics import OUT, SPECIES, SPLITS, cached, get_data, get_section
from context_experiments import vocabulary
from incremental_domains import features, section_graphs


def moran(codes, rows, cols, degree, n_types):
    n = len(codes)
    total = len(rows)
    mass = np.bincount(codes, minlength=n_types)
    p = mass / n
    same = codes[rows] == codes[cols]
    within = np.bincount(codes[rows[same]], minlength=n_types)
    endpoints = 2 * np.bincount(codes, weights=degree, minlength=n_types)
    numerator = within - p * endpoints + p * p * total
    return np.divide(
        numerator, total * p * (1 - p), out=np.zeros(n_types), where=(p > 0) & (p < 1)
    )


def coherence():
    records = []
    boundary = []
    for species in SPECIES:
        for split in SPLITS:
            s = get_section("spinal_" + species, split)
            d = get_data(s)

            def compute():
                graph = section_graphs(s, features(s, vocabulary([s])))[0].tocoo()
                rr, cc = graph.row, graph.col
                degree = np.asarray(graph.tocsr().sum(1)).ravel()
                types, codes = np.unique(d["types"], return_inverse=True)
                observed = moran(codes, rr, cc, degree, len(types))
                rng = np.random.default_rng(731)
                bins = pd.qcut(d["cov"][:, 1], 5, labels=False, duplicates="drop")
                groups = [np.flatnonzero(bins == b) for b in np.unique(bins)]
                null = []
                conditional = []
                for _ in range(199):
                    null.append(moran(rng.permutation(codes), rr, cc, degree, len(types)))
                    perm = codes.copy()
                    for ix in groups:
                        perm[ix] = rng.permutation(codes[ix])
                    conditional.append(moran(perm, rr, cc, degree, len(types)))
                null = np.asarray(null)
                conditional = np.asarray(conditional)
                p = (1 + (null >= observed).sum(0)) / 200
                pc = (1 + (conditional >= observed).sum(0)) / 200
                mass = np.bincount(codes, minlength=len(types))
                eligible = mass >= 10
                q = np.ones(len(types))
                qc = q.copy()
                q[eligible] = false_discovery_control(p[eligible])
                qc[eligible] = false_discovery_control(pc[eligible])
                y = d["y"]
                valid = y.notna().to_numpy()
                table = pd.crosstab(
                    pd.Series(d["types"][valid], name="type"),
                    pd.Series(y[valid].astype(str).to_numpy(), name="lamina"),
                )
                top = table.max(1) / table.sum(1)
                return pd.DataFrame(
                    dict(
                        type=types,
                        n=mass,
                        moran=observed,
                        p=p,
                        q=q,
                        conditional_p=pc,
                        conditional_q=qc,
                        null_mean=null.mean(0),
                        density_null_mean=conditional.mean(0),
                        dominant_lamina_fraction=top.reindex(types).to_numpy(),
                        dominant_lamina=table.idxmax(1).reindex(types).to_numpy(),
                    )
                )

            table = cached("coherence_" + s["name"], compute).copy()
            table["name"] = s["name"]
            table["species"] = species
            records.append(table)
            graph = section_graphs(s, features(s, vocabulary([s])))[0].tocoo()
            rr, cc = graph.row, graph.col
            y = pd.Categorical(d["y"]).codes
            use = (rr < cc) & (y[rr] >= 0) & (y[cc] >= 0)
            rr, cc = rr[use], cc[use]
            truth = y[rr] != y[cc]
            p = s["counts"][30].to_numpy(dtype=float) / 30
            signals = {
                "composition_hellinger": 0.5
                * np.sum((np.sqrt(p[rr]) - np.sqrt(p[cc])) ** 2, axis=1),
                "density_difference": np.abs(d["cov"][rr, 1] - d["cov"][cc, 1]),
            }
            for key, signal in signals.items():
                boundary.append(
                    dict(
                        name=s["name"],
                        species=species,
                        signal=key,
                        auc=roc_auc_score(truth, signal),
                        ap=average_precision_score(truth, signal),
                        prevalence=truth.mean(),
                    )
                )
            print(
                "COHERENCE", s["name"], int(((table.n >= 10) & (table.q < 0.05)).sum()), flush=True
            )
            pd.concat(records).to_csv(OUT / "type_coherence.csv", index=False)
            pd.DataFrame(boundary).to_csv(OUT / "boundary_signal.csv", index=False)


def markers():
    records = []
    selected = []
    profiles = []
    for species in SPECIES:
        training = [get_section("spinal_" + species, split) for split in SPLITS[:4]]
        tables = []
        for s in training:
            d = get_data(s)
            valid = d["y"].notna().to_numpy()
            table = pd.crosstab(
                pd.Series(d["y"][valid].astype(str).to_numpy(), name="lamina"),
                pd.Series(d["types"][valid], name="type"),
            )
            tables.append(table)
        laminae = sorted(set().union(*(set(t.index) for t in tables)))
        types = sorted(set().union(*(set(t.columns) for t in tables)))
        arrays = np.stack(
            [t.reindex(index=laminae, columns=types, fill_value=0).to_numpy() for t in tables]
        )
        pooled = arrays.sum(0)
        for i, lamina in enumerate(laminae):
            within = pooled[i]
            outside = pooled.sum(0) - within
            pin = (within + 0.5) / (within.sum() + 0.5 * len(types))
            pout = (outside + 0.5) / (outside.sum() + 0.5 * len(types))
            fold = pin / pout
            support_sections = (arrays[:, i, :] > 0).sum(0)
            # At least20 marker cells in this reference lamina and support in3 sections.
            eligible = (within >= 20) & (support_sections >= 3) & (fold >= 2)
            ranking = np.where(eligible, np.log(fold) * within / (within + 20), -np.inf)
            chosen = np.argsort(ranking)[::-1][:3]
            chosen = chosen[np.isfinite(ranking[chosen])]
            if not len(chosen):
                selected.append(
                    dict(species=species, lamina=lamina, type="__no_supported_marker", rank=0)
                )
            for rank, j in enumerate(chosen):
                selected.append(
                    dict(
                        species=species,
                        lamina=lamina,
                        type=types[j],
                        rank=rank + 1,
                        train_region_count=int(within[j]),
                        train_in_region_fraction=float(within[j] / within.sum()),
                        train_fold=float(fold[j]),
                        train_precision=float(within[j] / pooled[:, j].sum()),
                        binomial_zero30=float((1 - within[j] / within.sum()) ** 30),
                    )
                )
            for s in [get_section("spinal_" + species, split) for split in SPLITS[4:]]:
                d = get_data(s)
                y = d["y"].astype("string").fillna("__unlabeled").to_numpy()
                valid = y != "__unlabeled"
                inside = y == lamina
                if not inside.any():
                    continue
                for rank, j in enumerate(chosen):
                    col = types[j] + "_count"
                    count = (
                        s["counts"][30][col].to_numpy()
                        if col in s["counts"][30]
                        else np.zeros(len(y))
                    )
                    positive = count > 0
                    interior = inside & (d["boundary"] >= d["distance"][:, 29])
                    records.append(
                        dict(
                            name=s["name"],
                            species=species,
                            lamina=lamina,
                            type=types[j],
                            rank=rank + 1,
                            n_region=int(inside.sum()),
                            sensitivity=float(positive[inside].mean()),
                            specificity=float((~positive)[valid & (~inside)].mean()),
                            precision=float(inside[positive & valid].mean())
                            if (positive & valid).any()
                            else 0,
                            interior_n=int(interior.sum()),
                            interior_zero_fraction=float((~positive)[interior].mean())
                            if interior.any()
                            else np.nan,
                            region_zero_fraction=float((~positive)[inside].mean()),
                            auc=roc_auc_score(inside[valid], count[valid]),
                            ap=average_precision_score(inside[valid], count[valid]),
                            prevalence=float(inside[valid].mean()),
                        )
                    )
        # Pairwise reference composition separability and section-to-section shifts.
        fractions = (arrays + 0.5) / (arrays.sum(2, keepdims=True) + 0.5 * len(types))
        for a, la in enumerate(laminae):
            same = [
                np.sum((np.sqrt(fractions[i, a]) - np.sqrt(fractions[j, a])) ** 2) / 2
                for i in range(4)
                for j in range(i + 1, 4)
            ]
            for b, lb in enumerate(laminae):
                if b <= a:
                    continue
                cross = [
                    np.sum((np.sqrt(fractions[i, a]) - np.sqrt(fractions[i, b])) ** 2) / 2
                    for i in range(4)
                ]
                profiles.append(
                    dict(
                        species=species,
                        a=la,
                        b=lb,
                        between_lamina_hellinger=float(np.mean(cross)),
                        same_lamina_between_section_hellinger=float(np.mean(same)),
                    )
                )
    pd.DataFrame(selected).to_csv(OUT / "selected_markers.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "marker_detection.csv", index=False)
    pd.DataFrame(profiles).to_csv(OUT / "regional_profile_distances.csv", index=False)
    print("MARKERS", len(selected), len(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["coherence", "markers"])
    args = parser.parse_args()
    {"coherence": coherence, "markers": markers}[args.stage]()
