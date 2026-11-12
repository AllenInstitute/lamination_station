"""Validate the supported workflow against the winning benchmark on real sections."""

import torch  # noqa: F401

import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lamination_station import regions
from lamination_station.ring_model import RingMixture
from context_experiments import get_section
from incremental_domains import features, vocabulary
from incremental_models import fit_mixture


OUT = Path("/results/lamination_station_regions")
OUT.mkdir(parents=True, exist_ok=True)
datasets = ["spinal_human", "spinal_macaque", "spinal_mouse", "cortex_human", "cortex_merfish"]
records = []
checks = []
fig, axes = plt.subplots(5, 3, figsize=(13, 18), layout="constrained")
for row, dataset in enumerate(datasets):
    split = "confirmation1" if dataset.startswith("spinal") else "fresh1"
    s = get_section(dataset, split)
    prepared = regions.prepare_gradient_rings(
        cells=s["ref"], x_col="x", y_col="y", celltype_col="type", sample_col="section"
    )
    columns = vocabulary([s])
    old = features(s, columns)
    for actual, expected in zip(prepared.rings, old, strict=True):
        np.testing.assert_array_equal(actual.reindex(columns=columns).to_numpy(), expected)
        assert (actual[actual.columns.difference(columns)] == 0).all().all()
    scan = regions.fit_resolutions(features=prepared, n_regions=(8, 12, 16), seeds=(17, 29, 43))
    scan.summary.to_csv(OUT / f"{dataset}_resolution_summary.csv")
    original = fit_mixture(old, n_components=8, seed=17)
    result = scan.fits[8, 17]
    label_agreement = adjusted_rand_score(original["q"].argmax(1), result.assignments.region)
    assert label_agreement > 0.999999
    # Compare objective/probabilities where vocabulary dimensions match exactly.
    same_vocabulary = prepared.rings[0].columns.equals(pd.Index(columns))
    max_error = None
    if same_vocabulary:
        max_error = float(np.abs(original["q"] - result.memberships.to_numpy()).max())
        np.testing.assert_allclose(
            original["q"], result.memberships.to_numpy(), atol=1e-7, rtol=1e-7
        )
        np.testing.assert_allclose(original["history"], result.model.objective_history, atol=1e-7)
    checks.append(
        dict(
            dataset=dataset,
            n_cells=len(s["ref"]),
            counts_match=True,
            baseline_partition_ari=label_agreement,
            same_vocabulary=same_vocabulary,
            max_membership_error=max_error,
        )
    )
    for column, k in enumerate([8, 12, 16]):
        chosen = scan.best(n_regions=k)
        folder = OUT / f"{dataset}_k{k}"
        chosen.save(folder)
        reloaded = RingMixture.load(folder / "model.json")
        pd.testing.assert_frame_equal(reloaded.predict_proba(prepared.rings), chosen.memberships)
        for seed in [17, 29, 43]:
            fit = scan.fits[k, seed]
            meta = s["meta"].dropna(subset=["label"])
            ids = s["ref"].index.get_indexer(meta.index)
            records.append(
                dict(
                    dataset=dataset,
                    section=s["name"],
                    n_regions=k,
                    seed=seed,
                    reference_ari=adjusted_rand_score(
                        meta.label, fit.assignments.region.iloc[ids]
                    ),
                    occupied=int(fit.assignments.region.nunique()),
                    converged=fit.model.converged,
                    min_region_cells=int(fit.region_summary.n_cells.min()),
                    mean_membership=float(fit.assignments.membership_max.mean()),
                    seed_ari=float(scan.summary.loc[k, "mean_seed_ari"]),
                )
            )
        ax = axes[row, column]
        ax.scatter(
            s["ref"].x,
            s["ref"].y,
            c=chosen.assignments.region,
            s=2,
            cmap="tab20",
            vmin=0,
            vmax=19,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{dataset}: requested {k}, occupied {chosen.assignments.region.nunique()}",
            fontsize=10,
        )
    pd.DataFrame(records).to_csv(OUT / "resolution_scores.csv", index=False)
    (OUT / "parity_checks.json").write_text(json.dumps(checks, indent=2))
    print("VALIDATED", dataset, checks[-1], flush=True)
fig.savefig(OUT / "resolution_maps.png", dpi=150)
fig.savefig(OUT / "resolution_maps.pdf")
plt.close(fig)
table = (
    pd.DataFrame(records)
    .groupby(["dataset", "n_regions"])[
        ["reference_ari", "occupied", "converged", "min_region_cells", "seed_ari"]
    ]
    .mean()
)
links = "".join(
    f'<li><a href="{d}_k{k}/assignments.csv">{d}, K={k}: cell calls</a>; <a href="{d}_k{k}/manifest.json">settings</a></li>'
    for d in datasets
    for k in [8, 12, 16]
)
report = f"""<!doctype html><html><head><meta charset="utf-8"><title>Supported gradient-ring workflow validation</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 24px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>Selectable-resolution gradient-ring workflow</h1><p>The packaged workflow reproduces the benchmark's K=8 partitions on all five representative sections. Neighbor counts are checked directly, and saved JSON models reproduce memberships after loading. K=12 and K=16 are demonstrated for overclustering; no region count was selected using reference annotations.</p>
<p>These are previously inspected sections. Spinal references are polygon annotations; cortical references are composition-derived proxies. Scores are descriptive and do not establish new biological accuracy. The seed agreement compares partitions at fixed K. Higher K does not impose a nested split hierarchy, and membership strength is not calibrated confidence. Map colors identify components within each fit and are not aligned across resolutions.</p>
{table.to_html(float_format=lambda v: f"{v:.3f}")}<img src="resolution_maps.png">
<h2>Example analysis outputs</h2><p>Each directory also contains per-ring profiles, memberships, region sizes, gradients, initialization diagnostics, an objective trace, and model.json. These are representative workflow demonstrations, not final adjudicated anatomical calls.</p><ul>{links}</ul>
<p>The separate <a href="implementation_parity.json">numerical implementation check</a> compares the package and benchmark using identical vocabulary order: memberships agree within 4e-14 on all five sections. The workflow checks below align count columns by name; default type ordering can differ without changing the partition.</p><h2>Parity checks</h2><pre>{html.escape(json.dumps(checks, indent=2))}</pre></body></html>"""
(OUT / "report.html").write_text(report)
print(table.to_string())
