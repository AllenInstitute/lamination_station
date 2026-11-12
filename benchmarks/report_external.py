"""Summarize separate-section HRCHY and BANKSY comparisons."""

import torch  # noqa: F401
import hashlib
import itertools
import json
import pickle
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

out = Path("/results/lamination_station_external")
root = Path("/scratch/lamination_station_external")
previous = Path("/scratch/lamination_station_investigation")
shards = [out / f"hrchy_validation_spinal_{s}.csv" for s in ["human", "macaque", "mouse"]]
hrchy = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
assert len(hrchy) == 18 and not hrchy.duplicated(["name", "seed"]).any()
hrchy.to_csv(out / "hrchy_validation.csv", index=False)
banksy = pd.read_csv(out / "banksy_validation.csv")
assert len(banksy) == 36
runs = [hrchy, banksy]
baseline_records = []
for method in ["kmeans", "multinomial", "ring_multinomial"]:
    for p in previous.glob(f"spinal_*_fresh[12]_{method}_K8_seed*.pkl"):
        rec = pickle.load(p.open("rb"))["record"].copy()
        rec["origin"] = "archived_previous_investigation"
        baseline_records.append(rec)
runs.append(pd.DataFrame(baseline_records))
sq = pd.read_csv("/results/lamination_station_squidpy/validation_spinal.csv")
runs.append(sq[sq.method == "squidpy"])
all_runs = pd.concat(runs, ignore_index=True)
all_runs.to_csv(out / "all_runs.csv", index=False)
section = all_runs.groupby(["name", "method"]).ari.mean().unstack()
section.to_csv(out / "section_means.csv")
summary = (
    all_runs.groupby("method")
    .agg(ari=("ari", "mean"), ami=("ami", "mean"), occupied=("occupied", "mean"))
    .sort_values("ari", ascending=False)
)
summary["wins_vs_kmeans"] = [(section[m] > section.kmeans).sum() for m in summary.index]
summary.to_csv(out / "summary.csv")

methods = ["kmeans", "ring_multinomial", "hrchy", "banksy_kmeans", "banksy_leiden", "squidpy"]
titles = dict(
    kmeans="Neighborhood k-means",
    ring_multinomial="Ring multinomial",
    hrchy="HRCHY-CytoCommunity",
    banksy_kmeans="BANKSY + k-means",
    banksy_leiden="BANKSY + native Leiden",
    squidpy="Squidpy",
)
fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
for i, m in enumerate(methods):
    ax.barh(np.arange(len(section)) + (i - 2.5) * 0.14, section[m], height=0.135, label=titles[m])
ax.set(
    yticks=np.arange(len(section)),
    yticklabels=[n.replace("_", " ") for n in section.index],
    xlabel="Adjusted Rand index (mean of three seeds)",
    xlim=(0, 1),
)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8)
fig.savefig(out / "comparison.png", dpi=180)
fig.savefig(out / "comparison.pdf")
plt.close(fig)

selections = {m: json.loads((out / f"{m}_selected.json").read_text()) for m in ["hrchy", "banksy"]}
stability = []
manifest = []
diagnostics = []
fig, axes = plt.subplots(6, 5, figsize=(15, 19), layout="constrained")
for row, name in enumerate(section.index):
    path = previous / f"{name}.pkl"
    ref, _, meta = pickle.load(path.open("rb"))
    ix = ref.index.get_indexer(meta.index)
    assert (ix >= 0).all()
    predictions = {}
    for seed in [17, 29, 43]:
        c = selections["hrchy"]["hrchy"]
        stem = f"{name}_hrchy_{c['near']}_c{c['coarse']}_seed{seed}"
        predictions[("hrchy", seed)] = pickle.load((root / f"{stem}.pkl").open("rb"))["labels"][ix]
        losses = pd.read_csv(root / stem / "Epoch_TrainLoss.csv")
        assert np.isfinite(losses.TrainLoss).all() and losses.Epoch.max() == 1500
        diagnostics.append(
            dict(
                name=name,
                seed=seed,
                first_loss=losses.TrainLoss.iloc[0],
                last_loss=losses.TrainLoss.iloc[-1],
                epochs=1500,
            )
        )
        losses.to_csv(out / f"{stem}_loss.csv", index=False)
        for m in ["banksy_kmeans", "banksy_leiden"]:
            c = selections["banksy"][m]
            stored = pickle.load(
                (root / f"{name}_banksy_native_n{c['near']}_l{c['lam']}_seed{seed}.pkl").open("rb")
            )
            predictions[(m, seed)] = next(
                r["labels"][ix] for r in stored if r["method"] == m and r["settings"] == c
            )
        for m in ["kmeans", "ring_multinomial"]:
            predictions[(m, seed)] = pickle.load(
                (previous / f"{name}_{m}_K8_seed{seed}.pkl").open("rb")
            )["q"][ix].argmax(1)
    export = meta.copy()
    export[["x", "y"]] = ref.loc[meta.index, ["x", "y"]]
    for (m, seed), labels in predictions.items():
        export[f"{m}_{seed}"] = labels
        mask = meta.label.notna().to_numpy()
        recomputed = adjusted_rand_score(meta.label[mask], np.asarray(labels)[mask])
        saved = all_runs[
            (all_runs.name == name) & (all_runs.method == m) & (all_runs.seed == seed)
        ].ari.iloc[0]
        assert abs(recomputed - saved) < 1e-10, (name, m, seed)
    export.to_parquet(out / f"{name}_assignments.parquet")
    for m in ["hrchy", "banksy_kmeans", "banksy_leiden", "kmeans", "ring_multinomial"]:
        stability.append(
            dict(
                name=name,
                method=m,
                seed_ari=np.mean(
                    [
                        adjusted_rand_score(predictions[(m, a)], predictions[(m, b)])
                        for a, b in itertools.combinations([17, 29, 43], 2)
                    ]
                ),
            )
        )
    for col, m in enumerate(["reference", "kmeans", "hrchy", "banksy_kmeans", "banksy_leiden"]):
        if m == "reference":
            labels = pd.Categorical(meta.label).codes
            title = name.replace("_", " ") + "\nPolygon reference"
        else:
            labels = pd.Categorical(predictions[(m, 17)]).codes
            title = (
                titles[m]
                + f"\nARI {all_runs[(all_runs.name == name) & (all_runs.method == m) & (all_runs.seed == 17)].ari.iloc[0]:.3f}"
            )
        ax = axes[row, col]
        ax.scatter(export.x, export.y, c=labels, cmap="tab20", s=2, linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9)
    manifest.append(
        dict(
            name=name,
            section=ref.section.astype(str).unique().tolist(),
            reference_cells=len(ref),
            focal_cells=len(meta),
            labelled_cells=int(meta.label.notna().sum()),
            input=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
fig.savefig(out / "maps.png", dpi=150)
fig.savefig(out / "maps.pdf")
plt.close(fig)
pd.DataFrame(stability).to_csv(out / "stability.csv", index=False)
pd.DataFrame(diagnostics).to_csv(out / "training_diagnostics.csv", index=False)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
report = f"""<!doctype html><html><head><meta charset="utf-8"><title>HRCHY and BANKSY tissue benchmark</title><style>body{{max-width:1150px;margin:40px auto;font:16px/1.5 system-ui;padding:0 20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:right}}img{{width:100%}}</style></head><body>
<h1>HRCHY-CytoCommunity and BANKSY: spinal lamina benchmark</h1>
<p>Six separate validation sections: human, macaque and mouse spinal cord, two sections each. Direct polygon annotations only; three seeds per method. Mean ARI and section wins:</p>{summary.to_html(float_format=lambda v: f"{v:.3f}")}
<img src="comparison.png" alt="Per-section annotation agreement"><h2>Protocol and scope</h2>
<p>All methods receive cell types and physical coordinates, with the same cached reference-cell universe and focal evaluation cells. Reference sets retain the prior 10,000-cell cap. Each section is fitted independently without anatomical labels. These are transductive discovery fits, not supervised prediction on new donors. Some sections can share donors. Seeds are computational repetitions, not biological replicates.</p>
<p>Selection uses only the original three spinal development sections, seed 17; the frozen settings are {selections}. No validation label selected parameters. HRCHY development compared 30 neighbors with floor(sqrt(number of cells)), using eight fine structures and two coarse structures, default 128 hidden units, consistency repeats S=10, and all 1,500 epochs. BANKSY development compared spatial neighbors 15/30/60 and lambda .8/.95/1, with m=1 gradients and 20 PCs (or the smaller feature dimension). BANKSY + k-means requests eight clusters and 20 starts. Native BANKSY Leiden uses its distance-weighted shared-neighbor graph with 50 neighbors and a resolution sweep .05/.1/.2/.4/.8/1.6/3.2. Its number of clusters is unconstrained; occupied counts are reported above.</p>
<p>BANKSY uses one-hot cell-type indicators as features. This is a cell-type-only adaptation; these results do not evaluate its usual gene-expression input. Official BANKSY feature construction and native Leiden partitioner were used. HRCHY uses the released HRCHYCytoCommunityGrand implementation unchanged, with a standard in-memory PyG Data object containing a symmetrized kNN graph and one-hot labels. Ground-truth flags are disabled. Its released trainer recreates Adam each epoch; this benchmark retains that behavior rather than silently altering the method.</p>
<p>Baselines are cached results from the prior investigation on exactly the same reference/focal cells, K=8 and seeds 17/29/43. Neighborhood k-means uses Hellinger composition at 60 neighbors. The ring mixture models disjoint neighbor ranks 1–15, 16–30 and 31–60. Squidpy results come from the preceding benchmark. This is a finite parameter screen, not exhaustive optimization of any method. Full-density tissues, other taxonomies, cortex, expression input, and continuous laminar depth are outside this comparison.</p>
<h2>Spatial maps</h2><img src="maps.png" alt="Reference and predicted partitions"><p>Seed 17, selected in advance. Cluster colors are arbitrary and not matched between methods or to reference labels.</p>
<h2>Reproducibility</h2><p>Run external_comparison.py banksy development, then banksy validation. For HRCHY, run development separately with a third argument spinal_human, spinal_macaque or spinal_mouse; run <code>external_comparison.py hrchy select</code> to combine the development CSVs and select the configuration with highest three-section mean ARI. Then run validation with the same three subsets. Run report_external.py after all validation finishes. <a href="protocol.json">Protocol</a>, <a href="summary.csv">summary</a>, <a href="all_runs.csv">all runs</a>, <a href="section_means.csv">section means</a>, <a href="stability.csv">seed stability</a>, <a href="training_diagnostics.csv">training diagnostics</a>, <a href="manifest.json">input hashes and sections</a>. The environment/postInstall pins package versions; input caches remain in /scratch. Use TMPDIR=/scratch/runtime/tmp and MPLCONFIGDIR=/scratch/runtime/mpl. Core Lamination Station code was not modified.</p>
<p>Sources: <a href="https://github.com/huBioinfo/HRCHY-CytoCommunity">HRCHY implementation</a>, <a href="https://github.com/prabhakarlab/Banksy_py">BANKSY implementation</a>.</p></body></html>"""
(out / "report.html").write_text(report)
for name in ["external_comparison.py", "report_external.py"]:
    shutil.copy2(Path(__file__).parent / name, out / name)
print(summary.to_string())
