"""Render the paired gradient audit and its mathematical interpretation."""

import torch  # noqa: F401

import html
import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score

out = Path("/results/lamination_station_gradients")
sections = pd.read_csv(out / "section_means.csv")
summary = pd.read_csv(out / "summary.csv")
checks = pd.read_csv(out / "checks.csv")
analytic = json.loads((out / "analytic_checks.json").read_text())
shutil.copy2(__file__, out / Path(__file__).name)
shutil.copy2(Path(__file__).with_name("gradient_audit.py"), out / "gradient_audit.py")

# Preserve and independently score the exact assignments behind every result.
predictions = out / "predictions"
predictions.mkdir(exist_ok=True)
runs = pd.read_csv(out / "runs.csv")
for name, group in runs.groupby("name"):
    source = Path("/scratch/lamination_station_investigation") / f"{name}.pkl"
    ref, _, meta = pickle.load(source.open("rb"))
    exported = ref.copy()
    exported["evaluation_label"] = meta.label.reindex(ref.index)
    valid = exported.evaluation_label.notna()
    for row in group.itertuples():
        cached = (
            Path("/scratch/lamination_station_gradients")
            / f"{name}_{row.method}_{row.adjustment}_{row.seed}.pkl"
        )
        result = pickle.load(cached.open("rb"))
        assert result["index"].equals(ref.index)
        labels = result["labels"]
        score = adjusted_rand_score(exported.loc[valid, "evaluation_label"], labels[valid])
        assert abs(score - row.ari) < 1e-12
        exported[f"{row.method}_{row.adjustment}_seed{row.seed}"] = labels
    exported.to_csv(predictions / f"{name}.csv.gz")

fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True, layout="constrained")
for ax, method, title in zip(
    axes,
    ["hellinger_kmeans", "ring_multinomial"],
    ["Hellinger k-means", "Multiscale multinomial mixture"],
):
    subset = sections[sections.method == method].sort_values(
        ["tissue", "name"], ascending=[False, True]
    )
    positions = list(range(len(subset)))
    for position, row in zip(positions, subset.itertuples()):
        ax.plot([row.off, row.on], [position, position], color="#aab7c0", linewidth=2)
    ax.scatter(subset.off, positions, c="#777777", label="Adjustment off", zorder=3)
    ax.scatter(subset.on, positions, c="#167b9c", label="Adjustment on", zorder=3)
    labels = [
        name.replace("spinal_", "Spinal ")
        .replace("cortex_", "V1 ")
        .replace("merfish", "macaque")
        .replace("_fresh", " section ")
        for name in subset.name
    ]
    ax.set_yticks(positions, labels)
    ax.axhline(5.5, color="#dddddd", linestyle="--")
    ax.set(xlim=(0.3, 0.66), xlabel="Mean ARI across three seeds", title=title)
axes[0].invert_yaxis()
axes[1].legend(fontsize=9, loc="lower right")
fig.savefig(out / "ablation.png", dpi=180)
fig.savefig(out / "ablation.pdf")
plt.close(fig)

spinal = summary[summary.tissue == "spinal"]
effect = "; ".join(
    f"{row.method}: {row.off:.3f} off → {row.on:.3f} on" for row in spinal.itertuples()
)
wins = (
    sections.assign(improved=sections.on_minus_off > 0)
    .groupby(["tissue", "method"])
    .improved.agg(["sum", "count"])
)
report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Lamination station gradient audit</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#203040}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:7px 12px;border:1px solid #ccd5df;text-align:right}}img{{width:100%}}code,pre{{background:#f0f3f6}}pre{{padding:15px;white-space:pre-wrap}}a{{color:#166892}}</style>
<h1>Are the compositional gradients useful and correctly calculated?</h1>
<p>The local cell-type proportions are valid empirical neighborhood compositions. The quantity called a gradient is instead a normalized directional-imbalance descriptor. It cannot be interpreted as the rate of compositional change or as anatomical depth. The paired ablation shows a modest practical benefit: spinal mean ARI improves by 0.020 with k-means and 0.025 with the multiscale mixture; cortex proxy agreement changes only slightly.</p>
<h2>Paired ablation</h2>
<p><strong>Spinal means: {effect}.</strong> ARI is averaged over three seeds within each section, then equally over sections. Higher is better.</p>
{summary.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}
<p>Number of sections with improved ARI when enabling the adjustment:</p>{wins.to_html(border=0)}
<img src="ablation.png" alt="Per-section ARI with gradient adjustment on versus off">
<p>We fitted the same two pipelines to the same cells, using either ordinary spatial nearest neighbors (<code>elev_weight=0</code>) or the existing adjusted neighborhoods (<code>elev_weight=1</code>). K=8, seeds 17/29/43; Hellinger k-means uses 60 neighbors and 20 initializations. The mixture uses disjoint neighbor ranks 1–15, 16–30, 31–60, with five initializations. The far neighborhood has 90 neighbors. No settings were selected using these new ablation scores. The existing settings were previously selected with adjustment enabled, which may favor that condition.</p>
<p>Six spinal sections use direct polygon annotations. Four cortex sections use composition-derived layer labels: those scores measure agreement with those proxies, not independent anatomical accuracy. These are previously evaluated sections, not a new held-out cohort, and sections may share donors. Fit all cached reference cells (up to 10,000 per section), score the same fixed evaluation cells. This tests the gradient's effect on neighbor selection; it does not separately ablate the gradient and cosine input channels of the neural models.</p>
<h2>What the code actually calculates</h2>
<p>Let p_i be the vector of cell-type proportions in the far neighborhood and r_ij the spatial displacement. Ignoring clipping, the implementation is:</p>
<pre>d_ij = ||r_ij|| + epsilon
w_ij = ||p_j - p_i|| / d_ij
g_i  = sum_j [w_ij r_ij / d_ij] / [sum_j w_ij + epsilon]</pre>
<p>The norm removes the sign and identity of each compositional change before directions are combined. Equal changes on opposite sides cancel. For example, take p_A(x,y)=0.5+0.1x and p_B(x,y)=0.5−0.1x with symmetric neighbors at (±1,0) and (0,±1). The true derivatives are +0.1 and −0.1 along x, but the code's descriptor is exactly (0,0). Removing the left neighbor, without changing this underlying field, yields ({analytic["one_sided_same_ramp_descriptor"][0]:.3f},0). A tenfold weaker ramp yields ({analytic["one_sided_tenfold_weaker_ramp_descriptor"][0]:.3f},0), not one tenth of that value.</p>
<p>These analytic checks use exact compositions supplied to the gradient stage. The same formula was independently reconstructed from the public function's compositions on all ten real sections, including clipping, and matched its output. The problem is the definition, not a disagreement between a test reimplementation and the running code.</p>
<ul><li><strong>Direction:</strong> points toward an imbalance of dissimilar neighbors; it is not reliably the direction of greatest compositional change.</li>
<li><strong>Magnitude:</strong> normalization largely removes composition contrast, and clipping caps the largest values. It is not a derivative magnitude. Spatial sampling and tissue edges can change it even if the underlying composition field is unchanged.</li>
<li><strong>Neighborhood effect:</strong> the code adds an absolute alignment penalty to distance. It favors neighbors perpendicular to this descriptor, potentially preserving boundaries when its direction happens to be appropriate.</li>
<li><strong>Parameters:</strong> <code>n_bins</code> and <code>absgrad</code> are accepted but unused. The docstring already describes the result as a heuristic, not a signed derivative.</li>
<li><strong>Sample scope:</strong> neighborhoods are isolated by sample when requested, but coordinate normalization and clipping/quantile thresholds are pooled. Processing several sections together can therefore change results compared with processing them separately.</li></ul>
<h2>Implication for V1 and spinal cord analysis</h2>
<p>Keep neighborhood composition as the interpretable representation. Retain the existing neighborhood adjustment as an option: it improved five of six spinal sections for k-means and four of six for the mixture. These results support its practical use under the tested settings, not a universal improvement. Do not describe its vectors as validated laminar normals, boundary strength, or cortical/spinal depth. The audited code implements its stated heuristic consistently, but it does not calculate a conventional compositional gradient.</p>
<p>If boundary orientation is useful for the next iteration, a mathematically defensible candidate is to fit a local spatial derivative for each cell-type proportion, preserving signed changes. Combine these derivatives into a 2×2 matrix JᵀJ; its leading eigenvector estimates an unsigned axis of maximal composition change, and its eigenvalues quantify strength and directional ambiguity. This construction recovers the nonzero change axis in the symmetric-ramp example. It still needs bandwidth, sampling, and empirical accuracy checks; this audit does not establish that it improves real region calls. An unsigned axis alone also cannot distinguish superficial from deep cortex.</p>
<p>Gradient-sensitive spatial features are not unique to this project: <a href="https://www.nature.com/articles/s41588-024-01664-3">BANKSY</a> uses azimuthal Gabor features. Its prior benchmark here does not isolate the merits of any particular gradient estimator.</p>
<h2>Checks and reproducibility</h2>
<p>All ten sections passed exact agreement of existing cached adjusted counts with current preprocessing at 15/30/60 neighbors; ordinary 60-neighbor counts matched the public function with zero penalty; and the mathematical descriptor matched the public output. All 120 exported assignments were independently rescored and matched the recorded ARI. The seven existing geometry tests passed. No core package behavior was changed.</p>
{checks.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}
<p><a href="runs.csv">All runs</a> · <a href="section_means.csv">Per-section means</a> · <a href="protocol.json">Protocol</a> · <a href="analytic_checks.json">Analytic checks</a> · <a href="gradient_audit.py">Analysis script</a> · <a href="preprocessing_audited.py">Audited preprocessing snapshot</a> · <a href="prototypes.py">Mixture implementation</a> · <a href="ablation.pdf">Vector figure</a></p>
<h2>Section-level scores</h2>{sections.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)}
</html>"""
(out / "report.html").write_text(report)
assert all(
    (out / item).exists()
    for item in ["ablation.png", "runs.csv", "protocol.json", "preprocessing_audited.py"]
)
print(html.unescape(effect))
print(wins)
