"""Validate exported assignments and render the complete incremental experiment."""

import torch  # noqa: F401

import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from incremental_domains import (
    CACHE,
    DATASETS,
    OUT,
    PREVIOUS,
    SEEDS,
    features,
    load_section,
    read_pickle,
    score,
    section_graphs,
)

frames = [pd.read_csv(OUT / "evaluation.csv"), pd.read_csv(OUT / "dispersion_evaluation.csv")]
assert frames[1].optimizer_failures.sum() == 0
runs = pd.concat(frames, ignore_index=True)
assert len(runs) == 240
assert not runs.duplicated(["name", "method", "seed"]).any()
selected = json.loads((OUT / "selected.json").read_text())
assert (OUT / "selected.json").read_bytes() == (CACHE / "selected.json").read_bytes()
historical, manifest = [], []
models = OUT / "reference_models"
models.mkdir(exist_ok=True)
maps = OUT / "maps"
maps.mkdir(exist_ok=True)

for dataset in DATASETS:
    tissue = "spinal" if dataset.startswith("spinal") else "cortex"
    training_sections = [load_section(dataset, split) for split in ["development", "test"]]
    for seed in SEEDS:
        training = read_pickle(CACHE / f"final_{dataset}_training_{seed}.pkl")
        np.savez_compressed(
            models / f"{dataset}_{seed}.npz",
            profiles=np.stack(training["fit"]["profiles"]),
            columns=np.array(training["columns"], dtype=str),
            weights=training["fit"]["weights"],
            compatibility=training["compatibility"],
            train_names=np.array(training["train_names"], dtype=str),
        )
    training = read_pickle(CACHE / f"final_{dataset}_training_17.pkl")
    for split in ["fresh1", "fresh2"]:
        section = load_section(dataset, split)
        assert all(
            section["ref"].section.iloc[0] != train["ref"].section.iloc[0]
            for train in training_sections
        )
        assert all(
            not section["ref"].index.isin(train["ref"].index).any() for train in training_sections
        )
        rings = features(section, training["columns"])
        graph = section_graphs(section, rings)[0]
        ref = section["ref"]
        manifest.append(
            dict(
                name=section["name"],
                unseen_neighborhood_fraction=float(sum(rings)[:, -1].sum() / (60 * len(ref))),
                train_evaluation_sections_disjoint=True,
                train_evaluation_cells_disjoint=True,
            )
        )
        exported = ref.copy()
        exported["evaluation_label"] = section["meta"].label.reindex(ref.index)
        exported.to_csv(OUT / "predictions" / f"{section['name']}_cells.csv.gz")
        for row in runs[runs.name == section["name"]].itertuples():
            prediction = np.load(OUT / "predictions" / f"{row.name}_{row.method}_{row.seed}.npz")
            np.testing.assert_array_equal(prediction["index"], ref.index.astype(str).to_numpy())
            q = prediction["q"]
            assert np.isfinite(q).all() and (q >= 0).all()
            np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-10)
            rescored = score(section, q, graph)
            for metric in ["ari", "boundary_f1", "boundary_brier", "boundary_ap"]:
                assert abs(rescored[metric] - getattr(row, metric)) < 1e-10
        for seed in SEEDS:
            old = read_pickle(PREVIOUS / f"{section['name']}_ring_multinomial_K8_seed{seed}.pkl")
            historical.append(
                dict(
                    name=section["name"],
                    dataset=dataset,
                    tissue=tissue,
                    method="historical_ring",
                    seed=seed,
                    **score(section, old["q"], graph),
                )
            )

        if "--reuse-maps" in sys.argv:
            continue
        # Matched colors are for display only; the clustering and ARI need no label mapping.
        known = section["meta"].label.dropna()
        names = sorted(known.astype(str).unique())
        codes = {label: i for i, label in enumerate(names)}
        truth = section["meta"].label.reindex(ref.index).map(codes).fillna(-1).to_numpy(dtype=int)
        palette = plt.get_cmap("tab20")(np.arange(max(20, len(names))))
        xy = ref[["x", "y"]].to_numpy()
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
        for ax, method in zip(axes, ["reference", "local", "shared", "local_dm"], strict=True):
            colors = np.tile([0.8, 0.8, 0.8, 1.0], (len(ref), 1))
            if method == "reference":
                valid = truth >= 0
                colors[valid] = palette[truth[valid]]
                title = "Reference (gray = unscored)"
            else:
                q = np.load(OUT / "predictions" / f"{section['name']}_{method}_17.npz")["q"]
                labels = q.argmax(1)
                valid = truth >= 0
                counts = np.zeros((8, len(names)))
                np.add.at(counts, (labels[valid], truth[valid]), 1)
                rr, cc = linear_sum_assignment(-counts)
                for cluster, reference in zip(rr, cc, strict=True):
                    colors[labels == cluster] = palette[reference]
                ari = runs[
                    (runs.name == section["name"]) & (runs.method == method) & (runs.seed == 17)
                ].ari.iloc[0]
                title = f"{method}; ARI {ari:.3f}"
            ax.scatter(
                xy[:, 0], xy[:, 1], c=colors, s=1.5 if len(ref) < 3000 else 0.6, linewidths=0
            )
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(section["name"] + "; seed17; colors matched for display only", fontsize=11)
        fig.legend(
            handles=[Patch(facecolor=palette[i], label=label) for i, label in enumerate(names)],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(8, len(names)),
            fontsize=8,
        )
        fig.savefig(maps / f"{section['name']}.png", dpi=170, bbox_inches="tight")
        fig.savefig(maps / f"{section['name']}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
        for pair, method in enumerate(["local", "shared_adjacency"]):
            sensitivity = (
                pd.read_csv(
                    OUT / "predictions" / f"{section['name']}_{method}_uncertainty.csv.gz",
                    dtype={"index": str},
                )
                .set_index("index")
                .reindex(ref.index.astype(str))
            )
            assert sensitivity.notna().all().all()
            for offset, column in enumerate(["confidence", "block_assignment_agreement"]):
                ax = axes[2 * pair + offset]
                points = ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=sensitivity[column],
                    s=1.5 if len(ref) < 3000 else 0.6,
                    linewidths=0,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
                ax.set_aspect("equal")
                ax.set_title(f"{method}\n{column}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(points, ax=axes, shrink=0.6, label="Model score / assignment agreement")
        fig.suptitle(
            section["name"] + "; confidence and conditional block sensitivity", fontsize=11
        )
        fig.savefig(maps / f"{section['name']}_uncertainty.png", dpi=170)
        fig.savefig(maps / f"{section['name']}_uncertainty.pdf")
        plt.close(fig)

runs = pd.concat([runs, pd.DataFrame(historical)], ignore_index=True)
runs.to_csv(OUT / "all_runs.csv", index=False)
pd.DataFrame(manifest).to_csv(OUT / "verification.csv", index=False)
means = (
    runs.groupby(["name", "tissue", "method"])[
        ["ari", "boundary_f1", "boundary_brier", "boundary_ap", "occupied"]
    ]
    .mean()
    .reset_index()
)
means.to_csv(OUT / "all_section_means.csv", index=False)
summary = means.groupby(["tissue", "method"]).mean(numeric_only=True)
summary.to_csv(OUT / "all_summary.csv")
pivot = means.pivot(index=["name", "tissue"], columns="method", values="ari")
deltas = pd.DataFrame(
    {
        "Shared profiles": pivot.shared - pivot.local,
        "Gated smoothing after sharing": pivot.shared_gated - pivot.shared,
        "Nonzero adjacency after smoothing": pivot.shared_adjacency_nonzero - pivot.shared_gated,
        "Overdispersion without sharing": pivot.local_dm - pivot.local,
    }
).reset_index()
deltas.to_csv(OUT / "incremental_deltas.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
for ax, tissue, color in zip(axes, ["spinal", "cortex"], ["#267b9c", "#b77b2c"], strict=True):
    subset = deltas[deltas.tissue == tissue]
    for i, label in enumerate(deltas.columns[2:]):
        values = subset[label].to_numpy()
        ax.scatter(values, i + np.linspace(-0.13, 0.13, len(values)), color=color, s=30)
        ax.scatter(values.mean(), i, color="black", marker="|", s=220, zorder=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(4), deltas.columns[2:])
    ax.set(
        xlabel="Change in ARI; each dot is one section",
        title="Spinal: polygon labels"
        if tissue == "spinal"
        else "V1: composition-derived proxies",
    )
axes[0].invert_yaxis()
fig.savefig(OUT / "increments.png", dpi=180)
fig.savefig(OUT / "increments.pdf")
plt.close(fig)

# Label-permutation-invariant seed stability, evaluated on all reference cells.
stability = []
for name, frame in runs[runs.method != "historical_ring"].groupby("name"):
    for method in frame.method.unique():
        labels = [
            np.load(OUT / "predictions" / f"{name}_{method}_{seed}.npz")["q"].argmax(1)
            for seed in SEEDS
        ]
        pairwise = [adjusted_rand_score(labels[i], labels[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        stability.append(
            dict(
                name=name,
                method=method,
                tissue=frame.tissue.iloc[0],
                mean_seed_ari=np.mean(pairwise),
            )
        )
pd.DataFrame(stability).to_csv(OUT / "seed_stability.csv", index=False)

uncertainty = (
    pd.read_csv(OUT / "uncertainty.csv").groupby(["tissue", "method"]).mean(numeric_only=True)
)
thinning = pd.read_csv(OUT / "thinning.csv").groupby(["tissue", "method"]).mean(numeric_only=True)
calibration = (
    pd.read_csv(OUT / "boundary_calibration.csv")
    .groupby(["tissue", "method"])[["raw_brier", "calibrated_brier", "constant_brier"]]
    .mean()
)
diagnostics = (
    runs[runs.method != "historical_ring"].groupby("method").converged.agg(["sum", "count"])
)
main_methods = ["local", "shared", "shared_gated", "shared_adjacency_nonzero", "local_dm"]
display = (
    summary.reset_index()
    .query("method in @main_methods")
    .pivot(index="method", columns="tissue", values="ari")
    .reindex(main_methods)
)
check_data = json.loads((OUT / "numerical_checks.json").read_text())
maps_html = "\n".join(
    f'<details><summary>{html.escape(name)}</summary><img src="maps/{name}.png" alt="Region assignments for {name}"><a href="maps/{name}.pdf">Vector PDF</a><img src="maps/{name}_uncertainty.png" alt="Confidence and spatial-block sensitivity for {name}"><a href="maps/{name}_uncertainty.pdf">Sensitivity PDF</a></details>'
    for name in sorted(runs.name.unique())
)
capacity = pd.read_csv(OUT / "capacity_evaluation.csv")
assert len(capacity) == 36
for name, section_rows in capacity.groupby("name"):
    dataset, suffix = name.rsplit("_fresh", 1)
    section = load_section(dataset, "fresh" + suffix)
    training = read_pickle(CACHE / f"capacity_{dataset}_training_17.pkl")
    graph = section_graphs(section, features(section, training["columns"]))[0]
    for row in section_rows.itertuples():
        prediction = np.load(OUT / "predictions" / f"{name}_{row.method}_K12_{row.seed}.npz")
        assert prediction["q"].shape == (len(section["ref"]), 12)
        np.testing.assert_array_equal(
            prediction["index"], section["ref"].index.astype(str).to_numpy()
        )
        np.testing.assert_allclose(prediction["q"].sum(1), 1.0, atol=1e-10)
        for metric, value in score(section, prediction["q"], graph).items():
            if metric in ["ari", "boundary_f1", "boundary_brier"]:
                assert abs(value - getattr(row, metric)) < 1e-10
capacity_summary = capacity.groupby("method")[["ari", "boundary_f1", "occupied"]].mean()
capacity_summary.to_csv(OUT / "capacity_summary.csv")
report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Incremental domain modeling</title>
<style>body{{max-width:1150px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#203040}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:6px 10px;border:1px solid #ccd5df;text-align:right}}img{{width:100%}}pre{{padding:16px;background:#f0f3f6;white-space:pre-wrap}}a{{color:#166892}}details{{margin:12px 0}}</style>
<h1>Incremental domain modeling: shared profiles help V1 proxy agreement, not spinal accuracy overall</h1>
<p>All experiments retain gradient-adjusted multiscale neighborhood compositions. The main comparison changes region modeling one step at a time. Shared profiles improve the V1 proxy mean from {summary.loc[("cortex", "local"), "ari"]:.3f} to {summary.loc[("cortex", "shared"), "ari"]:.3f}, but reduce spinal mean ARI from {summary.loc[("spinal", "local"), "ari"]:.3f} to {summary.loc[("spinal", "shared"), "ari"]:.3f}. Boundary-sensitive smoothing contributes little; development selection disables learned adjacency in both tissues. These results support retaining sharing as an experimental V1 option, not changing the general default.</p>
<h2>Accuracy and increments</h2>
{display.to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<p>Local = independent multinomial mixture. Shared = target profiles regularized toward training profiles. Shared_gated adds the development-selected boundary-sensitive spatial term. Shared_adjacency_nonzero is the best nonzero development adjacency setting, included as a diagnostic: the actual selected adjacency setting is zero. Local_dm replaces the multinomial likelihood with a Dirichlet–multinomial, independently of sharing.</p>
<img src="increments.png" alt="Per-section gains and losses from each modeling increment">
<p>Dots are section-level differences after averaging three seeds; black bars indicate mean differences. A positive change improves ARI. The same cell types, cells, gradient-adjusted neighborhoods, K=8, and reference labels are used within each contrast. No evaluation-driven tuning or combination of the best evaluation branches was performed.</p>
<h2>What was implemented</h2>
<ol><li><strong>Shared region profiles:</strong> a balanced joint multiscale mixture on two training sections, with sample-specific prevalences. Target profiles adapt under Dirichlet MAP priors centered on training profiles. The chosen prior is 100 pseudo-cells per component per ring, for both tissues. Profiles are shared only within species/platform datasets. Each piece may omit regions; repeated disconnected instances are allowed.</li>
<li><strong>Boundary-sensitive coherence:</strong> fixed composition evidence plus symmetric mutual 12NN spatial coupling. Physical edges have a distance cap; a Hellinger-composition gate weakens edges across compositional changes. Gating follows physical-degree normalization to preserve attenuation. Uniform smoothing and smoothing without sharing are included as controls.</li>
<li><strong>Learned adjacency:</strong> training-only physical contacts between confident region assignments, regularized against endpoint-abundance expectations. Off-diagonal compatibility is symmetric, bounded, and distinct from same-region smoothing. It is a fixed training-derived potential, not a contact probability or a learned anatomical ordering.</li>
<li><strong>Overdispersion:</strong> a separate fixed-concentration Dirichlet–multinomial mixture, initialized from the multinomial fit. Mean profiles are optimized using the actual DM likelihood; the concentration is selected on development sections.</li>
<li><strong>Uncertainty:</strong> seed stability, entropy, spatial-block reweighting, cell-thinning sensitivity, and development-fitted calibration of edge-boundary scores. Cell memberships remain uncalibrated model scores.</li></ol>
<h2>Locked development choices</h2><pre>{html.escape(json.dumps(selected, indent=2))}</pre>
<p>DM concentrations: {html.escape((OUT / "dispersion_selected.json").read_text())}. Two-fold section cross-validation uses the historical development/test sections, seed17. Choices maximize mean ARI equally across sections/datasets, separately for spinal and V1. Final reference profiles are fitted to both historical sections, then applied to later sections. Evaluation uses seeds17,29,43 and full reference cells, with the same fixed labeled focal subset as earlier benchmarks.</p>
<h2>Interpretation and limits</h2>
<p>The six spinal evaluation sections have direct polygon annotations. Four V1 sections have composition-derived layer proxies, so improved V1 agreement is not independent anatomical validation. All evaluation sections have been examined in earlier work; they were reserved for this experiment's settings but are not a new cohort. Some sections may share donors. Only two training sections per dataset are available here.</p>
<p>The common training-derived cell-type vocabulary includes an unseen-type bucket, preserving every neighborhood count. The local control uses that same vocabulary. Historical ring-mixture scores using the original per-section vocabulary are included below to expose any baseline shift. Training/evaluation cell and section identities were checked for disjointness.</p>
<p>Known piece IDs are respected by the experimental graph builder and synthetic tests. The cached real sections have no manually verified fragment masks: their graphs use distance caps, which do not guarantee separation of closely apposed fragments. Likewise, the synthetic missing-region check demonstrates model capability, not exhaustive recovery of cropped real layers.</p>
<p>Graph refinement uses fixed composition scores and monotone damped mean-field updates, not joint retraining of all parameters. Negative adjacency results apply to this regularized potential and finite parameter grid, not every possible relational domain model. No expression information, anatomical template, or forced layer ordering was introduced.</p>
<h2>Exploratory spinal capacity check</h2>
<p>Inspection of the K=8 maps showed merged spinal regions; some references contain more than eight labels. We therefore repeated local-versus-shared fitting at K=12, with the same 100-pseudo-cell pooling strength and all three seeds. This follow-up was prompted by evaluation results and is exploratory; it was not used to retune the main comparison.</p>
{capacity_summary.to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<p><a href="capacity_evaluation.csv">All 36 capacity-check scores</a> · <a href="capacity_protocol.json">Follow-up protocol</a>. Their exported probabilities were independently rescored too. Increasing capacity does not by itself establish a generally better shared model.</p>
<h2>Boundary agreement</h2>
{summary[["boundary_f1", "boundary_ap", "boundary_brier", "occupied"]].to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<p>These metrics concern physical graph edges whose endpoints both have reference labels. F1/AP score region-label changes, not geometric boundary distances. Raw boundary score is 1−q_i·q_j, a mean-field marginal-product approximation; correlated neighboring assignments mean this is not automatically a calibrated joint probability. Boundary Brier alone can reward over-smoothing because boundaries are rare; interpret it with F1/AP and the maps.</p>
<h2>Calibration and robustness</h2>
{calibration.to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<p>A monotone logistic transform is fitted to development-only edge scores, with equal section weights, separately for each tissue and method. Lower Brier is better. The constant comparator uses development boundary prevalence. This is calibration of edge-boundary scores only; it does not calibrate individual-cell region probabilities or change region assignments.</p>
{uncertainty.to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<p>Five exponential-weight perturbations of a 6×6 spatial tiling refit target parameters, conditional on fixed neighborhoods and training profiles. Labels are matched to the original assignments without anatomical labels. The unstable-confident fraction is the fraction of all cells with original maximum membership≥0.9 but agreement&lt;0.8 across perturbations. These are sensitivity diagnostics, not confidence intervals. Warm starts and fixed training profiles limit the uncertainty captured.</p>
{thinning.to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<p>The thinning stress test retains 80% of reference cells once per section and recomputes gradient neighborhoods and graphs. Both original and thinned scores use identical retained evaluation cells. Shared-model stability can improve even when anatomical accuracy decreases, so stability is not a substitute for reference validation.</p>
<h2>Region maps</h2><p>Seed17. Predicted cluster colors are optimally matched to reference labels for display only. Gray predicted cells belong to unmatched clusters; gray reference cells are unscored. Numerical scores do not use this matching.</p>{maps_html}
<h2>Checks and convergence</h2><p>All 240 main assignment matrices and 36 exploratory capacity-check matrices passed index alignment and independent score-recomputation checks. All {len(check_data)} synthetic/numerical checks passed, including disconnected-piece isolation, rigid-transform invariance, missing regions, monotone optimization, and the DM likelihood's agreement with its gamma-function definition. All final DM inner optimization steps converged using the constrained Newton solver.</p>
{diagnostics.to_html(border=0)}<p>Sum = number converged; count = fits. Limits are 300 EM iterations for multinomial fits,200 graph iterations, and 80 DM iterations. Inspect saved optimizer diagnostics for any finite-budget limitations.</p>
<h2>Full section-level scores</h2>{pivot.to_html(float_format=lambda x: f"{x:.4f}", border=0)}
<h2>Files and reproduction</h2>
<p><a href="all_runs.csv">All scores</a> · <a href="all_section_means.csv">Section means</a> · <a href="tuning.csv">Development grid</a> · <a href="protocol.json">Protocol</a> · <a href="numerical_checks.json">Numerical checks</a> · <a href="verification.csv">Split and vocabulary checks</a> · <a href="seed_stability.csv">Seed stability</a> · <a href="boundary_calibration.csv">Calibration</a> · <a href="increments.pdf">Vector figure</a></p>
<p>Experimental implementations: <a href="incremental_models.py">models</a>, <a href="incremental_domains.py">main experiment</a>, <a href="incremental_dispersion.py">overdispersion</a>, <a href="incremental_uncertainty.py">robustness</a>, <a href="incremental_calibration.py">calibration</a>. Reference model arrays are in reference_models/; per-cell probability arrays, coordinates, and annotations are in predictions/. All fits are also cached under /scratch/lamination_station_incremental. No core package behavior or dependency versions were changed.</p>
<pre>export TMPDIR=/scratch/runtime/tmp MPLCONFIGDIR=/scratch/runtime/mpl
python benchmarks/check_incremental_models.py
python benchmarks/incremental_domains.py all
python benchmarks/incremental_dispersion.py all
python benchmarks/incremental_uncertainty.py
python benchmarks/incremental_calibration.py
python benchmarks/incremental_capacity.py
python benchmarks/report_incremental.py</pre>
<p>Run from the lamination_station repository in this capsule; the preparation stage uses the earlier benchmark's cached sections. Input section identities and counts are listed in input_manifest.json. This is an experimental implementation, not a replacement public package API.</p>
</html>"""
(OUT / "report.html").write_text(report)
sources = [
    "incremental_models.py",
    "incremental_domains.py",
    "incremental_dispersion.py",
    "incremental_uncertainty.py",
    "incremental_calibration.py",
    "incremental_capacity.py",
    "check_incremental_models.py",
    "report_incremental.py",
]
for name in sources:
    shutil.copy2(Path(__file__).with_name(name), OUT / name)
(OUT / "source_hashes.json").write_text(
    json.dumps(
        {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in sources}, indent=2
    )
)
print(display)
print("All 240 exported predictions independently verified.")
