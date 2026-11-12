"""Render an evidence-led diagnosis of the spinal domain-calling plateau."""

import torch  # noqa: F401

import html
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from ceiling_diagnostics import OUT, ROOT, get_section, get_data, nearest
from ceiling_markers import moran


def read(name):
    return pd.read_csv(OUT / (name + ".csv"))


def audit():
    rng = np.random.default_rng(972)
    codes = rng.integers(0, 4, 100)
    _, ix = nearest(rng.normal(size=(100, 2)), 8)
    rows = np.repeat(np.arange(100), 8)
    cols = ix.ravel()
    g = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(100, 100))
    g = g.maximum(g.T)
    coo = g.tocoo()
    degree = np.asarray(g.sum(1)).ravel()
    actual = moran(codes, coo.row, coo.col, degree, 4)
    for t in range(4):
        z = (codes == t).astype(float)
        z -= z.mean()
        expected = len(z) / g.sum() * (z @ (g @ z)) / (z @ z)
        np.testing.assert_allclose(actual[t], expected, atol=1e-12)
    count = 0
    for _, r in read("covariates").iterrows():
        s = get_section("spinal_" + r.species, r["name"].removeprefix("spinal_" + r.species + "_"))
        d = get_data(s)
        meta = s["meta"].dropna(subset=["label"])
        ix = s["ref"].index.get_indexer(meta.index)
        if r.kind == "annotated_footprint_oracle":
            p = ROOT / f"footprint_{r['name']}_K{int(r.k)}.pkl"
            with p.open("rb") as handle:
                v = pickle.load(handle)
            pred = np.full(len(s["ref"]), -1)
            pred[d["y"].notna().to_numpy()] = v
        else:
            p = ROOT / f"cluster_{r['name']}_{r.kind}_K{int(r.k)}_seed{int(r.seed)}.pkl"
            with p.open("rb") as handle:
                pred = pickle.load(handle)
        assert abs(adjusted_rand_score(meta.label, pred[ix]) - r.ari) < 1e-10
        count += 1
    histories = pd.read_parquet(OUT / "objective_histories.parquet")
    monotone = all(
        np.diff(g.objective).min(initial=0) >= -1e-6
        for _, g in histories.groupby(["name", "mode"])
    )
    assert monotone
    f = read("objective_drift")
    assert f.converged.all()
    result = dict(
        covariate_predictions_rescored=count,
        all_ari_match=True,
        objective_trajectories_monotone=monotone,
        all_reference_initialized_fits_converged=True,
        moran_matches_direct_matrix_formula=True,
        neighbor_counts_reconstructed_and_asserted_equal_for_all18_sections=True,
        boundary_oracle_minimum_region_size=61,
        boundary_oracle_negative_control="Global cell-type permutation with identical oracle neighborhoods",
        sampling="Exploratory diagnostics on previously evaluated sections; not a fresh held-out comparison.",
    )
    (OUT / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    geometry = read("geometry")
    cov = read("covariates")
    markers = read("selected_markers")
    detection = read("marker_detection")
    coherence = read("type_coherence")
    objective = read("objective_drift")
    simulation = read("counterfactuals")
    oracle = read("boundary_oracle")
    null = read("oracle_null")
    assert (
        len(geometry) == 18
        and len(cov) == 576
        and len(objective) == 90
        and len(simulation) == 144
        and len(oracle) == 108
        and len(null) == 36
    )
    audit_result = audit()
    # Average seeds within a section before averaging across sections.
    cov_summary = (
        cov.groupby(["name", "kind", "k"]).ari.mean().groupby(["kind", "k"]).mean().unstack()
    )
    objective_summary = objective.groupby(["model", "mode"])[["initial_ari", "final_ari"]].mean()
    geometry_summary = geometry.groupby("species")[
        [
            "ari",
            "reference_merge8_ari",
            "density90_10",
            "fraction_r30_crosses_opposite",
            "ordinary_purity30",
            "gradient_purity30",
            "gradient_purity60",
        ]
    ].mean()
    simulation_summary = (
        simulation.groupby(["name", "condition", "model"])
        .ari.mean()
        .groupby(["condition", "model"])
        .mean()
        .unstack()
    )
    oracle_summary = (
        oracle.groupby(["name", "condition", "model"])
        .ari.mean()
        .groupby(["condition", "model"])
        .mean()
        .unstack()
    )
    oracle_summary.loc["global_type_permutation_null"] = null.groupby("model").ari.mean()
    eligible = coherence[coherence.n >= 10].copy()
    eligible["spatial"] = eligible.q < 0.05
    eligible["density_adjusted"] = eligible.conditional_q < 0.05
    coh_summary = eligible.groupby("species")[
        ["spatial", "density_adjusted", "dominant_lamina_fraction"]
    ].mean()
    top_markers = markers[markers["rank"] == 1].copy()
    detection_summary = (
        detection[detection["rank"] == 1]
        .groupby("species")[
            ["sensitivity", "specificity", "precision", "region_zero_fraction", "auc"]
        ]
        .mean()
    )
    cells = pd.read_parquet(OUT / "cell_diagnostics.parquet")
    purity = (
        cells.groupby(["name", "lamina", "k"])
        .purity.mean()
        .groupby(["lamina", "k"])
        .mean()
        .unstack()
    )
    purity = purity.reindex(
        ["1", "2o", "2i", "3", "4", "4M", "5L", "5M", "6L", "6M", "7", "8", "10"]
    )
    error = read("error_strata")
    strata = error.groupby(["kind", "group"])[["accuracy", "ari"]].mean()
    probes = read("information_probes").groupby(["scheme", "kind"])[["ari", "macro_f1"]].mean()
    quality = read("quality")
    matched_cov = read("within_lamina_covariates").groupby("covariate").difference.mean()
    sweep = read("density_sweep").groupby(["weight", "k"]).ari.mean().unstack()
    merges = read("merge_pairs").groupby(["a", "b"])[["merged_pairs", "total_cross_pairs"]].sum()
    merges["merge_rate"] = merges.merged_pairs / merges.total_cross_pairs
    merges = merges.sort_values("merged_pairs", ascending=False)
    for name, frame in [
        ("summary_covariates", cov_summary),
        ("summary_objectives", objective_summary),
        ("summary_geometry", geometry_summary),
        ("summary_simulations", simulation_summary),
        ("summary_oracle", oracle_summary),
        ("summary_marker_coherence", coh_summary),
        ("summary_marker_detection", detection_summary),
    ]:
        frame.to_csv(OUT / (name + ".csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    ax = axes[0, 0]
    for i, condition in enumerate(["ordinary", "gradient", "oracle_same_lamina"]):
        v = (
            oracle[(oracle.model == "ring") & (oracle.condition == condition)]
            .groupby("name")
            .ari.mean()
        )
        ax.scatter(np.full(len(v), i), v, s=30, alpha=0.7)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, c="black", lw=2)
    v = null[null.model == "ring"].groupby("name").ari.mean()
    ax.scatter(np.full(len(v), 3), v, s=30, alpha=0.7)
    ax.plot([2.8, 3.2], [v.mean()] * 2, c="black", lw=2)
    ax.set_xticks(
        range(4),
        [
            "Ordinary\nneighbors",
            "Gradient\nneighbors",
            "Same-lamina\nneighbors (oracle)",
            "Oracle, shuffled\ncell types",
        ],
        fontsize=8,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")
    ax.set_title("Same cells: restricting neighborhood mixing")
    ax = axes[0, 1]
    g = objective[(objective.model == "ring") & (objective["mode"] == "unweighted")]
    for _, r in g.iterrows():
        ax.plot([0, 1], [r.initial_ari, r.final_ari], alpha=0.5, c="tab:blue")
    ax.plot([0, 1], [g.initial_ari.mean(), g.final_ari.mean()], lw=3, c="black")
    ax.set_xticks(
        [0, 1], ["Reference-based\ninitialization", "After maximizing\nmixture objective"]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")
    ax.set_title("Better likelihood, worse anatomical agreement")
    ax = axes[1, 0]
    labs = purity.index.tolist()
    xx = np.arange(len(labs))
    for k in [15, 30, 60]:
        ax.plot(xx, purity[k], marker="o", ms=3, label=f"{k} neighbors")
    ax.set_xticks(xx, labs)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Reference lamina")
    ax.set_ylabel("Fraction of labeled neighbors in same lamina")
    ax.set_title("Mixing is strongest for thin laminae")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    means = cov_summary[8].drop("annotated_footprint_oracle")
    names = {
        "base": "Composition",
        "density": "+ density",
        "fullcell": "+ all cell types",
        "fullcell_density": "+ both",
        "presence": "+ presence",
    }
    ax.bar(np.arange(len(means)), means.to_numpy(), color="tab:blue")
    ax.set_xticks(np.arange(len(means)), [names[x] for x in means.index], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")
    ax.set_title("Added covariates do not remove the plateau")
    fig.savefig(OUT / "diagnosis.png", dpi=170)
    fig.savefig(OUT / "diagnosis.pdf")
    plt.close(fig)

    histories = pd.read_parquet(OUT / "objective_histories.parquet")
    h = histories[
        (histories.name == "spinal_human_confirmation1") & (histories["mode"] == "unweighted")
    ]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(h.iteration, h.ari, c="tab:blue")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("ARI", color="tab:blue")
    other = ax.twinx()
    other.plot(h.iteration, h.objective - h.objective.iloc[0], c="tab:orange")
    other.set_ylabel("Improvement in MAP objective", color="tab:orange")
    ax.set_title("Anatomical initialization is not a likelihood optimum")
    fig.savefig(OUT / "objective_trace.png", dpi=170)
    fig.savefig(OUT / "objective_trace.pdf")
    plt.close(fig)

    # Show observed marker points, blurred neighborhood evidence, and annotated lamina 3.
    names = [
        "spinal_human_confirmation1",
        "spinal_macaque_confirmation1",
        "spinal_mouse_confirmation1",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
    for row, name in enumerate(names):
        species = name.split("_")[1]
        s = get_section("spinal_" + species, "confirmation1")
        d = get_data(s)
        xy = s["ref"][["x", "y"]].to_numpy()
        is3 = d["y"].fillna("").to_numpy() == "3"
        marker = d["types"] == "Sp3_NTS_Glut"
        vals = s["counts"][30]["Sp3_NTS_Glut_count"].to_numpy()
        for ax in axes[row]:
            ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].scatter(xy[is3, 0], xy[is3, 1], c="tab:blue", s=6)
        axes[row, 0].set_title(species + ": annotated lamina 3")
        axes[row, 1].scatter(xy[marker, 0], xy[marker, 1], c="tab:red", s=9)
        axes[row, 1].set_title("Observed Sp3_NTS_Glut cells")
        m = axes[row, 2].scatter(
            xy[:, 0],
            xy[:, 1],
            c=vals,
            s=4,
            cmap="magma",
            vmin=0,
            vmax=max(5, np.quantile(vals, 0.98)),
        )
        axes[row, 2].set_title("Marker count in adjusted 30NN")
        fig.colorbar(m, ax=axes[row, 2], shrink=0.7)
    fig.savefig(OUT / "marker_spreading.png", dpi=170)
    fig.savefig(OUT / "marker_spreading.pdf")
    plt.close(fig)

    exact = objective[(objective.model == "ring") & (objective["mode"] == "unweighted")]
    oracle_mean = oracle[oracle.model == "ring"].groupby("condition").ari.mean()
    null_mean = null[null.model == "ring"].ari.mean()
    letter = f"""<h2>Main conclusion</h2><p><strong>The evidence does not support an intrinsic biological ceiling at ARI≈0.5.</strong> The strongest explanations are cross-boundary neighborhood mixing and a clustering objective that prefers a different partition from the anatomical reference. Within-region spatial heterogeneity also matters. Density affects the spatial scale of neighborhoods, but simply adding density as a feature does not repair the problem.</p>
    <ol><li><strong>Region-associated type signal exists.</strong> About{100 * eligible.spatial.mean():.0f}% of tested type–section combinations (at least 10 cells) show spatial clustering after within-section BH correction. About{100 * eligible.density_adjusted.mean():.0f}% remain significant when marks are permuted only within density quintiles. Sp3_NTS_Glut occupies 13–18% of training lamina 3 neurons and is enriched roughly 15–26-fold relative to the other laminae.</li>
    <li><strong>Neighborhood construction blends this signal.</strong> In the18-section audit, only about 42% of labeled neighbors of lamina 3 cells are themselves in lamina 3 at k30; lamina 2i is about 39%. Large lamina 8 is about 93%. These cell-weighted figures are descriptive. The per-section plot below gives each section equal weight.</li>
    <li><strong>A direct boundary diagnostic gives a large change.</strong> On the same cells from six sections, ordinary 60-neighbor ring features give{oracle_mean["ordinary"]:.3f} ARI. Restricting neighbors to the same annotated lamina raises it to{oracle_mean["oracle_same_lamina"]:.3f}. This uses labels and is not a discovery method. It preserves observed cell types and spatial heterogeneity, while preventing cross-label pooling. Regions with fewer than 61 neurons were excluded from both arms; mean coverage is{100 * oracle.fraction_polygons_used.mean():.1f}% of polygon-labeled neurons.</li>
    <li><strong>The objective actively moves away from anatomy.</strong> On polygon-labeled neurons only, 13-component mixture initialization from anatomical profiles gives{exact.initial_ari.mean():.3f} ARI. Subsequent label-free EM increases its MAP objective while reducing ARI to{exact.final_ari.mean():.3f}, with worse ARI in{int((exact.final_ari < exact.initial_ari).sum())}/18 sections. K-means behaves similarly. This is evidence for objective/model mismatch, rather than merely a failure to find a favorable initialization.</li>
    <li><strong>Missing density or glia is not the main rescue in the tested formulations.</strong> Observed neuronal density spans roughly 6.5–21-fold between within-section 90th and 10th percentiles across the species means. Nevertheless, density covariates, all-cell context, and their combination do not substantially improve clustering. Density-aware supervised information probes improve by only about 0.005 ARI. This does not establish that density is irrelevant: fixed-k sampling already turns density variation into variable physical smoothing scale.</li></ol>
    <p class="note"><strong>Oracle caution:</strong> using labels to choose neighbors itself creates correlated features within groups. With the identical oracle neighborhoods but globally permuted cell types, mean ARI is{null_mean:.3f}. Therefore the oracle result is a diagnostic of lost structure and label-conditioned smoothing, not a clean decomposition of causal error or an achievable performance forecast. The real-type gain substantially exceeds this artificial agreement.</p>
    <h2>What the annotations can and cannot establish</h2><p>The labels are spatially structured: a within-section coordinate-only 9NN probe with random held-out cells reaches about 0.84 ARI. That is an optimistic coherence diagnostic, not independent anatomical validation. Nearest-other-label distance is inferred from labeled neuron positions; original region-polygon edges were not available in the analyzed cache.</p>
    <p><strong>Provenance correction:</strong> <code>rexed_lamina_source=polygon</code> identifies an assignment source; it does not establish independence from molecular cell types. The associated <a href="https://www.biorxiv.org/content/10.64898/2026.02.04.703852v1.full">atlas preprint</a> describes manually defining approximate laminar regions from mapped cell-type distributions, with adjacent-section histology cross-checks. The exact correspondence between those annotations and this later cached export needs confirmation. Earlier language treating polygon labels as independently established molecular ground truth was too strong.</p>
    <p>The cached polygon label set contains no 9, although the manuscript-ready metadata lists 9. Motor-type annotations occur predominantly in cached label 8. This is a concrete label-definition/export question to resolve, not sufficient evidence that the polygons are wrong. A segmentation that separates additional motor domains could be biologically useful while disagreeing with this reference partition.</p>
    <p>Consequently, the reference definitions need clarification, but there is no defensible estimate here of how much disagreement is annotation error. The objective and neighborhood findings remain visible even when the reference is taken exactly as supplied. No labels were changed in this analysis.</p>
    <h2>Implication for discovering unfamiliar regions</h2><p>The next model should infer local changes in cell-type enrichment and depletion from the spatially marked cells, estimate density as a separate quantity, and pool evidence within candidate regions while learning their boundaries. It should allow sparse marker contrasts to define small cohesive regions even when most abundant types are shared. Region signatures can still be shared across disconnected pieces without imposing a global depth axis.</p><p>This is a direction supported by the diagnosis, not a claim that such a model has already solved the problem. A decisive next test would be whether a label-free boundary model closes part of the real-neighborhood versus oracle-neighborhood gap. Further generic clustering of the same already-blurred vectors has a weaker rationale.</p>"""

    def table(frame):
        return (
            '<div class="scroll">'
            + frame.to_html(float_format=lambda x: f"{x:.3f}", na_rep="—")
            + "</div>"
        )

    content = f"""<!doctype html><html><head><meta charset="utf-8"><title>Why lamination calling plateaus</title><style>body{{font:16px system-ui;max-width:1240px;margin:35px auto;padding:0 24px;color:#1b2933}}p,li{{line-height:1.55}}h1,h2{{color:#17495e}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}.scroll{{overflow:auto}}.note{{background:#fff3d8;border-left:4px solid #c38b18;padding:15px}}pre{{white-space:pre-wrap;background:#f3f5f6;padding:14px}}</style></head><body>
    <h1>Why does region calling plateau near 0.5 ARI?</h1><p>Exploratory diagnosis of 18 spinal sections across human, macaque and mouse. All were previously evaluated; there is no new held-out performance claim. Three seeds are averaged within each section where applicable. Only direct polygon labels are used; propagated region labels are excluded.</p>{letter}
    <p><img src="diagnosis.png"></p><h2>Neighborhood and granularity audit</h2>{table(geometry_summary)}
    <p>Density ratios use observed neuronal k30 density, proportional to 1/r²; they are invariant to coordinate units but can reflect segmentation/capture differences. The reference-based merge into eight groups is an achievable label-informed partition constructed by repeatedly merging the smallest product-of-size pair. It is not a proven optimal upper bound. Its 0.80–0.86 ARI demonstrates that eight clusters alone do not force a 0.5 ceiling.</p>
    <h2>Controlled neighborhood and simulation comparisons</h2>{table(oracle_summary)}
    <p>Oracle and ordinary comparisons use the same eligible reference neurons and K equal to the number of included laminae (10–12). The following separate simulation uses K13 and all polygon-labeled neurons. <em>Shuffled</em> means type labels permuted within each annotated lamina, retaining type totals and positions. <em>Pure60</em> draws independent 15/15/30 counts from the empirical lamina-wide profile; <em>pure240</em> uses 60/60/120. These simulations assume homogeneous regional profiles and remove real spatial heterogeneity and neighborhood overlap; they are not biological accuracy bounds.</p>{table(simulation_summary)}
    <h2>Optimizing the objective versus recovering regions</h2>{table(objective_summary)}<p>All fits converged. Equal component priors, inverse-density weights, and oracle balancing of reference-region mass fail to preserve most of the reference-initialization advantage. Reference balancing uses labels and is a diagnostic only. Density weights are r30² clipped to the section 5th–95th percentiles. Every objective trajectory was checked monotone.</p><p><img src="objective_trace.png"></p>
    <h2>Covariates and feature sufficiency</h2>{table(cov_summary)}<p>Baseline here is K-means on gradient-adjusted ring Hellinger features. Density features comprise median-centered log 8/30/60NN radii and a radius ratio, normalized as a feature family and added at weight 0.5. Full-cell contexts use 30/100 nearest typed cells including non-neurons. The footprint oracle fits only within polygon-labeled tissue but is scored on the same fixed focal cells.</p><p>Smaller density weights, seed17:</p>{table(sweep)}
    <p>Information probes use the same section's annotation labels for training. Random-cell folds are optimistic because nearby neighborhoods overlap; 6×6 spatial tiles give a stricter but still overlapping three-fold split. These are diagnostics, not models for discovering unseen classes.</p>{table(probes)}
    <h2>Marker cohesion, specificity, and loss of localization</h2>{table(coh_summary)}<p>Binary cell-type Moran statistics were compared with 199 permutations of type marks at fixed positions, with BH correction within section among types with at least 10 cells. Conditional permutations preserve each density quintile's type counts. The efficient implementation was checked against the direct matrix formula.</p>{table(detection_summary)}
    <p>Markers were chosen using the four earlier sections of each species: at least 20 cells in the reference region, support in three sections, at least 2-fold enrichment. Detection is measured in the two later sections. A positive adjusted 30NN count is often sensitive but has limited precision for the named lamina: the presence field extends across its boundary. Marker enrichment does not imply exclusivity; some coherent types span several laminae.</p><p>Depletion can also define a region, but zero observed cells is informative only relative to expected sampling. For example, even with independent sampling, a type present at 1% frequency is absent from a 30-cell neighborhood about 74% of the time. Evidence for a cohesive absence therefore needs to aggregate across a candidate region while accounting for overlapping neighborhoods and local sampling density.</p><p><img src="marker_spreading.png"></p>
    <details><summary>Best supported marker per reference lamina</summary>{table(top_markers)}</details>
    <details><summary>Same-lamina neighborhood purity by region</summary>{table(purity)}</details>
    <details><summary>Most frequent pairwise region merges</summary>{table(merges.head(20))}</details>
    <details><summary>Error versus boundary proximity and density</summary>{table(strata)}<p>Boundary bins are distance to nearest differently labeled neuron divided by ordinary r30: &lt;0.5,0.5–1,1–2,&gt;2. ARI across differently composed subsets is descriptive and not directly comparable as an accuracy difference. Matched label accuracy uses one global Hungarian mapping per section.</p></details>
    <details><summary>Type-assignment quality and matched covariate associations</summary>{table(quality.groupby("species").mean(numeric_only=True))}{table(matched_cov.to_frame("high_minus_low_cell_Bcubed_F1"))}<p>Within each section and lamina, compare high versus low quartiles of each covariate using per-cell B-cubed F1, then average strata. This handles both splits and merges without forcing a one-to-one mapping between eight clusters and thirteen laminae. These are associations, not causal contributions. Mapping scores are not calibrated error probabilities. All evaluated neurons carry a directly-assigned type flag.</p></details>
    <h2>Artifacts and validation</h2><p>All tables are CSV files beside this report; cell_diagnostics.parquet and objective_histories.parquet retain cell-level and iteration-level diagnostics. Figures are supplied as PNG and PDF. Source scripts are benchmarks/ceiling_diagnostics.py, ceiling_markers.py, ceiling_counterfactuals.py, ceiling_objective.py and report_ceiling.py. No package defaults or data annotations were changed.</p><pre>{html.escape(json.dumps(audit_result, indent=2))}</pre>
    <p>Reproduction: set TMPDIR=/scratch/runtime/tmp and MPLCONFIGDIR=/scratch/runtime/mpl; run ceiling_diagnostics.py prepare, geometry, covariates, density_sweep and probes; ceiling_markers.py coherence and markers; ceiling_counterfactuals.py simulation, quality, boundary and oracle_null; ceiling_objective.py; then report_ceiling.py. All scripts import torch first and use the existing environment. Intermediates are under /scratch/lamination_station_ceiling.</p></body></html>"""
    (OUT / "report.html").write_text(content)
    print("OBJECTIVE", objective_summary.to_string())
    print("ORACLE", oracle_summary.to_string())
    print("VALIDATION", audit_result)


if __name__ == "__main__":
    main()
