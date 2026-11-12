"""Summarize the fixed-type categorical–Potts experiments and their limitations."""

import torch  # noqa: F401

import html
import importlib.metadata
import itertools
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from run_bass_fixed_types import OUT, ROOT, VERSION, DATASETS, get_section, baseline


def read_family(prefix, suffix=""):
    return pd.concat(
        [pd.read_csv(OUT / f"{prefix}_{d}{suffix}.csv") for d in DATASETS], ignore_index=True
    )


def load(key):
    with (ROOT / (VERSION + "_" + key + ".pkl")).open("rb") as handle:
        return pickle.load(handle)


def table(frame):
    return '<div class="scroll">' + frame.to_html(float_format=lambda v: f"{v:.3f}") + "</div>"


def matched_colors(truth, prediction, valid, n_labels):
    k = prediction.max() + 1
    contingency = np.zeros((n_labels, k), int)
    np.add.at(contingency, (truth[valid], prediction[valid]), 1)
    r, c = linear_sum_assignment(-contingency)
    mapping = np.arange(k) + n_labels
    mapping[c] = r
    return mapping[prediction]


def maps(evaluation, shared, tissue):
    datasets = [d for d in DATASETS if d.startswith(tissue)]
    fig, axes = plt.subplots(
        len(datasets), 4, figsize=(15, 4.5 * len(datasets)), layout="constrained"
    )
    for row, dataset in enumerate(datasets):
        split = "confirmation1" if tissue == "spinal" else "fresh1"
        s = get_section(dataset, split)
        ref = s["ref"]
        if tissue == "spinal":
            from ceiling_diagnostics import get_data

            labels = get_data(s)["y"]
        else:
            labels = s["meta"].label.reindex(ref.index)
        categories = sorted(labels.dropna().astype(str).unique())
        truth = pd.Categorical(labels.astype("string"), categories=categories).codes
        valid = truth >= 0
        record = evaluation[
            (evaluation.name == s["name"])
            & (evaluation.k == 8)
            & (evaluation.seed == 17)
            & (evaluation.method == "bass")
        ].iloc[0]
        local = load(record.cache_key)["q"].argmax(1)
        rec = shared[(shared.name == s["name"]) & (shared.k == 8) & (shared.seed == 17)].iloc[0]
        joint = load(rec.cache_key)["q"].argmax(1)
        splits = ["development", "test", "fresh1", "fresh2"]
        if tissue == "spinal":
            splits += ["confirmation1", "confirmation2"]
        offset = sum(len(get_section(dataset, v)["ref"]) for v in splits[: splits.index(split)])
        joint = joint[offset : offset + len(ref)]
        original = baseline(s, 8, 17)["q"].argmax(1)
        cmap = plt.get_cmap("tab20", max(20, len(categories) + 8))
        for ax, title, pred in zip(
            axes[row],
            ["Reference", "Gradient ring", "BASS-style", "Shared profiles"],
            [truth, original, local, joint],
        ):
            ax.scatter(ref.x, ref.y, s=2, c="#dddddd", linewidths=0)
            if title == "Reference":
                ax.scatter(
                    ref.x[valid],
                    ref.y[valid],
                    s=3,
                    c=truth[valid],
                    cmap=cmap,
                    vmin=0,
                    vmax=cmap.N - 1,
                    linewidths=0,
                )
                label = "polygon labels" if tissue == "spinal" else "composition proxy"
                ax.set_title(f"{dataset}: {label}", fontsize=10)
            else:
                colors = matched_colors(truth, pred, valid, len(categories))
                ax.scatter(
                    ref.x, ref.y, s=3, c=colors, cmap=cmap, vmin=0, vmax=cmap.N - 1, linewidths=0
                )
                # Report ARI on the same fixed focal evaluation set as the table.
                meta = s["meta"].dropna(subset=["label"])
                ids = ref.index.get_indexer(meta.index)
                ari = adjusted_rand_score(meta.label, pred[ids])
                ax.set_title(f"{title}: ARI {ari:.3f}", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(OUT / f"{tissue}_maps.png", dpi=160)
    fig.savefig(OUT / f"{tissue}_maps.pdf")
    plt.close(fig)


def main():
    evaluation = read_family("evaluation")
    shared = read_family("shared")
    validation = read_family("validation")
    shared_validation = read_family("shared", "_masked")
    pilot = read_family("pilot")
    gradient = read_family("evaluation", "_gradientinit")
    selected = pd.read_csv(OUT / "selected.csv")
    long_chains = read_family("long")
    strength = pd.read_csv(OUT / "strength_selection.csv")
    annotated = pd.concat(
        [
            pd.read_csv(OUT / f"annotated_tuning_{d}.csv")
            for d in DATASETS
            if d.startswith("spinal")
        ]
    )
    assert len(pilot) == 160 and len(evaluation) == 168 and len(shared) == 84
    assert len(validation) == 48 and len(gradient) == 16
    # Each section, not each cell or fitted seed, gets equal weight.
    combined = pd.concat([evaluation, shared], ignore_index=True)
    per_section = (
        combined.groupby(["dataset", "name", "k", "method"])[
            ["ari", "boundary_f1", "occupied", "thin_best_iou"]
        ]
        .mean()
        .reset_index()
    )
    per_section["tissue"] = per_section.dataset.str.split("_").str[0]
    summary = per_section.groupby(["dataset", "k", "method"]).ari.mean().unstack()
    summary["local_minus_ring"] = summary.bass - summary.gradient_ring
    summary["shared_minus_ring"] = summary.bass_shared - summary.gradient_ring
    tissue_summary = per_section.groupby(["tissue", "k", "method"]).ari.mean().unstack()
    paired = per_section.pivot(index=["dataset", "name", "k"], columns="method", values="ari")
    paired["local_gain"] = paired.bass - paired.gradient_ring
    paired["shared_gain"] = paired.bass_shared - paired.gradient_ring
    paired.to_csv(OUT / "paired_sections.csv")
    summary.to_csv(OUT / "summary_ari.csv")
    tissue_summary.to_csv(OUT / "summary_tissue.csv")
    # Local masked-type baselines select neighborhood size on pilot sections only.
    knn_pilot = read_family("knn_pilot")
    knn_pilot["tissue"] = knn_pilot.dataset.str.split("_").str[0]
    knn_grid = knn_pilot.groupby(["tissue", "k"]).nll.mean().reset_index()
    knn_choice = knn_grid.loc[knn_grid.groupby("tissue").nll.idxmin(), ["tissue", "k"]]
    knn_validation = read_family("knn_validation")
    knn_validation["tissue"] = knn_validation.dataset.str.split("_").str[0]
    chosen_knn = knn_validation.merge(knn_choice, on=["tissue", "k"])
    prediction = (
        validation.groupby(["dataset", "name", "k"])
        .agg(
            bass_nll=("nll", "mean"),
            global_nll=("base_nll", "mean"),
            half_nll_gap=("half_nll_gap", "mean"),
        )
        .reset_index()
    )
    prediction = prediction.merge(
        chosen_knn[["name", "nll"]].rename(columns={"nll": "knn_nll"}), on="name"
    )
    prediction = prediction.merge(
        shared_validation[["name", "k", "nll"]].rename(columns={"nll": "shared_nll"}),
        on=["name", "k"],
    )
    pred_summary = prediction.groupby(["dataset", "k"])[
        ["bass_nll", "shared_nll", "knn_nll", "global_nll", "half_nll_gap"]
    ].mean()
    pred_summary.to_csv(OUT / "summary_prediction.csv")
    # Seed reproducibility is not a formal posterior-convergence guarantee.
    stability = []
    for (name, k), group in evaluation[evaluation.method == "bass"].groupby(["name", "k"]):
        predictions = [load(r.cache_key)["q"].argmax(1) for _, r in group.iterrows()]
        stability.append(
            dict(
                name=name,
                dataset=group.dataset.iloc[0],
                k=k,
                seed_ari=np.mean(
                    [adjusted_rand_score(a, b) for a, b in itertools.combinations(predictions, 2)]
                ),
                sample_to_modal_ari=group.sample_ari_to_modal.mean(),
            )
        )
    stability = pd.DataFrame(stability)
    stability.to_csv(OUT / "seed_stability.csv", index=False)
    regions = pd.concat(
        [pd.read_csv(OUT / f"regions_{d}.csv") for d in DATASETS if d.startswith("spinal")]
    )
    joint_regions = pd.concat(
        [pd.read_csv(OUT / f"shared_regions_{d}.csv") for d in DATASETS if d.startswith("spinal")]
    )
    region_summary = (
        pd.concat([regions, joint_regions])
        .groupby(["k", "lamina", "method"])
        .best_iou.mean()
        .unstack()
    )
    region_summary.to_csv(OUT / "summary_regions.csv")
    # Validate cached results against the scored focal rows.
    rescored = 0
    for _, r in evaluation[evaluation.method == "bass"].iterrows():
        result = load(r.cache_key)
        s = get_section(r.dataset, r["name"].removeprefix(r.dataset + "_"))
        meta = s["meta"].dropna(subset=["label"])
        ids = s["ref"].index.get_indexer(meta.index)
        actual = adjusted_rand_score(meta.label, result["q"].argmax(1)[ids])
        assert abs(actual - r.ari) < 1e-10
        np.testing.assert_allclose(result["q"].sum(1), 1, atol=1e-12)
        rescored += 1
    checks = json.loads((OUT / "numerical_checks.json").read_text())
    checks["real_data_partitions_rescored"] = rescored
    versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "scipy", "pandas", "scikit-learn", "torch", "numba"]
    }
    (OUT / "versions.json").write_text(json.dumps(versions, indent=2))
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for dataset in DATASETS:
        frame = paired.loc[dataset].xs(8, level="k")
        axes[0].scatter(frame.gradient_ring, frame.bass, label=dataset, s=45)
        axes[1].scatter(frame.gradient_ring, frame.bass_shared, label=dataset, s=45)
    for ax, title in zip(axes, ["Independent sections", "Profiles shared across sections"]):
        ax.plot([0, 1], [0, 1], color="gray", ls="--")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Gradient-ring ARI",
            ylabel="BASS-style ARI",
            title=title,
        )
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "comparison.png", dpi=170)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    for tissue in ["spinal", "cortex"]:
        maps(evaluation, shared, tissue)
    init_comparison = gradient[gradient.method == "bass"][["name", "k", "ari"]].rename(
        columns={"ari": "gradient_initialization"}
    )
    init_comparison = init_comparison.merge(
        evaluation[(evaluation.method == "bass") & (evaluation.seed == 17)][["name", "k", "ari"]],
        on=["name", "k"],
    ).rename(columns={"ari": "local_initialization"})
    long_comparison = (
        long_chains[["name", "ari"]]
        .rename(columns={"ari": "long_chain"})
        .merge(
            evaluation[
                (evaluation.method == "bass") & (evaluation.seed == 17) & (evaluation.k == 8)
            ][["name", "ari"]],
            on="name",
        )
        .rename(columns={"ari": "standard_chain"})
    )
    tuning_comparison = (
        annotated[["name", "dataset", "ari"]]
        .rename(columns={"ari": "annotation_tuned"})
        .merge(
            evaluation[(evaluation.seed == 17) & (evaluation.k == 8)].pivot(
                index="name", columns="method", values="ari"
            ),
            on="name",
        )
    )
    tuning_comparison.to_csv(OUT / "summary_annotation_tuning.csv", index=False)
    headline = f"""<p><strong>The modified model does not replace the existing spinal caller in this experiment.</strong>
    At K=8, mean spinal ARI is {tissue_summary.loc[("spinal", 8), "gradient_ring"]:.3f} for the gradient-ring baseline,
    {tissue_summary.loc[("spinal", 8), "bass"]:.3f} for independent BASS-style fits, and
    {tissue_summary.loc[("spinal", 8), "bass_shared"]:.3f} with shared profiles.
    Cortical proxy agreement is {tissue_summary.loc[("cortex", 8), "gradient_ring"]:.3f},
    {tissue_summary.loc[("cortex", 8), "bass"]:.3f}, and {tissue_summary.loc[("cortex", 8), "bass_shared"]:.3f}, respectively.
    Longer chains and gradient initialization do not remove the spinal deficit. A separate spatial-strength
    choice using historical anatomical labels reaches {tuning_comparison.annotation_tuned.mean():.3f} on the later spinal sections (seed 17).</p>
    <p>This supports retaining the gradient-ring baseline for spinal analyses. The cortical result merits further evaluation against independent anatomical labels. The experiment suggests that removing fixed-neighborhood mixing is insufficient when the replacement assumes one composition profile per region and a uniform local continuity prior; it does not establish that richer boundary models cannot work.</p>"""
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fixed-type BASS-style benchmark</title>
    <style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#1b2933}}p,li{{line-height:1.5}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}img{{max-width:100%}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;background:#f3f5f6;padding:12px}}</style></head><body>
    <h1>Fixed-type BASS-style region model</h1>{headline}
    <p>This experiment jointly samples region labels and regional type compositions from individual typed cells. It uses the categorical–Potts hierarchy of <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9351148/">BASS</a>, with fixed existing cell types, geometric degree-normalized edges, empirical-Bayes profile shrinkage, and spatial strength selected by masked-type prediction. This is a custom adaptation, not a benchmark of the published BASS package.</p>
    <p>Every cell contributes its own type once. Missing types have no emission term. Profiles have a Dirichlet prior centered on the overall observed type composition. This provides shrinkage, but does not implement explicit sparse marker selection. Swendsen–Wang cluster updates and single-cell Gibbs updates alternate with profile sampling. Counts from fixed neighborhoods are used only for initialization; the original gradient-ring method remains the comparator.</p>
    <h2>Model selection and scope</h2>{table(selected)}
    <p>Parameters were selected using 20% masked cell types on the historical development/test sections, without anatomical labels. The grid used five spatial strengths and two prior strengths. K=8 is the primary matched-capacity comparison; K=13 is a separate spinal sensitivity analysis. Every configuration used 500 burn-in sweeps plus 100 retained samples spaced five sweeps apart. Evaluation uses 1,000 burn-in sweeps and 200 retained samples. These are exploratory comparisons on previously inspected sections, not new donor-level validation.</p>
    <p>The independent model learns all profiles anew in each section. The shared model learns common profiles jointly across all six spinal sections or four cortical sections within a species/platform, with separate spatial graphs. This is transductive sharing without region labels; it does not test transfer to a wholly new biological region. Shared fits use the independently selected hyperparameters without further tuning. No common orientation, depth coordinate, layer order, or anatomical template is imposed.</p>
    <h2>Annotation agreement</h2>{table(tissue_summary)}{table(summary)}
    <p>Scores average seeds within sections, then weight sections equally. Spinal scores use the same fixed focal neurons with direct polygon labels as previous benchmarks. Cortical references are composition-derived proxies and are not independent anatomical truth. The tables report actual comparison results without selecting the best model by these annotation scores.</p><img src="comparison.png">
    <details><summary>Every section, averaged across three seeds</summary>{table(paired)}</details>
    <h2>Can it predict genuinely masked cell types?</h2>{table(pred_summary)}
    <p>Lower negative log likelihood is better. Validation masks use a separate seed and later sections; hidden types are excluded from initialization, profile estimation, and label likelihood. The type vocabulary is the supplied codebook. The KNN comparator uses observed-only neighbors and five baseline-profile pseudo-counts; its neighborhood size is selected on the pilot sections. Global frequencies are estimated only from observed types. Shared-model masked predictions are seed 17; independent-model numbers average three seeds. Masked cells are interspersed within sections, so this tests local interpolation, not donor-independent extrapolation. The half-chain NLL gap is a mixing diagnostic, not a convergence certificate.</p>{table(knn_choice.set_index("tissue"))}
    <h2>Thin-lamina recovery</h2>{table(region_summary.loc[8])}
    <p>For each reference lamina, best IoU is the largest intersection-over-union with any predicted cluster, evaluated on all directly polygon-labeled neurons. It penalizes both missing cells and incorporating other laminae. This matching is solely for evaluation. All shown results use three seeds and four evaluation sections per species.</p>
    <h2>Spatial maps</h2><p>Representative sections fixed in advance: confirmation1 for each spinal species and fresh1 for each cortex dataset. All maps use K=8 and seed 17. Cluster colors are matched to reference colors for display only; unmatched clusters receive additional colors. Gray reference cells lack a reference label.</p><img src="spinal_maps.png"><img src="cortex_maps.png">
    <h2>Stability and initialization sensitivity</h2>{table(stability.groupby(["dataset", "k"])[["seed_ari", "sample_to_modal_ari"]].mean())}{table(init_comparison.set_index(["name", "k"]))}
    <p>Long-chain check: 5,000 burn-in sweeps plus 500 retained samples spaced five sweeps apart, on one prespecified section per dataset, K=8 and seed 17.</p>{table(long_comparison.set_index("name"))}
    <h2>Does anatomical tuning of spatial strength rescue spinal performance?</h2>{table(strength.set_index("beta"))}{table(tuning_comparison.groupby("dataset")[["gradient_ring", "bass", "annotation_tuned"]].mean())}
    <p>This secondary diagnostic explicitly uses anatomical labels on the six historical development/test spinal sections to choose spatial strength, keeping K=8 and prior strength fixed. The selected beta is then applied to the twelve later spinal sections with seed 17. It is separate from the primary masked-type selection and does not turn these previously inspected sections into fresh validation. It was added after observing the primary model's deficit, to test whether the tuning criterion explained the failure.</p>
    <p>Seed agreement compares inferred partitions, without anatomical labels. Low agreement or sensitivity to gradient-ring initialization means inference uncertainty remains material; posterior probabilities should not be presented as calibrated anatomical confidence. Shared-profile models do not require separate pieces to have the same arrangement, but the benchmark graph uses a distance cap rather than manually verified piece masks. Geometry can therefore still connect nearby tissue fragments or fragment sparse regions.</p>
    <h2>Validation and reproducibility</h2><pre>{html.escape(json.dumps(checks, indent=2))}</pre>
    <p>Sources: benchmarks/bass_fixed_types.py, check_bass_fixed_types.py, run_bass_fixed_types.py, bass_shared_sections.py, bass_sensitivity.py, and report_bass_fixed_types.py. No core package defaults or annotations were changed. No new dependency was installed. Intermediate fits are under /scratch/lamination_station_bass; CSV tables and PNG/PDF figures are beside this report.</p>
    <details><summary>Protocol</summary><pre>{html.escape((OUT / "protocol.json").read_text())}</pre></details>
    <p>Reproduction: export TMPDIR=/scratch/runtime/tmp MPLCONFIGDIR=/scratch/runtime/mpl NUMBA_CACHE_DIR=/scratch/runtime/numba; run the numerical checks, protocol, pilot for each dataset, select, evaluate/validate/gradientinit for each dataset, and bass_shared_sections.py with and without --masked. Run bass_sensitivity.py with each dataset and long; run strength for each spinal dataset, then all select, then annotated_tuning for each spinal dataset. Finally run this report script. versions.json records the environment versions.</p></body></html>"""
    (OUT / "report.html").write_text(document)
    print("ARI", summary.to_string())
    print("PREDICTION", pred_summary.to_string())
    print("STABILITY", stability.groupby(["dataset", "k"]).seed_ari.mean().to_string())
    print("VALIDATED", rescored)


if __name__ == "__main__":
    main()
