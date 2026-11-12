"""Audit all saved predictions and render the context/domain experiments report."""

import torch  # noqa: F401

import hashlib
import html
import importlib.metadata
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from context_experiments import OUT, ROOT, get_section, tag

SOURCES = Path(__file__).resolve().parent
labels = {
    "ring": "Original gradient ring mixture",
    "kmeans": "Selected broader/weighted context",
    "patch": "Spatial patches then clustering",
    "molecular": "Finer cell-type contexts",
    "tangent": "Local Jacobian / tangent contexts",
    "density": "Full-density cortical contexts",
    "transfer": "Supervised ExtraTrees transfer",
    "svm": "Supervised RBF-SVM transfer",
    "forest_spatial": "Supervised ExtraTrees + spatial refinement",
}


def spec_from_row(row):
    keys = ["method", "kind"]
    if row.method in ["ring", "kmeans", "patch", "molecular", "tangent", "density"]:
        keys.append("k")
    if row.method == "patch":
        keys.append("patches")
    if row.method == "molecular":
        keys.append("weight")
    if row.method == "tangent":
        keys.append("ratio")
    if row.method == "transfer":
        keys.append("leaf")
    if row.method == "svm":
        keys.append("c")
    if row.method == "forest_spatial":
        keys.append("strength")
    return {k: int(row[k]) if k in ["k", "patches", "ratio", "leaf"] else row[k] for k in keys}


def prediction(row):
    spec = spec_from_row(row)
    prefix = (
        "transfer"
        if row.method == "transfer"
        else "structured"
        if row.method in ["svm", "forest_spatial"]
        else "prediction"
    )
    path = ROOT / f"{prefix}_{row['name']}_{tag(spec)}_seed{int(row.seed)}.pkl"
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    return obj if isinstance(obj, dict) else dict(q=obj)


def section_for_row(row, sections):
    name = row["name"]
    if name not in sections:
        sections[name] = get_section(row.dataset, name.removeprefix(row.dataset + "_"))
    return sections[name]


def main():
    expected = {
        "screen": 190,
        "molecular_screen": 36,
        "geometry_screen": 20,
        "density_screen": 8,
        "transfer_screen": 60,
        "replication": 120,
        "molecular_replication": 18,
        "geometry_replication": 30,
        "density_replication": 12,
        "transfer_replication": 30,
        "confirmation": 72,
        "molecular_confirmation": 18,
        "geometry_confirmation": 18,
        "transfer_confirmation": 18,
    }
    tables = []
    for name, n in expected.items():
        f = pd.read_csv(OUT / (name + ".csv"))
        assert len(f) == n, (name, len(f), n)
        f["source_table"] = name
        tables.append(f)
    all_scores = pd.concat(tables, ignore_index=True)
    sections = {}
    for _, row in all_scores.iterrows():
        s = section_for_row(row, sections)
        obj = prediction(row)
        q = obj["q"]
        assert len(q) == len(s["ref"]) and np.isfinite(q).all() and (q >= 0).all()
        np.testing.assert_allclose(q.sum(1), 1, atol=1e-7)
        meta = s["meta"].dropna(subset=["label"])
        ix = s["ref"].index.get_indexer(meta.index)
        ari = adjusted_rand_score(meta.label, q.argmax(1)[ix])
        assert abs(ari - row.ari) < 1e-10
        if "training" in obj:
            assert row["name"] not in obj["training"]
            assert all(t.endswith(("_development", "_test")) for t in obj["training"])
    frozen = json.loads((OUT / "spinal_frozen_before_confirmation.json").read_text())
    assert json.loads((OUT / "selection.json").read_text())["spinal"] == frozen["main"]
    assert (
        json.loads((OUT / "transfer_selection.json").read_text())["spinal"] == frozen["structured"]
    )
    assert json.loads((OUT / "molecular_selection.json").read_text()) == frozen["molecular"]
    assert (
        json.loads((OUT / "geometry_selection.json").read_text())["spinal"] == frozen["geometry"]
    )
    audit = dict(
        predictions_rescored=len(all_scores),
        all_ari_match=True,
        all_probabilities_valid=True,
        supervised_training_target_disjoint=True,
        spinal_selection_unchanged=True,
        frozen_sha256=hashlib.sha256(
            (OUT / "spinal_frozen_before_confirmation.json").read_bytes()
        ).hexdigest(),
        versions={
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "scipy", "pandas", "scikit-learn", "anndata", "h5py"]
        },
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    all_scores.to_csv(OUT / "all_scores.csv", index=False)
    later = all_scores[all_scores.stage != "screen"].copy()
    per_section = later.groupby(
        ["stage", "tissue", "dataset", "name", "method"], as_index=False
    ).agg(
        ari=("ari", "mean"),
        seed_sd=("ari", "std"),
        boundary_f1=("boundary_f1", "mean"),
        macro_f1=("macro_f1", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        known_class_fraction=("known_class_fraction", "mean"),
    )
    per_section.to_csv(OUT / "per_section.csv", index=False)
    summary = per_section.groupby(["stage", "tissue", "method"], as_index=False).agg(
        ari=("ari", "mean"), boundary_f1=("boundary_f1", "mean"), n_sections=("name", "nunique")
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    summary["cohort"] = summary.stage + " / " + summary.tissue
    comparison = (
        summary.pivot(index="method", columns="cohort", values="ari")
        .reindex(labels)
        .dropna(how="all")
    )
    comparison.index = [labels[i] for i in comparison.index]
    confirmed = per_section[per_section.stage == "confirmation"]
    paired = confirmed.pivot(index=["dataset", "name"], columns="method", values="ari")
    gains = paired.drop(columns="ring").sub(paired.ring, axis=0)
    gains.to_csv(OUT / "confirmation_paired_gains.csv")
    gains_summary = pd.DataFrame(
        {
            "mean_gain": gains.mean(),
            "sections_improved": (gains > 0).sum(),
            "worst_section_gain": gains.min(),
            "best_section_gain": gains.max(),
        }
    )
    gains_summary.to_csv(OUT / "confirmation_gains_summary.csv")
    species = confirmed.groupby(["dataset", "method"]).ari.mean().unstack()
    species.to_csv(OUT / "confirmation_species.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, (stage, tissue, title) in zip(
        axes,
        [
            ("confirmation", "spinal", "Six previously unscored spinal sections"),
            ("replication", "cortex", "Four historical V1 sections (proxy labels)"),
        ],
        strict=True,
    ):
        sub = per_section[(per_section.stage == stage) & (per_section.tissue == tissue)]
        methods = [m for m in labels if m in sub.method.unique()]
        for y, m in enumerate(methods):
            vals = sub[sub.method == m].ari.to_numpy()
            ax.scatter(vals, np.full(len(vals), y), alpha=0.55, s=26)
            ax.plot([vals.mean()], [y], "k|", markersize=18, markeredgewidth=2)
        ax.set_yticks(range(len(methods)), [labels[m] for m in methods], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("ARI: dots = sections; black mark = mean")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=160)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)

    plots = []
    export = OUT / "predictions"
    export.mkdir(exist_ok=True)
    for name in confirmed.name.unique():
        rows = later[(later.name == name) & (later.seed == 17)]
        first = rows.iloc[0]
        s = section_for_row(first, sections)
        meta = s["meta"].dropna(subset=["label"])
        ix = s["ref"].index.get_indexer(meta.index)
        categories = sorted(meta.label.astype(str).unique())
        truth = pd.Categorical(meta.label.astype(str), categories=categories).codes
        cmap = plt.get_cmap("tab20")
        methods = ["ring", "kmeans", "tangent", "transfer", "svm"]
        methods = [m for m in methods if m in rows.method.to_numpy()]
        fig, axes = plt.subplots(
            1, len(methods) + 1, figsize=(3.5 * (len(methods) + 1), 4), layout="constrained"
        )
        xy = s["ref"][["x", "y"]].to_numpy()
        axes[0].scatter(xy[ix, 0], xy[ix, 1], c=[cmap(t) for t in truth], s=3)
        axes[0].set_title("Direct polygon reference", fontsize=9)
        exported = s["ref"][["x", "y"]].copy()
        exported["reference_polygon"] = s["meta"].label.reindex(exported.index)
        for ax, method in zip(axes[1:], methods, strict=True):
            row = rows[rows.method == method].iloc[0]
            obj = prediction(row)
            pred = obj["q"].argmax(1)
            if "classes" in obj:
                actual = obj["classes"][pred]
                lookup = {c: i for i, c in enumerate(categories)}
                codes = np.array([lookup.get(v, -1) for v in actual])
                exported[method] = actual
            else:
                tab = np.zeros((len(categories), obj["q"].shape[1]), int)
                np.add.at(tab, (truth, pred[ix]), 1)
                r, c = linear_sum_assignment(-tab)
                mapping = dict(zip(c, r, strict=True))
                codes = np.array([mapping.get(v, -1) for v in pred])
                exported[method] = pred
            colors = [cmap(v) if v >= 0 else (0.65, 0.65, 0.65, 1) for v in codes]
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=2)
            ax.set_title(f"{labels[method]}\nARI={row.ari:.3f}", fontsize=8)
        for ax in axes:
            ax.set_aspect("equal")
            margin = np.maximum(np.ptp(xy, axis=0) * 0.03, 1e-9)
            ax.set_xlim(xy[:, 0].min() - margin[0], xy[:, 0].max() + margin[0])
            ax.set_ylim(xy[:, 1].min() - margin[1], xy[:, 1].max() + margin[1])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(name, fontsize=11)
        fig.legend(
            handles=[Patch(color=cmap(i), label=c) for i, c in enumerate(categories)],
            loc="outside lower center",
            ncol=min(len(categories), 14),
            fontsize=8,
        )
        figure = name + ".png"
        fig.savefig(OUT / figure, dpi=160)
        fig.savefig(OUT / (name + ".pdf"))
        plt.close(fig)
        plots.append(figure)
        exported.index.name = "cell_id"
        exported.to_parquet(export / (name + ".parquet"))
    class_records = []
    for dataset in confirmed.dataset.unique():
        yy, pp = [], []
        for _, row in later[
            (later.stage == "confirmation")
            & (later.dataset == dataset)
            & (later.method == "svm")
            & (later.seed == 17)
        ].iterrows():
            s = section_for_row(row, sections)
            obj = prediction(row)
            meta = s["meta"].dropna(subset=["label"])
            ix = s["ref"].index.get_indexer(meta.index)
            yy.extend(meta.label.astype(str))
            pp.extend(obj["classes"][obj["q"].argmax(1)][ix])
        categories = sorted(set(yy) | set(pp))
        cm = confusion_matrix(yy, pp, labels=categories)
        pd.DataFrame(cm, index=categories, columns=categories).to_csv(
            OUT / (dataset + "_transfer_confusion.csv")
        )
        for i, category in enumerate(categories):
            precision = cm[i, i] / max(cm[:, i].sum(), 1)
            recall = cm[i, i] / max(cm[i].sum(), 1)
            class_records.append(
                dict(
                    dataset=dataset,
                    lamina=category,
                    support=int(cm[i].sum()),
                    precision=precision,
                    recall=recall,
                    f1=2 * precision * recall / max(precision + recall, 1e-12),
                )
            )
    class_metrics = pd.DataFrame(class_records)
    class_metrics.to_csv(OUT / "svm_confirmation_per_lamina.csv", index=False)
    semantic_summary = (
        confirmed[confirmed.method.isin(["svm", "transfer"])]
        .groupby(["dataset", "method"])[["macro_f1", "balanced_accuracy", "known_class_fraction"]]
        .mean()
    )

    screening = (
        all_scores[all_scores.stage == "screen"]
        .groupby(["tissue", "candidate"])
        .ari.mean()
        .reset_index()
    )
    screening.to_csv(OUT / "screening_summary.csv", index=False)
    final_text = (
        (OUT / "interpretation.html").read_text() if (OUT / "interpretation.html").exists() else ""
    )
    protocol = "".join(
        f"<h3>{html.escape(p.name)}</h3><pre>{html.escape(p.read_text())}</pre>"
        for p in sorted(OUT.glob("*protocol.json"))
    )
    selected = "".join(
        f"<h3>{html.escape(p.name)}</h3><pre>{html.escape(p.read_text())}</pre>"
        for p in sorted(OUT.glob("*selection.json"))
    )
    checks = json.loads((OUT / "numerical_checks.json").read_text())
    content = f"""<!doctype html><html><head><meta charset="utf-8"><title>Lamination station: larger changes to domain modeling</title>
    <style>body{{font:16px system-ui;max-width:1280px;margin:40px auto;padding:0 24px;color:#18212c}}h1,h2{{color:#173c50}}p,li{{line-height:1.55}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:16px}}.note{{background:#fff3d9;padding:16px;border-left:4px solid #cc9a30}}.scroll{{overflow:auto}}</style></head><body>
    <h1>Testing larger changes to region and lamina calling</h1>
    <p>Experimental analysis, 9 November 2026. Original gradient-adjusted features are retained. Package defaults are unchanged.</p>
    {final_text}
    <h2>What was tested, and why</h2>
    <ul><li><a href="https://www.nature.com/articles/s41467-023-44367-9">MENDER</a> motivates context at multiple ranges: add8–256-neighbor compositions, including frequency-normalized residuals, to the existing gradient features.</li>
    <li><a href="https://doi.org/10.1016/j.crmeth.2024.100838">SCGP</a> motivates treating spatial structure explicitly. Our different prototype forms adjacency-constrained Ward patches, then clusters their composition profiles so disconnected patches may share an identity.</li>
    <li><a href="https://www.nature.com/articles/s41467-025-64259-4">CellTransformer</a> uses neighborhood expression and cell-type information to learn region representations. We tested cheaper information-content alternatives: finer cell types, averaged expression, and within-cell-type expression residuals. These are not CellTransformer runs.</li>
    <li>A signed composition Jacobian estimates a local normal; anisotropic neighborhoods follow the local tangent without assuming one global cortical axis. This augments the existing heuristic gradient.</li>
    <li>Complete cortical cell pools test whether10,000-cell subsampling blurred thin layers. Scored cells are unchanged.</li>
    <li>Explicitly supervised transfer learns named laminae from annotated training sections, with ExtraTrees, RBF-SVM, or spatially refined probabilities. This changes the task from discovery to anatomical annotation.</li></ul>
    <p>These are custom experimental adaptations, not new performance claims about the cited packages. Inputs contain no absolute position, slide orientation, or anatomical region label in the unsupervised arms. Supervised arms use training-section labels only. Physical components are inferred from proximity; nearby pieces can still be connected, and original gradient features retain their original neighborhood behavior.</p>
    <h2>Evaluation safeguards</h2>
    <p>Parameters were selected using the initial development/test sections only, separately by tissue. Historical fresh1/fresh2 sections are exploratory replication because they were already evaluated in earlier rounds. Six additional spinal sections—the fifth and sixth eligible sections per species—were reserved for confirmation. Spinal selections were recorded before confirmation and checked unchanged afterward. Every later supervised fit uses only the two initial training sections of the same species/platform.</p>
    <p class="note">Spinal evaluation uses direct polygon annotations only. V1 uses composition-derived proxy labels, not independent anatomical ground truth. Transfer has access to training annotations and is not an unsupervised-method comparison. Sections are not guaranteed independent donors; the mouse sections come from the same named experiment. Three seeds measure algorithmic variability, not biological replication.</p>
    <h2>Agreement with reference regions</h2><p>Mean ARI, higher is better. Each section first averages three seeds, then sections receive equal weight. A dash means that arm was not tested on that tissue. Context clustering can use12 clusters if selected; the original mixture uses8. Supervised models predict the anatomical classes present in their training data.</p>
    <div class="scroll">{comparison.to_html(float_format=lambda x: f"{x:.3f}", na_rep="—")}</div><p><img src="comparison.png"></p>
    <h2>Fresh spinal confirmation: gains relative to the original mixture</h2>
    <div class="scroll">{gains_summary.rename(index=labels).to_html(float_format=lambda x: f"{x:.3f}")}</div>
    <h3>By species</h3><div class="scroll">{species.rename(columns=labels).to_html(float_format=lambda x: f"{x:.3f}")}</div>
    <h3>By section</h3><div class="scroll">{paired.rename(columns=labels).to_html(float_format=lambda x: f"{x:.3f}")}</div>
    <h2>Named-lamina accuracy and remaining failures</h2><p>All confirmation reference classes were represented in training. Higher overall ARI does not imply every lamina is reliably recovered. Macro F1 and balanced accuracy give rare laminae equal weight. The per-lamina table pools both confirmation sections of each species for seed17.</p>
    <div class="scroll">{semantic_summary.to_html(float_format=lambda x: f"{x:.3f}")}</div>
    <details><summary>Per-lamina precision and recall for supervised SVM</summary>{class_metrics.to_html(index=False, float_format=lambda x: f"{x:.3f}")}</details>
    <h2>Spatial maps</h2><p>Seed17. Reference colors for unsupervised clusters are assigned by one-to-one overlap matching solely for display; unmatched clusters are gray. Supervised calls retain their learned anatomical names. Map color matching is never used for training or ARI. Reference panels show evaluated polygon-labeled cells; predictions show all benchmark reference neurons.</p>
    {"".join('<p><img src="' + p + '"></p>' for p in plots)}
    <h2>Validation and deliverables</h2><p>Independently re-scored all {len(all_scores)} saved predictions: every ARI matched; probability rows were finite, nonnegative and normalized; supervised training and target sections were disjoint; frozen spinal selections were unchanged.</p>
    <ul>{"".join("<li>" + html.escape(c) + "</li>" for c in checks)}</ul>
    <p>Tables: <a href="summary.csv">cohort summary</a>, <a href="per_section.csv">per-section metrics</a>, <a href="all_scores.csv">all model fits</a>, <a href="screening_summary.csv">screening results</a>, <a href="confirmation_paired_gains.csv">paired gains</a>. Seed17 confirmation predictions with cell IDs and coordinates are in predictions/*.parquet; confusion counts for supervised spinal calls are in *_transfer_confusion.csv. Figures are available as both PNG and PDF.</p>
    <details><summary>Complete screening results (not confirmation estimates)</summary>{screening.to_html(index=False, float_format=lambda x: f"{x:.3f}")}</details>
    <details><summary>Selected configurations</summary>{selected}</details>
    <details><summary>Protocols and model details</summary>{protocol}</details>
    <details><summary>Audit and versions</summary><pre>{html.escape(json.dumps(audit, indent=2))}</pre></details>
    <h2>Reproduction</h2><p>Sources are in code/lamination_station/benchmarks/context_*.py, check_context_experiments.py, and report_context.py. No dependencies were added. Use the existing capsule environment and set TMPDIR=/scratch/runtime/tmp and MPLCONFIGDIR=/scratch/runtime/mpl. Run context_experiments.py prepare; context_expression.py extract; each model script with screen. Run context_experiments.py freeze before any confirmation scoring. Then run each model script with replication and confirmation where supported, followed by check_context_experiments.py and report_context.py. The main screening command writes selection.json; the frozen audit file records and verifies the spinal selections used for this run. Cached inputs and fits are under /scratch/lamination_station_context. Core package behavior was not changed.</p>
    </body></html>"""
    (OUT / "report.html").write_text(content)
    print(summary.to_string(index=False))
    print("AUDIT", json.dumps(audit))


if __name__ == "__main__":
    main()
