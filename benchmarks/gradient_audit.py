"""Paired ablation of the existing gradient-based neighborhood adjustment."""

import torch

import inspect
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/root/capsule/code/lamination_station/src")
sys.path.insert(0, "/scratch/lamination_station_investigation")
from lamination_station.preprocessing import calculate_comp_grads
from prototypes import fit_rings

torch.set_num_threads(2)
threadpool_limits(2)
source = Path("/scratch/lamination_station_investigation")
cache = Path("/scratch/lamination_station_gradients")
out = Path("/results/lamination_station_gradients")
cache.mkdir(exist_ok=True)
out.mkdir(exist_ok=True)
shutil.copy2(__file__, out / "gradient_audit.py")
shutil.copy2(source / "prototypes.py", out / "prototypes.py")
shutil.copy2(inspect.getfile(calculate_comp_grads), out / "preprocessing_audited.py")


def descriptor(displacement, difference, eps=0.01):
    """The exact pre-clipping gradient-stage formula, with supplied compositions."""
    distance = np.linalg.norm(displacement, axis=1) + eps
    weight = np.linalg.norm(difference, axis=1) / distance
    return ((weight / distance)[:, None] * displacement).sum(0) / (weight.sum() + eps)


# A linear two-type composition ramp: p_A=.5+.1*x, p_B=.5-.1*x.
offsets = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
delta = np.column_stack([0.1 * offsets[:, 0], -0.1 * offsets[:, 0]])
ramp = descriptor(offsets, delta)
assert np.linalg.norm(ramp) < 1e-14
synthetic = {
    "symmetric_linear_ramp_descriptor": ramp.tolist(),
    "true_composition_jacobian": [[0.1, 0.0], [-0.1, 0.0]],
    "one_sided_same_ramp_descriptor": descriptor(offsets[[0, 2, 3]], delta[[0, 2, 3]]).tolist(),
    "one_sided_tenfold_weaker_ramp_descriptor": descriptor(
        offsets[[0, 2, 3]], delta[[0, 2, 3]] / 10
    ).tolist(),
    "interpretation": "Gradient stage supplied exact compositions; not a simulation of cell-type sampling.",
}
(out / "analytic_checks.json").write_text(json.dumps(synthetic, indent=2))

records = []
checks = []
for path in sorted(source.glob("*_fresh[12].pkl")):
    name = path.stem
    ref, adjusted, meta = pickle.load(path.open("rb"))
    ix = ref.index.get_indexer(meta.index)
    assert (ix >= 0).all()
    valid = meta.label.notna().to_numpy()
    checkpoint = cache / f"{name}_counts.pkl"
    if checkpoint.exists():
        plain, check = pickle.load(checkpoint.open("rb"))
    else:
        # Independent physical kNN counts, excluding self by identity.
        xy = ref[["x", "y"]].to_numpy(dtype=float)
        neighbors = NearestNeighbors(n_neighbors=61).fit(xy).kneighbors(xy, return_distance=False)
        neighbors = np.array([row[row != i][:60] for i, row in enumerate(neighbors)])
        onehot = np.eye(len(ref.type.cat.categories))[ref.type.cat.codes.to_numpy()]
        plain = {
            k: pd.DataFrame(
                onehot[neighbors[:, :k]].sum(1), index=ref.index, columns=adjusted[k].columns
            )
            for k in [15, 30, 60]
        }
        zero = calculate_comp_grads(ref, "x", "y", "type", near_k=60, far_k=90, elev_weight=0)
        np.testing.assert_array_equal(plain[60], zero[5])
        for k in [15, 30, 60]:
            current = calculate_comp_grads(ref, "x", "y", "type", near_k=k, far_k=90)
            pd.testing.assert_frame_equal(current[5], adjusted[k])
        # Verify the isolated mathematical formula against actual public output.
        scaled = (xy - xy.mean(0)) / np.sqrt(np.mean((xy - xy.mean(0)) ** 2))
        nearest = (
            NearestNeighbors(n_neighbors=91).fit(scaled).kneighbors(scaled, return_distance=False)
        )
        nearest = np.array([row[row != i][:90] for i, row in enumerate(nearest)])
        raw = np.array(
            [
                descriptor(scaled[row] - scaled[i], current[4][row] - current[4][i])
                for i, row in enumerate(nearest)
            ]
        )
        norm = np.linalg.norm(raw, axis=1)
        clip = np.quantile(norm, 0.9)
        raw *= np.minimum(1, np.divide(clip, norm, out=np.ones_like(norm), where=norm > 0))[
            :, None
        ]
        np.testing.assert_allclose(raw, current[2], atol=1e-12)
        check = {
            "name": name,
            "n_reference": len(ref),
            "n_evaluation": int(valid.sum()),
            "cached_adjusted_counts_equal_current_code": True,
            "zero_weight_counts_equal_independent_knn": True,
            "gradient_formula_matches_public_output": True,
            "fraction_cells_counts_changed_k60": float(
                (plain[60].to_numpy() != adjusted[60].to_numpy()).any(1).mean()
            ),
            "mean_count_l1_change_k60": float(
                np.abs(plain[60].to_numpy() - adjusted[60].to_numpy()).sum(1).mean()
            ),
        }
        pickle.dump((plain, check), checkpoint.open("wb"))
    checks.append(check)
    for adjustment, counts in [("off", plain), ("on", adjusted)]:
        rings = [
            counts[15].to_numpy(),
            (counts[30] - counts[15]).to_numpy(),
            (counts[60] - counts[30]).to_numpy(),
        ]
        for method in ["hellinger_kmeans", "ring_multinomial"]:
            for seed in [17, 29, 43]:
                target = cache / f"{name}_{method}_{adjustment}_{seed}.pkl"
                if target.exists():
                    result = pickle.load(target.open("rb"))
                else:
                    if method == "hellinger_kmeans":
                        x = counts[60].to_numpy()
                        labels = KMeans(n_clusters=8, n_init=20, random_state=seed).fit_predict(
                            np.sqrt(x / x.sum(1, keepdims=True))
                        )
                    else:
                        q, _, _ = fit_rings(rings, n_components=8, seed=seed)
                        labels = q.argmax(1)
                    record = {
                        "name": name,
                        "tissue": "spinal" if name.startswith("spinal") else "cortex_proxy",
                        "method": method,
                        "adjustment": adjustment,
                        "seed": seed,
                        "ari": adjusted_rand_score(meta.label[valid], labels[ix][valid]),
                    }
                    result = {"record": record, "labels": labels, "index": ref.index}
                    pickle.dump(result, target.open("wb"))
                records.append(result["record"])
    pd.DataFrame(records).to_csv(out / "runs.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
    print(
        name,
        pd.DataFrame(records)
        .query("name == @name")
        .groupby(["method", "adjustment"])
        .ari.mean()
        .to_dict(),
        flush=True,
    )

runs = pd.DataFrame(records)
sections = (
    runs.groupby(["name", "tissue", "method", "adjustment"])
    .ari.mean()
    .unstack("adjustment")
    .reset_index()
)
sections["on_minus_off"] = sections["on"] - sections["off"]
sections.to_csv(out / "section_means.csv", index=False)
summary = sections.groupby(["tissue", "method"])[["off", "on", "on_minus_off"]].mean()
summary.to_csv(out / "summary.csv")
print(summary, flush=True)
(out / "protocol.json").write_text(
    json.dumps(
        {
            "primary_contrast": "elev_weight=0 versus existing elev_weight=1; all other clustering settings fixed",
            "sections": "Previously evaluated six spinal and four cortex sections; this is a paired diagnostic, not a new independent validation set",
            "fitting": "All cached reference cells per section; up to 10000; same fixed evaluation cells",
            "settings": "K8, seeds17/29/43; Hellinger kmeans near60 n_init20; ring mixture ranks1-15,16-30,31-60 n_init5 max_iter300; far90",
            "selection": "No tuning for this ablation. Existing settings were selected on earlier spinal development sections with adjustment enabled, potentially favoring enabled adjustment.",
            "truth": "Spinal direct polygon annotations; cortex composition-derived proxies, not independent anatomy",
            "scope": "Tests effect of gradients through neighbor selection; does not isolate their extra neural-network input channels",
        },
        indent=2,
    )
)
