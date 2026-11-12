"""Check numerical parity with the experimental winner using identical count schemas."""

import torch  # noqa: F401

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lamination_station.ring_model import FitSettings, fit_ring_mixture
from context_experiments import get_section
from incremental_domains import features, vocabulary
from incremental_models import fit_mixture


records = []
out = Path("/results/lamination_station_regions")
out.mkdir(parents=True, exist_ok=True)
for dataset in [
    "spinal_human",
    "spinal_macaque",
    "spinal_mouse",
    "cortex_human",
    "cortex_merfish",
]:
    split = "confirmation1" if dataset.startswith("spinal") else "fresh1"
    s = get_section(dataset, split)
    columns = vocabulary([s])
    arrays = features(s, columns)
    frames = tuple(pd.DataFrame(x, index=s["ref"].index, columns=columns) for x in arrays)
    old = fit_mixture(arrays, n_components=8, seed=17)
    new = fit_ring_mixture(rings=frames, settings=FitSettings(n_regions=8, seed=17))
    q = new.predict_proba(frames).to_numpy()
    np.testing.assert_allclose(q, old["q"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(new.objective_history, old["history"], atol=1e-7)
    records.append(
        dict(
            dataset=dataset,
            max_membership_error=float(abs(q - old["q"]).max()),
            max_objective_error=float(abs(new.objective_history - old["history"]).max()),
        )
    )
    print(records[-1], flush=True)
(out / "implementation_parity.json").write_text(json.dumps(records, indent=2))
