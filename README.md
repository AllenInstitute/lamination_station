# Lamination Station

A Bayesian modeling toolkit for identifying spatial structures from cell coordinates and categorical cell-type labels. The experimental model variants use PyTorch and Pyro.

Install with Python **3.12 or newer**:

```sh
python -m pip install -e .
```

The original [spinal cord](examples/spc_laminae_example.ipynb) and [cortex](examples/v1_example.ipynb) notebooks illustrate the method, but contain machine-specific data paths and historical postprocessing code. Use the package functions for the corrected behavior below; `structure_pred_nearest` is no longer an output.

## Preparing and fitting data

```python
import torch
import lamination_station as ls

# df has unique row IDs, physical x/y coordinates, cell types, and section IDs.
prepared = ls.pp.calculate_comp_grads(
    df, "x", "y", "cell_type",
    near_k=25, far_k=100, sample_col="section",
)
features, cell_types, gradients, _, _, counts, type_vectors = prepared
out, logits, compositions, losses = ls.model.run_model(
    features, cell_types, counts, gradients, type_vectors,
    NUM_STRUCTURES=10, num_epochs=100, lr_steps=(1e-3,),
    batch_size=256, normalize_sampling=False, seed=17,
)
```

Keep every input in the same row order and retain the count/vector column order returned by preprocessing. `far_k` must be smaller than the number of cells in **every** section. Missing labels, nonfinite values, and empty neighborhoods raise explicit errors. If distance cutoffs leave empty neighborhoods, filter the same rows from all inputs together, or recompute neighborhoods.

Coordinates now use a single isotropic scale by default, preserving physical angles and neighbor order under rotation. `coordinate_scaling="axiswise"` reproduces the earlier metric. Use `sample_col` to prevent neighbors across sections with overlapping coordinate frames. The directional feature remains a heuristic composition-change descriptor.

## Changes that affect existing fits

- Networks are registered inside SVI under distinct names. Earlier versions left parts of the classifier and the entire decoder untrained in several variants.
- Local model **and guide** terms now use full-population minibatch scaling. `normalize_sampling=True` retains balanced sampling but applies inverse-probability weights, so balancing no longer changes the target tissue distribution. Use `False` for ordinary shuffled minibatches.
- Partial and oversized batches train normally. All variants accept an optional `seed`; they no longer silently reset it to zero.
- `identity_model(ignore_cur_type=True)` preserves the observed one-hot target. Its encoder already excludes central identity.
- Graph diffusion converts affinities to reciprocal lengths, handles isolated sources, and rejects unknown structure IDs. Pass `weight_is_distance=True` when supplying actual distances.
- Plot jitter obeys `seed`, `ylim` works independently of `xlim`, and gradient arrows account for the coordinate scale.

Old hyperparameters and raw loss magnitudes will not be directly comparable after the training and scaling corrections. `clear_params=False` reuses parameters only when the variant, dimensions, and vocabulary match; it is not a complete optimizer/checkpoint resume API.

## Graph and statistical interpretation

`po.build_structure_graph(out)` remains a **top-two co-assignment graph**. Its overlap weights are neither anatomical contact probabilities nor calibrated confidence. It now preserves only observed node IDs and stores separate `affinity`, reciprocal `length`, and cell `support` attributes. Physical-contact graph inference is a separate modeling improvement.

`po.compute_diffusion_time` retains its historical name, but computes a spectral coordinate, not rooted diffusion pseudotime. It rejects disconnected graphs; analyze components separately. `n_neighbors` counts other cells, excluding self. PageRank and shortest-path cell scores are display coordinates.

The seven variants retain their existing likelihood/guide designs. Duplicate reconstruction objectives, overlap-derived memberships, and MAP uncertainty limitations require a separate statistical redesign; these correctness fixes do not establish calibrated posteriors or anatomical accuracy.

## Composition mixture and physical adjacency

An alternative to the neural runners is a single multinomial mixture over neighbor counts. It learns component compositions and population weights with MAP EM, checks convergence, and selects among multiple initializations by the penalized likelihood. It avoids reconstructing the same counts through two likelihood heads. The returned object can predict and score new sections with the same cell-type vocabulary:

```python
fitted = ls.composition.fit_composition_mixture(
    counts, n_components=8, n_init=5, seed=17,
)
memberships = fitted.predict_proba(counts)
prepared_cells = features.copy()
prepared_cells["structure"] = memberships.to_numpy().argmax(axis=1)

# Here x/y and the distance cap are in micrometers.
contact_graph = ls.spatial_graph.build_spatial_structure_graph(
    prepared_cells,
    x_col="x", y_col="y", structure_col="structure", sample_col="section",
    max_distance=50.0, n_neighbors=6,
)
```

Choose the distance cap for the measurement scale and tissue density. This graph aggregates **mutual physical neighbors**, preserves disconnected structures, and records cell-edge counts and supporting sections. Its degree-normalized affinity is a descriptive proximity measure, not a membrane-contact or edge probability. It uses hard component assignments; assess sensitivity to assignments, spatial scale, and sampling density.

For new sections, supply the same count columns in the same order. Decide the taxonomy vocabulary before fitting and include zero-filled columns for known types absent from the training section. `fitted.score_samples(new_counts)` returns conditional multinomial log PMFs, while `fitted.converged` and `objective_history` expose optimization diagnostics.

Component memberships represent uncertainty about discrete composition classes, not physical mixing fractions. Overlapping neighborhoods reuse observations, so their product remains a composite likelihood; memberships are not calibrated anatomical confidence. The component count and neighborhood size still need sensitivity checks. Anatomically separate regions with identical composition remain indistinguishable without additional spatial modeling.

## Regression checks

```sh
python -m pip install -e '.[dev]'
python -m ruff check src test
python -m mypy --strict src/lamination_station test
python -c 'import torch; import pytest; raise SystemExit(pytest.main())'
```

Tests cover all seven variants, actual network weight updates, population-gradient equivalence for uniform and balanced minibatches, count families, partial batches, invalid inputs, geometric invariance, and graph edge cases. Our code is checked strictly; Pyro's dynamically generated third-party APIs are treated as untyped.

Written by [mtvector](https://github.com/mtvector), using [Pyro](https://pyro.ai/) and loosely based on an early concept for [ANTIPODE](https://github.com/mtvector/scANTIPODE).
