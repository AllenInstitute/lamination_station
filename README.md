# Lamination Station

A toolkit for calling regions from spatial cell coordinates and categorical cell-type labels. The recommended analysis workflow is the **gradient-ring composition mixture**; the original neural variants remain available as experimental models.

Install with Python **3.12 or newer**:

```sh
python -m pip install -e .
```

## Recommended region-calling workflow

```python
import torch
import lamination_station as ls

# One row per input cell, with a unique cell ID as the dataframe index.
# Use physical coordinates and nonempty string cell-type labels.
prepared = ls.regions.prepare_gradient_rings(
    cells=df, x_col="x", y_col="y", celltype_col="cell_type",
    sample_col="section",  # explicitly use None for a single coordinate frame
    # piece_col="piece",   # isolate separately identified pieces within a section
)
result = ls.regions.fit_regions(features=prepared, n_regions=16, seed=17)
called_cells = df.join(result.assignments.add_prefix("ls_"))
print(result.region_summary)
print(result.model.converged)
result.save("/results/regions_16")  # creates a new directory; never overwrites one
```

**Resolution is `n_regions`.** Eight regions reproduces the benchmark's default capacity; use a larger count, such as 12 or 16, to overcluster for subsequent review. The model may leave components without hard assignments, so inspect the requested versus occupied count and small-region support. Region IDs are arbitrary component labels, not anatomical lamina names.

By default, neighborhoods use gradient-adjusted ranks **1–15, 16–30, and 31–60**, drawn from 90 candidate neighbors. These are disjoint rings, each with its own learned cell-type composition. The MAP mixture uses five reproducible initializations and retains the highest-objective fit, with diagnostics for every start. Neighborhood scales are independent of clustering resolution:

```python
settings = ls.regions.GradientSettings(ring_ends=(15, 30, 60), far_k=90)
```

Sample and optional piece IDs prevent cross-group neighbors even when coordinate frames overlap. Every supplied group must have more than `far_k` cells; small pieces raise an error rather than silently changing scale or dropping cells. Within a pooled fit, all cells have equal weight and share component profiles and mixture weights. To fit sections independently, call the workflow separately for each section. Piece IDs isolate neighborhoods but do not force different region identities in disconnected pieces.

### Compare resolution and reproducibility

Reuse the prepared features to compare component counts and seeds:

```python
scan = ls.regions.fit_resolutions(
    features=prepared, n_regions=(8, 12, 16), seeds=(17, 29, 43),
)
print(scan.summary)  # seed agreement, convergence, occupied regions, minimum support
fine = scan.best(n_regions=16)  # best objective among seeds at this chosen resolution
```

The scan does **not** choose a region count by maximizing training likelihood. Compare spatial maps, region compositions, small-region support, and seed stability. Fits at different resolutions are not a nested hierarchy: increasing the count can move boundaries as well as split components. Adjacent gradient neighborhood scales can be checked by preparing a second set of features explicitly.

### Review and merge overclustered regions

```python
# Example only: replace these IDs after inspecting your own region maps/profiles.
reviewed = fine.merge({3: 1, 7: 1})
reviewed.save("/results/regions_16_reviewed")
```

This maps fine regions 3 and 7 into region 1; unmapped IDs stay unchanged. Mappings are applied once, not transitively. The original fine label is retained for every cell. Coarse memberships are summed, but hard assignments are mapped directly from the fine labels, so merging cannot move a cell across an existing fine-region boundary. The complete mapping and source-fit fingerprint are exported for review and reproducibility.

### Number regions using spatial adjacency

After fitting, infer a useful display order directly from physical contacts:

```python
ordered = result.order_laminae()  # automatic endpoints, no anchors or layer labels
numbered_cells = result.assignments.join(ordered.annotate(result.assignments.region))
ordered.save("/scratch/lamination/region_order")  # new directory
```

**Harmonic ordering is the default and always infers its endpoints.** In each
connected region graph it selects the most separated pair by effective
resistance, using contact weights as conductances, and assigns any additional
terminal branches to their nearer pole. It then computes a harmonic coordinate
on the full graph. You do not need to recognize branching layers or identify
superficial/deep regions. Parallel V1/V2 L4/L5 paths and their reconvergence stay
in the graph; it is never reduced to a spanning tree.

`ordered.regions` contains display numbers, graph coordinates, inferred endpoint
roles, numerical tie groups, and diagnostics. `ordered.edges` retains all original
contacts, and `ordered.precedes` records increasing contact paths; parallel arms
can be incomparable even though display numbers are unique. No anatomical
homology or physical depth is inferred. Different branch lengths/weights can
place analogous layers at different coordinates. Endpoint-pair ties and
ambiguous terminal choices are flagged, but do not block numbering. Isolated
regions receive display numbers with missing coordinates. Weak peripheral
regions can attract an inferred endpoint; inspect neighborhood/sample sensitivity.

Direction is a reproducible convention based on region IDs unless you optionally
provide `reference_order`, a Series indexed by region ID. That reference can
only flip direction, break coordinate ties, and place disconnected components;
it does not select endpoints or branches. An existing anatomical order can thus
keep plots facing the familiar direction without guiding endpoint inference.
Legacy Fiedler ordering remains available as `method=ls.ordering.OrderMethod.SPECTRAL`.
No mode requires supervised anchors.

Sample/piece isolation follows feature preparation. Contacts use mutual kNN
(default 12), capped at 2.5 times the smaller endpoint's median first-six-neighbor
spacing. Counts are normalized by region incident degrees, including within-region
contacts, and averaged over available sections within units and over units.
Use `unit_col="donor"` to balance sections within donors. Pairs require `min_cells=20`
per region per section. Exact-distance ties can depend on cell row order; known
piece IDs guarantee separation that a distance cap alone cannot provide for tiny
fragments. Optional `min_weight` excludes weak edges from ordering while retaining
them in the exported edge table. Original assignments and probabilities stay intact.

For already computed graphs, use `ls.ordering.order_regions(adjacency=...)`.
For a target subset of a larger tissue section, use
`ls.contacts.local_cell_contacts(cells=full_context, ..., target_ids=target.index)`
and `ls.contacts.aggregate_region_contacts(...)` first. This prevents artificial
contacts across excluded tissue. These functions expose raw per-section counts,
normalized graphs, and sample balancing for sensitivity analysis and caching.

Run the [branching example](examples/order_branched_laminae.py) to export a graph
with parallel V1/V2 paths and a vector figure:

```sh
python examples/order_branched_laminae.py /scratch/lamination/branch_demo
```

The underlying methods are spectral seriation
([Atkins et al., 1998](https://epubs.siam.org/doi/abs/10.1137/S0097539795285771))
effective-resistance distances ([NetworkX documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.resistance_distance.html)),
and the graph Dirichlet/harmonic problem
([Grady, 2006](https://leogrady.net/wp-content/uploads/2017/01/grady2006random.pdf)).

### Cell and model exports

`result.assignments` retains input row IDs and contains `region`, `membership_max`, `membership_margin`, and `membership_entropy`. Memberships measure the mixture model's relative support, **not calibrated anatomical confidence**. Overlapping neighborhoods reuse cells, and anatomically distinct regions can share compositions. No anatomical annotations are read by the fitting code.

Exports include assignments, memberships, per-ring profiles, region sizes, gradients, optimization histories, the fitted JSON model, and a manifest recording scales, vocabulary, seed, versions, and input/fit fingerprints. Known absent types can be retained by supplying `cell_types=(...)` during preparation; types outside that vocabulary raise an error. Prepared inputs are checked for accidental modification before fitting/exporting.

`ls.ring_model.RingMixture.load(".../model.json")` reloads a model without pickle. Its `predict_proba(prepared.rings)` requires the same vocabulary and ring sizes. Use prediction when deliberately reusing existing profiles; refit to discover new region compositions in a new section. Keep the accompanying manifest to reproduce the spatial preprocessing.

The runnable [CSV analysis example](examples/call_regions.py) exposes section/piece columns, neighborhood scales, and region count on the command line.

## Historical neural workflows

The original [spinal cord](examples/spc_laminae_example.ipynb) and [cortex](examples/v1_example.ipynb) notebooks illustrate the method, but contain machine-specific data paths and historical postprocessing code. Use the package functions for the corrected behavior below; `structure_pred_nearest` is no longer an output. The following runner is a historical neural model, not the recommended gradient-ring mixture.

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
