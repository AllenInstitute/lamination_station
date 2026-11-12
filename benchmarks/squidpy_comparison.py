"""Compare Squidpy niches against cached tissue benchmarks; run development, validation, historical."""
import torch
import hashlib
import importlib.metadata
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans

package_src = Path(__file__).resolve().parents[1] / 'src'
if not (package_src/'lamination_station').exists():
    package_src = Path('/code/lamination_station/src')
sys.path.insert(0, str(package_src))
from lamination_station import composition  # noqa: E402
torch.set_num_threads(2)
old = Path('/scratch/lamination_station_benchmark')
previous = Path('/scratch/lamination_station_investigation')
cache = Path('/scratch/lamination_station_squidpy')
out = Path('/results/lamination_station_squidpy')
cache.mkdir(exist_ok=True)
out.mkdir(exist_ok=True)
phase = sys.argv[1]
subset = sys.argv[2] if len(sys.argv) > 2 else ''
resolutions = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]


def fit_niches(ref, *, near, scale, seed, name, focal=None):
    """Run the public Squidpy API with no anatomical labels supplied."""
    target = cache / f'{name}_n{near}_s{int(scale)}_seed{seed}.pkl'
    if target.exists():
        return pickle.load(target.open('rb'))
    start = time.perf_counter()
    a = ad.AnnData(obs=ref[['type', 'section']].copy())
    a.obs['type'] = a.obs['type'].astype('category').cat.remove_unused_categories()
    a.obsm['spatial'] = ref[['x', 'y']].to_numpy()
    assert a.obs.section.nunique() == 1
    sq.gr.spatial_neighbors(a, coord_type='generic', n_neighs=near, set_diag=False)
    graph = a.obsp['spatial_connectivities']
    assert not graph.diagonal().any()
    assert np.all(np.diff(graph.indptr) == near)
    mask = None if focal is None else pd.Series(a.obs_names.isin(focal), index=a.obs_names)
    sq.gr.calculate_niche_neighborhood(
        a, groups='type', resolutions=resolutions, n_neighbors=15,
        scale=scale, distance=1, abs_nhood=False, rng=seed, cluster_mask=mask,
    )
    predictions = {r: a.obs[f'nhood_niche_res={r}'].astype(str).to_numpy() for r in resolutions}
    result = {'predictions': predictions, 'seconds': time.perf_counter()-start,
              'index': ref.index, 'near': near, 'scale': scale, 'seed': seed}
    pickle.dump(result, target.open('wb'))
    return result


def evaluate(labels, *, ref, meta, name, method, seed, seconds, **extra):
    """Score fixed focal cells; labels enter only this evaluation function."""
    ix = ref.index.get_indexer(meta.index)
    assert (ix >= 0).all() and len(labels) == len(ref)
    mask = meta.label.notna().to_numpy()
    predicted = np.asarray(labels)[ix]
    assert not np.isin(predicted, ['not_a_niche', 'nan']).any()
    return dict(name=name, dataset=name.rsplit('_', 1)[0], method=method,
                seed=seed, seconds=seconds, n_reference=len(ref), n_focal=len(meta),
                n_labelled=int(mask.sum()), occupied=len(np.unique(predicted)),
                ari=adjusted_rand_score(meta.label[mask], predicted[mask]) if mask.any() else np.nan,
                ami=adjusted_mutual_info_score(meta.label[mask], predicted[mask]) if mask.any() else np.nan,
                **extra)


if phase == 'development':
    protocol = {
        'primary_selection': 'Highest mean ARI on original three spinal development sections only.',
        'grid': {'physical_neighbors': [15, 30, 60], 'scale': [True, False], 'resolution': resolutions},
        'seeds': [17, 29, 43], 'development_seed': 17,
        'validation': 'Six separate spinal sections and four cortex sections from previous investigation.',
        'fit_scope': 'All cached reference cells, at most 10000; score fixed focal cells, at most 2048.',
        'matched_granularity': 'For selected neighborhood and scaling, choose resolution with cluster count closest to 8 using no labels; tie favors lower resolution.',
        'historical': 'Original seven test sections; restrict Squidpy clustering to historical focal cells while retaining full reference neighborhood composition.',
        'reference_labels': 'Spinal direct polygon labels only. Cortex composition-derived proxies; mouse cortex unlabelled.',
        'limits': 'Transductive section fits, not held-out-donor prediction. Validation sections were evaluated in earlier investigations. Fixed k changes physical footprint after subsampling.',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    records = []
    for path in sorted(old.glob('spinal_*_development.pkl')):
        prep, meta, ref = pickle.load(path.open('rb'))
        for near, scale in itertools.product([15, 30, 60], [True, False]):
            result = fit_niches(ref, near=near, scale=scale, seed=17, name=path.stem)
            for resolution, labels in result['predictions'].items():
                records.append(evaluate(labels, ref=ref, meta=meta, name=path.stem,
                                        method='squidpy', seed=17, seconds=result['seconds'],
                                        near=near, scale=scale, resolution=resolution))
            print(path.stem, near, scale, 'done', flush=True)
            pd.DataFrame(records).to_csv(out / 'development.csv', index=False)
    scores = pd.DataFrame(records).groupby(['near', 'scale', 'resolution']).ari.mean().sort_values(ascending=False)
    near, scale, resolution = scores.index[0]
    choice = dict(near=int(near), scale=bool(scale), resolution=float(resolution), development_ari=float(scores.iloc[0]))
    (out / 'selected.json').write_text(json.dumps(choice, indent=2))
    print(choice, flush=True)

elif phase in ['validation', 'historical']:
    selected = json.loads((out / 'selected.json').read_text())
    records = []
    paths = sorted(previous.glob('*_fresh[12].pkl')) if phase == 'validation' else sorted(old.glob('*_test.pkl'))
    paths = [p for p in paths if p.stem.startswith(subset)]
    for path in paths:
        name = path.stem
        if phase == 'validation':
            ref, counts, meta = pickle.load(path.open('rb'))
        else:
            prep, meta, ref = pickle.load(path.open('rb'))
        ix = ref.index.get_indexer(meta.index)
        export = meta.copy()
        export[['x', 'y']] = ref.loc[meta.index, ['x', 'y']]
        for seed in [17, 29, 43]:
            result = fit_niches(ref, near=selected['near'], scale=selected['scale'], seed=seed,
                                name=name, focal=meta.index if phase == 'historical' else None)
            matched = min(resolutions, key=lambda r: abs(len(np.unique(result['predictions'][r][ix])) - 8))
            for method, resolution in [('squidpy', selected['resolution']), ('squidpy_near8', matched)]:
                labels = result['predictions'][resolution]
                records.append(evaluate(labels, ref=ref, meta=meta, name=name, method=method,
                                        seed=seed, seconds=result['seconds'], resolution=resolution,
                                        near=selected['near'], scale=selected['scale']))
                export[f'{method}_{seed}'] = labels[ix]
            if phase == 'validation':
                for method in ['kmeans', 'dataset_tuned_kmeans', 'multinomial', 'ring_kmeans']:
                    baseline = pickle.load((previous / f'{name}_{method}_K8_seed{seed}.pkl').open('rb'))
                    start = time.perf_counter()
                    if method == 'kmeans':
                        x = counts[60].to_numpy()
                        labels = KMeans(n_clusters=8, n_init=20, random_state=seed).fit_predict(np.sqrt(x/x.sum(1, keepdims=True)))
                    elif method == 'multinomial':
                        fit = composition.fit_composition_mixture(counts[30], n_components=8, n_init=5, seed=seed)
                        labels = fit.predict_proba(counts[30]).to_numpy().argmax(1)
                    else:
                        labels = baseline['q'].argmax(1)
                    records.append(evaluate(labels, ref=ref, meta=meta, name=name, method=method,
                                            seed=seed, seconds=time.perf_counter()-start if method in ['kmeans', 'multinomial'] else np.nan,
                                            baseline_origin='rerun_current' if method in ['kmeans', 'multinomial'] else 'cached_previous_investigation'))
                    export[f'{method}_{seed}'] = labels[ix]
                    # Check cached metrics against this independent scoring path.
                    if method not in ['kmeans', 'multinomial']:
                        assert abs(records[-1]['ari']-baseline['record']['ari']) < 1e-10
            else:
                for method in ['neural', 'multinomial', 'hellinger_kmeans']:
                    baseline = pickle.load((old / f'{name}_{method}_{seed}.pkl').open('rb'))
                    labels = baseline['labels']
                    records.append(evaluate(labels, ref=ref.loc[meta.index], meta=meta, name=name,
                                            method=method, seed=seed, seconds=np.nan,
                                            baseline_origin='cached_previous_benchmark'))
                    export[f'{method}_{seed}'] = labels
                    if meta.label.notna().any():
                        assert abs(records[-1]['ari']-baseline['score']['ari']) < 1e-10
            print(name, seed, 'done', flush=True)
            suffix = f'_{subset}' if subset else ''
            pd.DataFrame(records).to_csv(out / f'{phase}{suffix}.csv', index=False)
        export.to_parquet(out / f'{name}_assignments.parquet')
    print(pd.DataFrame(records).groupby('method').ari.mean(), flush=True)

versions = {p: importlib.metadata.version(p) for p in ['squidpy', 'scanpy', 'anndata', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'torch', 'igraph']}
(out / 'versions.json').write_text(json.dumps(versions, indent=2))
(out / 'script_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
