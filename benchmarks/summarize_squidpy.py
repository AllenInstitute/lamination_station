"""Produce tables, spatial maps, and an HTML report for the Squidpy comparison."""
import torch  # noqa: F401
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

out = Path('/results/lamination_station_squidpy')
shards = [out/f'validation_{subset}.csv' for subset in ['spinal', 'cortex_human', 'cortex_merfish']]
if all(p.exists() for p in shards):
    combined = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    assert len(combined) == 180 and not combined.duplicated(['name', 'method', 'seed']).any()
    combined.to_csv(out/'validation.csv', index=False)
validation = pd.read_csv(out/'validation.csv')
historical = pd.read_csv(out/'historical.csv')
assert len(validation) == 180 and len(historical) == 105
summaries = {}
stability = []
for phase, runs in [('validation', validation), ('historical', historical)]:
    summary = runs.groupby(['dataset', 'method']).agg(
        ari=('ari', 'mean'), ari_min=('ari', 'min'), ari_max=('ari', 'max'),
        ami=('ami', 'mean'), occupied=('occupied', 'mean'), sections=('name', 'nunique'),
    ).reset_index()
    summary.to_csv(out/f'{phase}_summary.csv', index=False)
    summaries[phase] = summary
    for name in runs.name.unique():
        frame = pd.read_parquet(out/f'{name}_assignments.parquet')
        for method in runs[runs.name == name].method.unique():
            scores = [adjusted_rand_score(frame[f'{method}_{a}'], frame[f'{method}_{b}'])
                      for a, b in itertools.combinations([17, 29, 43], 2)]
            stability.append(dict(phase=phase, name=name, method=method, mean_seed_ari=np.mean(scores)))
pd.DataFrame(stability).to_csv(out/'stability.csv', index=False)

methods = ['kmeans', 'multinomial', 'squidpy', 'squidpy_near8']
titles = {'kmeans': 'Hellinger k-means', 'multinomial': 'Current multinomial',
          'squidpy': 'Squidpy: selected', 'squidpy_near8': 'Squidpy: ~8 clusters',
          'neural': 'Archived neural', 'hellinger_kmeans': 'Hellinger k-means'}
colors = ['#576a89', '#b77632', '#147d83', '#86b7b4']
section_scores = validation.groupby(['name', 'method']).ari.mean().unstack()
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained', gridspec_kw={'width_ratios': [1.4, 1]})
for ax, prefix, title in zip(axes, ['spinal', 'cortex'], ['Spinal cord: independent polygon labels', 'Cortex: composition-derived proxy labels']):
    data = section_scores[section_scores.index.str.startswith(prefix)]
    for i, method in enumerate(methods):
        ax.barh(np.arange(len(data))+(i-1.5)*0.19, data[method], height=.18, color=colors[i], label=titles[method])
    ax.set(yticks=np.arange(len(data)), yticklabels=[x.replace('_', ' ') for x in data.index], xlabel='Adjusted Rand index (mean of 3 seeds)', xlim=(0, 1), title=title)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
axes[1].legend(fontsize=8, loc='lower right')
fig.savefig(out/'accuracy.png', dpi=180)
fig.savefig(out/'accuracy.pdf')
plt.close(fig)

fig, axes = plt.subplots(6, 4, figsize=(14, 19), layout='constrained')
for row, name in enumerate(sorted(validation[validation.dataset.str.startswith('spinal')].name.unique())):
    frame = pd.read_parquet(out/f'{name}_assignments.parquet')
    truth = frame.label.astype('string')
    cats = sorted(truth.dropna().unique())
    palette = dict(zip(cats, plt.cm.tab20(np.linspace(0, 1, len(cats))), strict=True))
    for col, method in enumerate(['reference', 'kmeans', 'multinomial', 'squidpy']):
        if method == 'reference':
            cell_colors = [palette.get(v, '#dddddd') if pd.notna(v) else '#dddddd' for v in truth]
            title = name.replace('_', ' ')+'\nPolygon annotations'
        else:
            labels = frame[f'{method}_17'].astype(str)
            tab = pd.crosstab(labels, truth)
            ii, jj = linear_sum_assignment(-tab.to_numpy())
            mapping = {tab.index[i]: palette[tab.columns[j]] for i,j in zip(ii,jj,strict=True)}
            # Unmatched subdivisions get their majority label's color, for display only.
            for label in tab.index:
                if label not in mapping:
                    mapping[label] = palette[tab.loc[label].idxmax()]
            cell_colors = [mapping.get(v, '#dddddd') for v in labels]
            ari = validation[(validation.name == name)&(validation.method == method)&(validation.seed == 17)].ari.iloc[0]
            title = f'{titles[method]}: ARI {ari:.3f}'
        ax = axes[row, col]
        ax.scatter(frame.x, frame.y, c=cell_colors, s=2, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
fig.savefig(out/'spinal_maps.png', dpi=160)
fig.savefig(out/'spinal_maps.pdf')
plt.close(fig)

selected = json.loads((out/'selected.json').read_text())
vtable = summaries['validation'].pivot(index='dataset', columns='method', values='ari')
htable = summaries['historical'].pivot(index='dataset', columns='method', values='ari')
sp = validation[validation.dataset.str.startswith('spinal')].groupby('method').ari.mean()
wins = int((section_scores.loc[section_scores.index.str.startswith('spinal'), 'squidpy'] > section_scores.loc[section_scores.index.str.startswith('spinal'), 'kmeans']).sum())
report = f'''<!doctype html><html><head><meta charset="utf-8"><title>Squidpy versus Lamination Station</title>
<style>body{{max-width:1150px;margin:35px auto;padding:0 20px;font:16px/1.55 system-ui;color:#26343b}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}code{{background:#eef2f3}}.note{{background:#eef5f5;padding:16px}}</style></head><body>
<h1>Squidpy versus Lamination Station</h1>
<p class="note">On six spinal validation sections, mean annotation ARI was <b>{sp['squidpy']:.3f}</b> for development-selected Squidpy, <b>{sp['kmeans']:.3f}</b> for Hellinger k-means, and <b>{sp['multinomial']:.3f}</b> for the current multinomial mixture. Squidpy beat k-means on {wins}/6 sections. These section-level results, not seed repetitions, are the biological comparison units.</p>
<h2>Separate validation sections</h2><img src="accuracy.png">{vtable.to_html(float_format=lambda x:f'{x:.3f}')}
<p>Each dataset contributes two sections, with seeds 17, 29, 43. Higher adjusted Rand index (ARI) means better partition agreement; 1 is identical, 0 is chance-level. Cortex labels were derived from composition and are not independent anatomical truth.</p>
<h2>Protocol</h2><p>Squidpy 1.8.4's public <code>calculate_niche_neighborhood</code> API consumes only coordinates and cell types. Neighborhood profiles use directed spatial kNN excluding self, relative frequencies, one graph hop, and 15 neighbors in the composition-space graph. The spatial graph is built within each section. There is no anatomical label column in the AnnData passed to Squidpy.</p>
<p>We screened spatial k=15/30/60, scaling on/off, and eight Leiden resolutions on the original three spinal development sections, seed 17. Highest mean development ARI selected <code>{selected}</code>. Settings were saved before validation. All validation fits use all cached reference cells, with a 10,000-cell cap inherited from the previous study, and score the same fixed focal cells (at most 2,048). This is transductive discovery on separate sections, not supervised prediction or donor-held-out validation. Sections may share donors, and these validation sections had been studied previously with other methods.</p>
<p>The near-eight-cluster comparison uses the selected neighborhood and scaling, choosing among the same resolution grid by distance to eight observed focal clusters, without annotation labels. Its cluster count is approximate, not constrained to exactly eight. The main k-means baseline uses Hellinger-transformed counts, 60 neighbors, K=8, 20 initializations; the multinomial mixture uses 30 neighbors, K=8, five initializations. Those neighborhood sizes were selected on development data in the prior investigation. Both were rerun using current code. Existing counts use Lamination Station's directional neighborhood adjustment; Squidpy uses ordinary spatial kNN. This primary comparison therefore tests complete workflows, not just clustering algorithms.</p>
<p>Additional archived baselines are dataset-tuned Hellinger k-means (neighborhood size selected separately per dataset on original development data) and multiscale ring k-means. Cached baseline scores were independently recomputed from predictions and checked against their saved metrics.</p>
<h2>Original neural benchmark</h2>{htable.to_html(float_format=lambda x:f'{x:.3f}',na_rep='Unlabelled')}
<p>These are the original seven test sections, one per dataset, with three seeds. Squidpy uses the frozen settings above; its clustering is restricted to the original focal cells via <code>cluster_mask</code>, retaining neighborhood information from the same reference section. Neural, multinomial and k-means predictions here are archived results, not rerun. The archived neural configuration was 200 epochs, hidden dimensions 64/32, Poisson observations, eight structures. This is not a claim against all seven neural variants or every possible training budget. Historical comparison differs from the full-reference validation fitting scope above.</p>
<h2>Spatial inspection</h2><img src="spinal_maps.png"><p>Seed 17, chosen in advance. Colors are aligned to reference labels for visualization only, using assignment matching and majority labels for extra clusters; this can hide subdivisions. Gray reference cells lack direct polygon annotations. Numerical ARI uses original cluster identities.</p>
<h2>Limits</h2><p>The study evaluates discrete domain calls, not continuous laminar depth, physical mixing fractions, or calibrated uncertainty. Neighborhood radius changes with sampling density. Seeds measure computational stability and are not independent biological replicates. No broad claim of superiority is warranted from these selected sections alone. The Squidpy resolution grid is charged as a complete pipeline in saved timing values; baseline timing is for a single fit, and cached historical timings are omitted, so those columns are not a fair speed comparison.</p>
<h2>Reproduce and inspect</h2><p>Run <code>squidpy_comparison.py development</code>, then <code>validation</code> and <code>historical</code>, then <code>summarize_squidpy.py</code>. Set TMPDIR=/scratch/runtime/tmp, MPLCONFIGDIR=/scratch/runtime/mpl, OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2, NUMBA_NUM_THREADS=2. Cached input files remain under /scratch/lamination_station_benchmark and /scratch/lamination_station_investigation; source paths and hashes are in <a href="input_manifest.json">input_manifest.json</a>. This reproduction requires those input caches.</p>
<p><a href="protocol.json">Protocol</a> · <a href="selected.json">Frozen selection</a> · <a href="development.csv">Development sweep</a> · <a href="validation.csv">Validation runs</a> · <a href="historical.csv">Historical comparison</a> · <a href="stability.csv">Seed stability</a> · <a href="versions.json">Versions</a>. Cell-level assignments are in the adjacent Parquet files. Figures have vector PDF copies.</p>
<p>Method reference: <a href="https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche_neighborhood.html">Squidpy neighborhood-niche documentation</a>.</p></body></html>'''
(out/'report.html').write_text(report)
inputs = []
for directory, patterns in [('/scratch/lamination_station_benchmark', ['*_development.pkl', '*_test.pkl', '*_test_*_*.pkl']), ('/scratch/lamination_station_investigation', ['*_fresh[12].pkl', '*_fresh[12]_*_K8_seed*.pkl'])]:
    for pattern in patterns:
        for p in sorted(Path(directory).glob(pattern)):
            inputs.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'input_manifest.json').write_text(json.dumps(inputs, indent=2))
for source in Path(__file__).parent.glob('*squidpy*.py'):
    shutil.copy2(source, out/source.name)
print(vtable.to_string())
print(htable.to_string())
print('spinal means', sp.to_dict())
print('Saved', out/'report.html')
