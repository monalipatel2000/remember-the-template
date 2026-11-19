"""Figures for the exemplar-memory paper. Reads saved result files only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import *
FIG = OUTPUTS / 'figures'; FIG.mkdir(exist_ok=True)
PAL = ['#2a78d6', '#eb6834', '#1baf7a']
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'pdf.fonttype': 42})
test = json.loads(recorded('final_test.json').read_text())
date = json.loads(recorded('final_date.json').read_text())
val = json.loads(recorded('final_val.json').read_text())
al = json.loads((RESULTS / 'alfred/date_benchmark_results.json').read_text())

# Figure 2: three evaluation settings, three models.
groups = ['Held-out test', 'Ordinary folds', 'Held-out date']
vals = {'ALFRED (local)': [test['ALFRED']['macro'], al['ordinary_folds']['mean_macro'], al['unseen_date']['mean_macro']],
        'CLIP head': [test['head']['macro'], date['ordinary']['head']['macro'], date['unseen_date']['head']['macro']],
        'CLIP head + exemplar memory': [test['head+memory']['macro'], date['ordinary']['head+memory']['macro'],
                                        date['unseen_date']['head+memory']['macro']]}
fig, ax = plt.subplots(figsize=(3.4, 2.5)); x = np.arange(3)
for i, (k, v) in enumerate(vals.items()):
    ax.bar(x + (i - 1) * .27, v, width=.25, color=PAL[[1, 2, 0][i]], label=k)
    for xx, vv in zip(x, v): ax.text(xx + (i - 1) * .27, vv + 1, f'{vv:.1f}', ha='center', fontsize=6, color=INK)
ax.set_xticks(x); ax.set_xticklabels(groups, color=INK); ax.set_ylim(0, 105); ax.set_ylabel('Macro-F1 (%)')
ax.grid(axis='x', visible=False); ax.legend(fontsize=6.5, loc='upper center', ncol=1, bbox_to_anchor=(.72, 1.04))
fig.savefig(FIG / 'fig2_main.pdf', bbox_inches='tight'); fig.savefig(FIG / 'fig2_main.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# Figure 3: sensitivity on validation (information only; reported model is the marked setting).
s = val['sensitivity']
fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9), sharey=True, gridspec_kw={'wspace': .08})
for ax, pre, name, chosen, log in ((axes[0], 'K=', 'Neighbours K', 16, True), (axes[1], 'head_weight=', 'Weight on the head', .5, False),
                                   (axes[2], 'tau=', 'Temperature', .05, True)):
    xs = [float(k.split('=')[1]) for k in s if k.startswith(pre)]; ys = [s[k] for k in s if k.startswith(pre)]
    ax.plot(xs, ys, marker='o', ms=4, lw=2, color=PAL[0])
    ax.plot([chosen], [ys[xs.index(chosen)]], marker='o', ms=9, mfc='none', mec=INK, mew=1.2, lw=0)
    if log: ax.set_xscale('log')
    ax.set_xlabel(name); ax.set_ylim(66, 83)
axes[0].set_ylabel('Validation macro-F1 (%)')
fig.savefig(FIG / 'fig3_sensitivity.pdf', bbox_inches='tight'); fig.savefig(FIG / 'fig3_sensitivity.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('figures written')
