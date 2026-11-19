"""Post-hoc breakdown of SAVED test predictions by template-seen vs template-novel (no model is trained or rerun)."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

from paths import *
E = np.load(recorded('final_test_predictions.npz'), allow_pickle=True)
B = np.load(recorded('baseline_test_predictions.npz'), allow_pickle=True)
N = np.load(recorded('novel_confirm_predictions.npz'), allow_pickle=True)
assert (E['images'] == B['images']).all()
y = N['y']; assert (y == E['targets']).all(); novel = N['novel']; fam = N['family']
am = lambda a: a.argmax(1) if a.ndim == 2 else a
S = (13, 29, 47)
P = {'ALFRED (local)': np.stack([am(E[f'ALFRED_seed{s}']) for s in S]),
     'Hate-CLIPper (adapted)': np.stack([am(B[f'HateCLIPper_seed{s}']) for s in S]),
     'MemeCLIP (adapted)': np.stack([am(B[f'MemeCLIP_seed{s}']) for s in S]),
     'CLIP head, image only': N['image_only_head'], 'CLIP head, image + text': N['image_text_head'],
     'Exemplar memory alone': np.stack([am(E[f'memory_seed{s}']) for s in S]),
     'EMC': N['EMC']}
mf = lambda t, p: 100 * f1_score(t, p, average='macro')
res = {}; rng = np.random.default_rng(13)
for name, m in {'all': np.ones(len(y), bool), 'seen': ~novel, 'novel': novel}.items():
    r = {'n': int(m.sum())}
    for k, v in P.items(): r[k] = float(np.mean([mf(y[m], v[i][m]) for i in range(3)]))
    idx = np.where(m)[0]; members = [idx[fam[idx] == c] for c in np.unique(fam[idx])]
    for base in ('CLIP head, image only', 'MemeCLIP (adapted)'):
        d = []
        for _ in range(2000):
            b = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
            d.append(100 * np.mean([f1_score(y[b], P['EMC'][i][b], average='macro') - f1_score(y[b], P[base][i][b], average='macro') for i in range(3)]))
        r[f'EMC minus {base}'] = dict(diff=r['EMC'] - r[base], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
    res[name] = r
    print(name, json.dumps(r, indent=1))
(OUTPUTS / 'seen_novel_breakdown.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
