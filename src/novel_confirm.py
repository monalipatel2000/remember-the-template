"""Confirmation of the template-residual head on the MOOD test split (PROTOCOL.md, Amendment 4). Run once."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score

from paths import *
DEV = 'cuda'
SEEDS, EPOCHS, BATCH, D = (13, 29, 47), 40, 256, 384
OUT = OUTPUTS / 'novel_confirm.json'
assert not OUT.exists(), 'novel_confirm.json exists. This is run once.'
T = lambda a: torch.tensor(np.asarray(a), device=DEV)

man = pd.read_csv(MANIFEST); assert len(man) == 10005
feats = {}
for file in (FEAT / 'train_features.npz', FEAT / 'val_features.npz', FEAT / 'test_features.npz'):
    z = np.load(file, allow_pickle=True); zi, zt = z['image'], z['text']
    for i, name in enumerate(z['images']): feats[str(name)] = (zi[i], zt[i])
I = T(np.stack([feats[n][0] for n in man.image])); Tx = T(np.stack([feats[n][1] for n in man.image]))
exact = man.group_id.astype('category').cat.codes.values
f = pd.read_csv(DATA / 'template_families_0.7.csv').set_index('image').family.loc[man.image].values
a_ = pd.Series(f).astype('category').cat.codes.values; n_ = len(man)
rows = np.concatenate([np.arange(n_), np.arange(n_)]); cols = np.concatenate([a_, a_.max() + 1 + exact])
M = csr_matrix((np.ones(2 * n_), (rows, cols)), shape=(n_, cols.max() + 1)); _, fam = connected_components(M @ M.T, directed=False)
onehot = torch.zeros(n_, int(fam.max()) + 1, device=DEV); onehot[torch.arange(n_), T(fam)] = 1
C = F.normalize((onehot.T @ I) / onehot.sum(0)[:, None], dim=-1)[T(fam)]
X = {'image-only head': I, 'image+text head': torch.cat([I, Tx], 1), 'template-residual head': torch.cat([I, C, I - C, Tx], 1)}
tr, te = np.where(man.new_split == 'train')[0], np.where(man.new_split == 'test')[0]; assert len(tr) == 7000 and len(te) == 1502
y = T(man.target.values); yn = man.target.values[te]; ft = fam[te]
novel = ~np.isin(ft, fam[tr]); size_all = pd.Series(fam).map(pd.Series(fam).value_counts()).values[te]


def fit_head(Xa, ya, Xq, seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xa.shape[1], D), nn.ReLU(), nn.Dropout(.3), nn.Linear(D, 6)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EPOCHS * ((len(ya) + BATCH - 1) // BATCH))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(ya), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]; loss = F.cross_entropy(m(Xa[b]), ya[b], label_smoothing=.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(): return m(Xq).argmax(1).cpu().numpy()


P = {k: np.stack([fit_head(v[T(tr)], y[T(tr)], v[T(te)], s) for s in SEEDS]) for k, v in X.items()}
E = np.load(recorded('final_test_predictions.npz'), allow_pickle=True)
assert (E['images'] == man.image.values[te].astype(str)).all()
P['EMC (earlier, for reference)'] = np.stack([E[f'head_memory_seed{s}'].argmax(1) for s in SEEDS])
mf = lambda t, p: 100 * f1_score(t, p, average='macro')
subsets = {'all test': np.ones(len(te), bool), 'template-novel': novel, 'template seen in training': ~novel,
           'novel, family has other unlabelled members': novel & (size_all > 1), 'novel, singleton family': novel & (size_all == 1)}
res = {'n': {k: int(m.sum()) for k, m in subsets.items()}}
rng = np.random.default_rng(13)
for name, m in subsets.items():
    r = {k: dict(macro=float(np.mean([mf(yn[m], v[i][m]) for i in range(3)])), macro_per_seed=[mf(yn[m], v[i][m]) for i in range(3)],
                 acc=float(100 * (v[:, m] == yn[m]).mean())) for k, v in P.items()}
    idx = np.where(m)[0]; members = [idx[ft[idx] == c] for c in np.unique(ft[idx])]; d = []
    for _ in range(2000):
        b = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        d.append(100 * np.mean([f1_score(yn[b], P['template-residual head'][i][b], average='macro') - f1_score(yn[b], P['image-only head'][i][b], average='macro') for i in range(3)]))
    r['residual_minus_image_only'] = dict(diff=r['template-residual head']['macro'] - r['image-only head']['macro'], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                                          wins_all_seeds=bool(all(x > z_ for x, z_ in zip(r['template-residual head']['macro_per_seed'], r['image-only head']['macro_per_seed']))))
    res[name] = r
    print(f"{name} (n={int(m.sum())})")
    for k in P: print(f"   {k:30s} macro {r[k]['macro']:.2f} {[round(x, 2) for x in r[k]['macro_per_seed']]} acc {r[k]['acc']:.2f}")
    v = r['residual_minus_image_only']; print(f"   residual minus image-only {v['diff']:+.2f} [{v['lo']:.2f}, {v['hi']:.2f}] all seeds {v['wins_all_seeds']}")
v = res['template-novel']['residual_minus_image_only']; res['confirmed'] = bool(v['wins_all_seeds'] and v['lo'] > 0); print('CONFIRMED', res['confirmed'])
np.savez_compressed(OUTPUTS / 'novel_confirm_predictions.npz', y=yn, family=ft, novel=novel, **{k.split(' (')[0].replace(' ', '_').replace('+', '_').replace('-', '_'): v for k, v in P.items()})
OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
