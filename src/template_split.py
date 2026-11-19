"""Template-disjoint against ordinary folds on MOOD training+validation rows (PROTOCOL.md, Amendment 3). Run once."""
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
from sklearn.model_selection import StratifiedGroupKFold

from paths import *
DEV = 'cuda'
SEEDS, EPOCHS, BATCH, K, LAM, TAU, D = (13, 29, 47), 40, 256, 16, .5, .05, 384
OUT = OUTPUTS / 'template_split.json'
assert not OUT.exists(), 'template_split.json exists. This is run once.'
T = lambda a: torch.tensor(np.asarray(a), device=DEV)

man = pd.read_csv(MANIFEST)
d = man[man.new_split.isin(['train', 'val'])].reset_index(drop=True); assert len(d) == 8503
feats = {}
for n in ('train', 'val'):
    z = np.load(FEAT / f'{n}_features.npz', allow_pickle=True)
    zi, zt = z['image'], z['text']
    for i, name in enumerate(z['images']): feats[str(name)] = (zi[i], zt[i])
I = T(np.stack([feats[n][0] for n in d.image])); X = torch.cat([I, T(np.stack([feats[n][1] for n in d.image]))], 1)
yn = d.target.values; y = T(yn)
exact = d.group_id.astype('category').cat.codes.values


def families(th):
    """Family ids on the 8,503 rows, merged with exact-duplicate groups so that neither is ever split."""
    f = pd.read_csv(DATA / f'template_families_{th}.csv').set_index('image').family.loc[d.image].values
    a = pd.Series(f).astype('category').cat.codes.values; n = len(d)
    rows = np.concatenate([np.arange(n), np.arange(n)]); cols = np.concatenate([a, a.max() + 1 + exact])
    _, lab = connected_components(csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, cols.max() + 1)) @
                                  csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, cols.max() + 1)).T, directed=False)
    return lab


def fit_head(Xa, ya, Xq, seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xa.shape[1], D), nn.ReLU(), nn.Dropout(.3), nn.Linear(D, 6)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EPOCHS * ((len(ya) + BATCH - 1) // BATCH))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(ya), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = F.cross_entropy(m(Xa[b]), ya[b], label_smoothing=.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(): return F.softmax(m(Xq), 1)


def run(groups):
    P = {k: np.zeros((3, len(yn)), int) for k in ('image-only head', 'image+text head', 'memory alone', 'EMC')}
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=13).split(yn, yn, groups):
        assert not set(groups[a]) & set(groups[b])
        a, b = T(a), T(b)
        sim, idx = (I[b] @ I[a].T).topk(K, dim=1)
        mem = torch.zeros(len(b), 6, device=DEV).scatter_add_(1, y[a][idx], F.softmax(sim / TAU, 1)); bb = b.cpu().numpy()
        P['memory alone'][:, bb] = mem.argmax(1).cpu().numpy()
        for i, s in enumerate(SEEDS):
            h = fit_head(X[a], y[a], X[b], s)
            P['image+text head'][i, bb] = h.argmax(1).cpu().numpy(); P['EMC'][i, bb] = (LAM * h + (1 - LAM) * mem).argmax(1).cpu().numpy()
            P['image-only head'][i, bb] = fit_head(I[a], y[a], I[b], s).argmax(1).cpu().numpy()
    return P


def summarise(P, boot_groups):
    r = {k: dict(macro=float(np.mean([100 * f1_score(yn, v[i], average='macro') for i in range(3)])),
                 macro_per_seed=[100 * f1_score(yn, v[i], average='macro') for i in range(3)],
                 acc=float(100 * (v == yn).mean())) for k, v in P.items()}
    members = [np.where(boot_groups == c)[0] for c in np.unique(boot_groups)]; rng = np.random.default_rng(13)
    diffs = {k: [] for k in ('image-only head', 'image+text head')}
    for _ in range(2000):
        b = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        f = {k: np.mean([f1_score(yn[b], P[k][i][b], average='macro') for i in range(3)]) for k in ('EMC', *diffs)}
        for k in diffs: diffs[k].append(100 * (f['EMC'] - f[k]))
    r['EMC_minus'] = {k: dict(diff=r['EMC']['macro'] - r[k]['macro'], lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                              emc_wins_all_seeds=bool(all(x > z_ for x, z_ in zip(r['EMC']['macro_per_seed'], r[k]['macro_per_seed']))))
                      for k, v in diffs.items()}
    return r


res = {}
fam = {th: families(th) for th in (0.7, 0.5, 0.9)}
res['families'] = {str(th): dict(n=int(len(np.unique(f))), rows_in_families_ge2=int((pd.Series(f).map(pd.Series(f).value_counts()) >= 2).sum()),
                                 largest=int(np.bincount(f).max())) for th, f in fam.items()}
# Share of rows that, under ordinary folds, have a family member in their training folds (label-free description).
ordinary_folds = np.zeros(len(yn), int)
for k, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=13).split(yn, yn, exact)): ordinary_folds[b] = k
for th, f in fam.items():
    cnt = pd.crosstab(f, ordinary_folds).values; tot = cnt.sum(1)
    res['families'][str(th)]['rows_with_family_member_in_training_under_ordinary_folds'] = float(100 * np.mean((tot[f] - cnt[f, ordinary_folds]) > 0))
print(res['families'], flush=True)
res['ordinary'] = summarise(run(exact), fam[0.7])
for th in (0.7, 0.5, 0.9): res[f'template_disjoint_{th}'] = summarise(run(fam[th]), fam[th])
for k, r in res.items():
    if k == 'families': continue
    print(k)
    for m in ('image-only head', 'image+text head', 'memory alone', 'EMC'): print(f"   {m:16s} macro {r[m]['macro']:.2f} {[round(x, 2) for x in r[m]['macro_per_seed']]} acc {r[m]['acc']:.2f}")
    print('   EMC minus', {a: (round(v['diff'], 2), round(v['lo'], 2), round(v['hi'], 2), v['emc_wins_all_seeds']) for a, v in r['EMC_minus'].items()})
OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
