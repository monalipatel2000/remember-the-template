"""Three designs aimed at unfamiliar templates, on template-disjoint folds (PROTOCOL.md, Amendment 4). Run once."""
import json, sys
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
SEEDS, EPOCHS, BATCH, D = (13, 29, 47), 40, 256, 384
OUT = OUTPUTS / 'novel_template.json'
assert not OUT.exists(), 'novel_template.json exists. This is run once.'
T = lambda a: torch.tensor(np.asarray(a), device=DEV)
NAMES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

man = pd.read_csv(MANIFEST)
d = man[man.new_split.isin(['train', 'val'])].reset_index(drop=True); assert len(d) == 8503
feats = {}
for n in ('train', 'val'):
    z = np.load(FEAT / f'{n}_features.npz', allow_pickle=True); zi, zt = z['image'], z['text']
    for i, name in enumerate(z['images']): feats[str(name)] = (zi[i], zt[i])
I = T(np.stack([feats[n][0] for n in d.image])); Tx = T(np.stack([feats[n][1] for n in d.image]))
yn = d.target.values; y = T(yn)
exact = d.group_id.astype('category').cat.codes.values
f = pd.read_csv(DATA / 'template_families_0.7.csv').set_index('image').family.loc[d.image].values
a_ = pd.Series(f).astype('category').cat.codes.values; n_ = len(d)
rows = np.concatenate([np.arange(n_), np.arange(n_)]); cols = np.concatenate([a_, a_.max() + 1 + exact])
M = csr_matrix((np.ones(2 * n_), (rows, cols)), shape=(n_, cols.max() + 1))
_, fam = connected_components(M @ M.T, directed=False)

# Fixed emotion prompts, encoded with the same frozen CLIP text tower.
pfile = FEAT / 'emotion_prompts.npy'
if not pfile.exists():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REV).cuda().eval(); proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV)
    forms = ['a meme expressing {}', 'a person feeling {}', 'a reaction image showing {}', '{}']
    tk = proc.tokenizer([fm.format(e) for fm in forms for e in NAMES], padding=True, return_tensors='pt').to('cuda')
    with torch.no_grad(): P_ = F.normalize(model.text_projection(model.text_model(**tk).pooler_output).float(), dim=-1)
    np.save(pfile, P_.cpu().numpy()); del model
P = T(np.load(pfile))                                   # 24 x 768, form-major, emotion-minor
ZS = torch.cat([I @ P.T, Tx @ P.T], 1) * 20             # 48 prompt similarities per meme
zero_shot = (I @ P.T).view(len(I), 4, 6).mean(1).argmax(1).cpu().numpy()

# Label-free family centroids over all rows.
onehot = torch.zeros(len(I), int(fam.max()) + 1, device=DEV); onehot[torch.arange(len(I)), T(fam)] = 1
C = F.normalize((onehot.T @ I) / onehot.sum(0)[:, None], dim=-1)[T(fam)]
X = {'image-only head': I, 'image+text head': torch.cat([I, Tx], 1), 'A language-anchored': torch.cat([I, Tx, ZS], 1),
     'B template-balanced': torch.cat([I, Tx], 1), 'C template-residual': torch.cat([I, C, I - C, Tx], 1),
     'ABC combined': torch.cat([I, C, I - C, Tx, ZS], 1)}
WEIGHTED = {'B template-balanced', 'ABC combined'}


def fit_head(Xa, ya, Xq, seed, w=None):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xa.shape[1], D), nn.ReLU(), nn.Dropout(.3), nn.Linear(D, 6)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EPOCHS * ((len(ya) + BATCH - 1) // BATCH))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(ya), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]; l = F.cross_entropy(m(Xa[b]), ya[b], label_smoothing=.1, reduction='none')
            loss = l.mean() if w is None else (l * w[b]).sum() / w[b].sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(): return m(Xq).argmax(1).cpu().numpy()


Pr = {k: np.zeros((3, len(yn)), int) for k in X}
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=13).split(yn, yn, fam):
    assert not set(fam[a]) & set(fam[b])
    size = pd.Series(fam[a]).map(pd.Series(fam[a]).value_counts()).values; w = T(1 / size); w = (w / w.mean()).float()
    ta, tb = T(a), T(b)
    for k in X:
        for i, s in enumerate(SEEDS): Pr[k][i, b] = fit_head(X[k][ta], y[ta], X[k][tb], s, w if k in WEIGHTED else None)
    print('fold done', flush=True)

mf = lambda t, p: 100 * f1_score(t, p, average='macro')
res = {k: dict(macro=float(np.mean([mf(yn, v[i]) for i in range(3)])), macro_per_seed=[mf(yn, v[i]) for i in range(3)], acc=float(100 * (v == yn).mean()))
       for k, v in Pr.items()}
res['zero-shot prompts alone'] = dict(macro=float(mf(yn, zero_shot)), acc=float(100 * (zero_shot == yn).mean()))
members = [np.where(fam == c)[0] for c in np.unique(fam)]; rng = np.random.default_rng(13)
cands = [k for k in X if k not in ('image-only head',)]; diffs = {k: [] for k in cands}
for _ in range(2000):
    b = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
    base = np.mean([f1_score(yn[b], Pr['image-only head'][i][b], average='macro') for i in range(3)])
    for k in cands: diffs[k].append(100 * (np.mean([f1_score(yn[b], Pr[k][i][b], average='macro') for i in range(3)]) - base))
res['minus_image_only'] = {k: dict(diff=res[k]['macro'] - res['image-only head']['macro'], lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                                   wins_all_seeds=bool(all(x > z_ for x, z_ in zip(res[k]['macro_per_seed'], res['image-only head']['macro_per_seed']))))
                           for k, v in diffs.items()}
for k, v in res['minus_image_only'].items(): v['passes'] = bool(v['wins_all_seeds'] and v['lo'] > 0)
for k in list(X) + ['zero-shot prompts alone']:
    print(f"{k:24s} macro {res[k]['macro']:.2f} {[round(x, 2) for x in res[k].get('macro_per_seed', [])]} acc {res[k]['acc']:.2f}")
for k, v in res['minus_image_only'].items(): print(f"{k:24s} minus image-only {v['diff']:+.2f} [{v['lo']:.2f}, {v['hi']:.2f}] all seeds {v['wins_all_seeds']} PASS {v['passes']}")
np.savez_compressed(OUTPUTS / 'novel_template_predictions.npz', y=yn, family=fam, **{k.replace(' ', '_').replace('+', '_'): v for k, v in Pr.items()})
OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
