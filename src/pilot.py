"""Pilot: exemplar-contrast retrieval network on frozen CLIP features (MOOD).

Train on the 7,000 grouped training rows, score the 1,503 known validation rows. Grouped test is never loaded.
Decision rule, written before the first run: the full model is "promising" only if its validation macro-F1 beats
the best control (MLP, kNN vote, MLP+kNN average) in all three seeds. Everything is reported either way.
Epochs are fixed at 40 for every neural arm; nothing is selected on validation.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from paths import *
DEV = 'cuda'
SEEDS, EPOCHS, BATCH, K, D = (13, 29, 47), 40, 256, 16, 384

man = pd.read_csv(MANIFEST)


def load(split, name):
    d = man[man.new_split == split].reset_index(drop=True)
    z = np.load(FEAT / f'{name}_features.npz', allow_pickle=True)
    p = {str(n): i for i, n in enumerate(z['images'])}; o = np.array([p[i] for i in d.image])
    return (torch.tensor(z['image'][o]), torch.tensor(z['text'][o]), torch.tensor(d.target.values),
            d.group_id.astype('category').cat.codes.values, d)


Ti, Tt, Ty, Tg, _ = load('train', 'train')
Vi, Vt, Vy, _, _ = load('val', 'val')
Ti, Tt, Ty, Vi, Vt, Vy = (a.to(DEV) for a in (Ti, Tt, Ty, Vi, Vt, Vy))
tg = torch.tensor(pd.Series(man[man.new_split == 'train'].group_id.values).astype('category').cat.codes.values, device=DEV)


def neighbours(qi, exclude_group=None):
    """Top-K training items by CLIP image cosine. For training queries, the query's own duplicate group is excluded."""
    s = qi @ Ti.T
    if exclude_group is not None: s = s.masked_fill(exclude_group[:, None] == tg[None, :], -2.)
    sim, idx = s.topk(K, dim=1)
    return idx, sim


Tn, Ts = neighbours(Ti, tg)
Vn, Vs = neighbours(Vi)


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.q = nn.Sequential(nn.Linear(1536, D), nn.ReLU(), nn.Dropout(.3))
        if mode != 'mlp':
            width = 1536 * (2 if 'nodiff' not in mode else 1)
            self.n = nn.Linear(width, D)
            self.lab = nn.Embedding(6, D)
            self.sim = nn.Linear(1, D)
            self.att = nn.MultiheadAttention(D, 4, dropout=.1, batch_first=True)
            self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D * (1 if mode == 'mlp' else 2), 6)

    def forward(self, qi, qt, idx, sim):
        q = self.q(torch.cat([qi, qt], 1))
        if self.mode == 'mlp': return self.head(q)
        ni, nt = Ti[idx], Tt[idx]
        feats = [ni, nt]
        if 'nodiff' not in self.mode: feats += [qi[:, None] - ni, qt[:, None] - nt]
        tok = self.n(torch.cat(feats, 2)) + self.sim(sim[..., None])
        if 'nolabel' not in self.mode: tok = tok + self.lab(Ty[idx])
        a, _ = self.att(q[:, None], tok, tok)
        return self.head(torch.cat([q, self.norm(a[:, 0])], 1))


def run(mode, seed):
    torch.manual_seed(seed)
    m = Net(mode).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EPOCHS * ((len(Ty) + BATCH - 1) // BATCH))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Ty), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = F.cross_entropy(m(Ti[b], Tt[b], Tn[b], Ts[b]), Ty[b], label_smoothing=.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(): return F.softmax(m(Vi, Vt, Vn, Vs), 1)


def knn_prob(tau=.05):
    w = F.softmax(Vs / tau, 1)
    return torch.zeros(len(Vy), 6, device=DEV).scatter_add_(1, Ty[Vn], w)


f1 = lambda p: 100 * f1_score(Vy.cpu().numpy(), p.argmax(1).cpu().numpy(), average='macro')
acc = lambda p: 100 * float((p.argmax(1) == Vy).float().mean())
res = {}
kp = knn_prob()
res['knn_vote'] = {'macro': [f1(kp)] * 3, 'acc': [acc(kp)] * 3}
probs = {}
for mode in ('mlp', 'full', 'nolabel', 'nodiff', 'nolabel_nodiff'):
    probs[mode] = [run(mode, s) for s in SEEDS]
    res[mode] = {'macro': [f1(p) for p in probs[mode]], 'acc': [acc(p) for p in probs[mode]]}
    print(mode, [round(x, 2) for x in res[mode]['macro']], flush=True)
avg = [(p + kp) / 2 for p in probs['mlp']]
res['mlp_plus_knn'] = {'macro': [f1(p) for p in avg], 'acc': [acc(p) for p in avg]}
for k, v in res.items():
    v['macro_mean'], v['acc_mean'] = float(np.mean(v['macro'])), float(np.mean(v['acc']))
    print(f"{k:16s} macro {v['macro_mean']:.2f} {[round(x, 2) for x in v['macro']]} acc {v['acc_mean']:.2f}")
best = max(('mlp', 'knn_vote', 'mlp_plus_knn'), key=lambda k: res[k]['macro_mean'])
res['decision'] = {'best_control': best,
                   'full_beats_best_control_all_seeds': bool(all(a > b for a, b in zip(res['full']['macro'], res[best]['macro'])))}
print(res['decision'])
(OUTPUTS / 'pilot_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
