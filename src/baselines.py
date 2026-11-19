"""MemeCLIP and Hate-CLIPper heads on frozen CLIP ViT-L/14, reimplemented from the official repositories.
See PROTOCOL.md, Amendment 1. Run once."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from paths import *
DEV = 'cuda'
SEEDS = (13, 29, 47)
OUT = OUTPUTS / 'baselines.json'
assert not OUT.exists(), 'Baselines have already been scored. They are run once.'

man = pd.read_csv(MANIFEST)
Z = np.load(FEAT / 'unprojected_features.npz', allow_pickle=True)
pos = {str(n): i for i, n in enumerate(Z['images'])}
T = lambda a: torch.tensor(np.asarray(a), device=DEV)


def part(split):
    d = man[man.new_split == split].reset_index(drop=True); o = np.array([pos[i] for i in d.image])
    return {k: T(Z[k][o]) for k in ('vision_pooled', 'text_pooled', 'image_projected')}, T(d.target.values), d


TR, ytr, _ = part('train'); VA, yva, _ = part('val'); TE, yte, dte = part('test')
INIT = T(Z['memeclip_init'])


class MemeCLIP(nn.Module):
    """image_map/text_map -> adapter residual mix (0.2) -> L2 norm -> Hadamard product -> pre-output -> cosine classifier."""
    keys = ('image_projected', 'text_pooled')

    def __init__(self):
        super().__init__()
        self.image_map = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(.1))
        self.text_map = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(.1))
        adapter = lambda: nn.Sequential(nn.Linear(1024, 256, bias=False), nn.ReLU(), nn.Linear(256, 1024, bias=False), nn.ReLU())
        self.img_adapter, self.text_adapter = adapter(), adapter()
        self.pre_output = nn.Sequential(nn.Dropout(.4), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(.2))
        self.weight = nn.Parameter(INIT.clone())

    def forward(self, img, txt):
        ip, tp = self.image_map(img), self.text_map(txt)
        i = .2 * self.img_adapter(ip) + .8 * ip
        t = .2 * self.text_adapter(tp) + .8 * tp
        f = self.pre_output(F.normalize(i, dim=-1) * F.normalize(t, dim=-1))
        return 30 * F.linear(F.normalize(f, dim=-1), F.normalize(self.weight, dim=-1))


class HateCLIPper(nn.Module):
    """Linear maps to 32-d -> L2 norm -> outer product (cross fusion) -> pre-output -> linear output."""
    keys = ('vision_pooled', 'text_pooled')

    def __init__(self):
        super().__init__()
        self.image_map = nn.Sequential(nn.Linear(1024, 32), nn.Dropout(.2))
        self.text_map = nn.Sequential(nn.Linear(768, 32), nn.Dropout(.2))
        self.pre_output = nn.Sequential(nn.Dropout(.4), nn.Linear(32 * 32, 32), nn.ReLU(), nn.Dropout(.1))
        self.output = nn.Linear(32, 6)

    def forward(self, img, txt):
        i, t = F.normalize(self.image_map(img), p=2, dim=1), F.normalize(self.text_map(txt), p=2, dim=1)
        return self.output(self.pre_output(torch.bmm(i.unsqueeze(2), t.unsqueeze(1)).reshape(len(i), -1)))


def score(p, y):
    yt, pr = y.cpu().numpy(), p.argmax(1).cpu().numpy()
    return dict(macro=100 * f1_score(yt, pr, average='macro'), weighted=100 * f1_score(yt, pr, average='weighted'), acc=100 * float((pr == yt).mean()))


def run(cls, epochs, clip, seed):
    torch.manual_seed(seed)
    m = cls().to(DEV); a, b = cls.keys
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    best = (-1, None, None, 0)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(ytr), device=DEV)
        for i in range(0, len(perm), 16):
            bi = perm[i:i + 16]
            loss = F.cross_entropy(m(TR[a][bi], TR[b][bi]), ytr[bi])
            opt.zero_grad(); loss.backward()
            if clip: nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
        m.eval()
        with torch.no_grad():
            pv = F.softmax(m(VA[a], VA[b]), 1); v = score(pv, yva)['macro']
            if v > best[0]: best = (v, pv, F.softmax(m(TE[a], TE[b]), 1), ep + 1)
    return best


res, preds = {}, {}
for name, cls, epochs, clip in (('MemeCLIP', MemeCLIP, 10, None), ('Hate-CLIPper', HateCLIPper, 20, .1)):
    runs = [run(cls, epochs, clip, s) for s in SEEDS]
    preds[name] = [r[2] for r in runs]
    res[name] = {'best_epoch': [r[3] for r in runs]}
    for split, y, j in (('val', yva, 1), ('test', yte, 2)):
        sc = [score(r[j], y) for r in runs]
        res[name][split] = {**{k: float(np.mean([s[k] for s in sc])) for k in ('macro', 'weighted', 'acc')},
                            'macro_sd': float(np.std([s['macro'] for s in sc], ddof=1)), 'macro_per_seed': [s['macro'] for s in sc]}
    print(name, 'val', round(res[name]['val']['macro'], 2), 'test', round(res[name]['test']['macro'], 2),
          [round(x, 2) for x in res[name]['test']['macro_per_seed']], 'acc', round(res[name]['test']['acc'], 2), 'epochs', res[name]['best_epoch'], flush=True)

# Paired bootstrap against the saved EMC test predictions (same rows, same seeds).
E = np.load(recorded('final_test_predictions.npz'), allow_pickle=True)
assert (E['images'] == dte.image.values.astype(str)).all() and (E['targets'] == dte.target.values).all()
yn = dte.target.values
emc = np.stack([E[f'head_memory_seed{s}'].argmax(1) for s in SEEDS])
emc_macro = [100 * f1_score(yn, emc[i], average='macro') for i in range(3)]
rng = np.random.default_rng(13); res['EMC_minus'] = {}
for name in preds:
    pb = np.stack([p.argmax(1).cpu().numpy() for p in preds[name]]); d = []
    for _ in range(2000):
        b = rng.integers(0, len(yn), len(yn))
        d.append(100 * np.mean([f1_score(yn[b], emc[i][b], average='macro') - f1_score(yn[b], pb[i][b], average='macro') for i in range(3)]))
    res['EMC_minus'][name] = dict(diff=float(np.mean(emc_macro)) - res[name]['test']['macro'], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                                  emc_wins_all_seeds=bool(all(x > y_ for x, y_ in zip(emc_macro, res[name]['test']['macro_per_seed']))))
print(json.dumps(res['EMC_minus'], indent=1))
np.savez_compressed(OUTPUTS / 'baseline_test_predictions.npz', images=dte.image.values.astype(str), targets=yn,
                    **{f"{k.replace('-', '')}_seed{s}": preds[k][i].cpu().numpy() for k in preds for i, s in enumerate(SEEDS)})
OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
