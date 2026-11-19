"""Final evaluation of the exemplar-memory CLIP classifier. Stages: val | date | met | test. See PROTOCOL.md."""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from paths import *
DEV = 'cuda'
SEEDS, EPOCHS, BATCH, K, LAM, TAU, D = (13, 29, 47), 40, 256, 16, .5, .05, 384
STAGE = sys.argv[1]
T = lambda a: torch.tensor(np.asarray(a), device=DEV)


def fit_head(X, y, Xq, seed, ncls):
    """Pilot `mlp` arm, unchanged. Returns softmax probabilities on Xq."""
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(X.shape[1], D), nn.ReLU(), nn.Dropout(.3), nn.Linear(D, ncls)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EPOCHS * ((len(y) + BATCH - 1) // BATCH))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(y), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = F.cross_entropy(m(X[b]), y[b], label_smoothing=.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(): return F.softmax(m(Xq), 1)


def memory(keys, y, qkeys, ncls, k=K, tau=TAU):
    sim, idx = (qkeys @ keys.T).topk(min(k, len(y)), dim=1)
    return torch.zeros(len(qkeys), ncls, device=DEV).scatter_add_(1, y[idx], F.softmax(sim / tau, 1))


def scores(p, y, names=None):
    yt, pr = y.cpu().numpy(), p.argmax(1).cpu().numpy()
    out = dict(macro=100 * f1_score(yt, pr, average='macro'), weighted=100 * f1_score(yt, pr, average='weighted'),
               acc=100 * float((pr == yt).mean()))
    if names: out['per_class_f1'] = dict(zip(names, (100 * f1_score(yt, pr, average=None)).round(2).tolist()))
    return out


def summarise(runs):
    return {**{k: float(np.mean([r[k] for r in runs])) for k in ('macro', 'weighted', 'acc')},
            'macro_sd': float(np.std([r['macro'] for r in runs], ddof=1)) if len(runs) > 1 else 0.,
            'macro_per_seed': [r['macro'] for r in runs], 'seeds': runs}


def mood(split, file):
    man = pd.read_csv(MANIFEST)
    d = man[man.new_split == split].reset_index(drop=True)
    z = np.load(file, allow_pickle=True)
    p = {str(n): i for i, n in enumerate(z['images'])}; o = np.array([p[i] for i in d.image])
    return T(z['image'][o]), T(z['text'][o]), d


CLIPDIR = FEAT
NAMES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
cat = lambda a, b: torch.cat([a, b], 1)
out = OUTPUTS / f'final_{STAGE}.json'

if STAGE == 'val':
    Ti, Tt, dtr = mood('train', CLIPDIR / 'train_features.npz'); Vi, Vt, dva = mood('val', CLIPDIR / 'val_features.npz')
    Ty, Vy = T(dtr.target.values), T(dva.target.values)
    heads = {'image': (Ti, Vi), 'text': (Tt, Vt), 'image+text': (cat(Ti, Tt), cat(Vi, Vt))}
    mems = {'image': (Ti, Vi), 'text': (Tt, Vt), 'image+text': (cat(Ti, Tt) / 2 ** .5, cat(Vi, Vt) / 2 ** .5)}
    hp = {h: [fit_head(a, Ty, b, s, 6) for s in SEEDS] for h, (a, b) in heads.items()}
    mp = {m: memory(a, Ty, b, 6) for m, (a, b) in mems.items()}
    res = {'ablation': {}, 'sensitivity': {}}
    for h in hp: res['ablation'][f'head[{h}]'] = summarise([scores(p, Vy) for p in hp[h]])
    for m in mp: res['ablation'][f'memory[{m}]'] = summarise([scores(mp[m], Vy)])
    for h in hp:
        for m in mp: res['ablation'][f'head[{h}] + memory[{m}]'] = summarise([scores(LAM * p + (1 - LAM) * mp[m], Vy) for p in hp[h]])
    base = hp['image+text']
    for k in (1, 4, 8, 16, 32, 64, 128):
        q = memory(Ti, Ty, Vi, 6, k=k); res['sensitivity'][f'K={k}'] = summarise([scores(LAM * p + (1 - LAM) * q, Vy) for p in base])['macro']
    for lam in (0., .25, .5, .75, 1.):
        res['sensitivity'][f'head_weight={lam}'] = summarise([scores(lam * p + (1 - lam) * mp['image'], Vy) for p in base])['macro']
    for tau in (.01, .02, .05, .1, .2, 1.):
        q = memory(Ti, Ty, Vi, 6, tau=tau); res['sensitivity'][f'tau={tau}'] = summarise([scores(LAM * p + (1 - LAM) * q, Vy) for p in base])['macro']
    # Matched ALFRED on the same validation rows, from saved predictions.
    al = []
    for s in SEEDS:
        z = np.load(RESULTS / f'alfred/seed{s}_val_predictions.npz', allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z['images'])}; sel = [pos[i] for i in dva.image]
        assert (z['targets'][sel] == dva.target.values).all()
        al.append(scores(T(z['probabilities'][sel]), Vy))
    res['ablation']['ALFRED (saved, matched local)'] = summarise(al)
    for k, v in res['ablation'].items(): print(f"{k:38s} macro {v['macro']:.2f} +- {v['macro_sd']:.2f}  acc {v['acc']:.2f}")
    print({k: round(v, 2) for k, v in res['sensitivity'].items()})

elif STAGE == 'date':
    Ti, Tt, d = mood('train', CLIPDIR / 'train_features.npz')
    t = pd.read_csv(DATA / 'mood_archive_dates.csv')
    d = d.merge(t, on='image', how='left'); assert len(d) == 7000
    X, y = cat(Ti, Tt), T(d.target.values); yn = d.target.values
    gid = d.group_id.astype('category').cat.codes.values; dates = d.date.values
    big = d.date.value_counts(); big = sorted(big[big >= 100].index.tolist()); cov = np.isin(dates, big)
    arms = ('head', 'memory', 'head+memory')
    P = {r: {a: {s: torch.zeros(7000, 6, device=DEV) for s in SEEDS} for a in arms} for r in ('unseen_date', 'ordinary')}

    def fill(regime, tr_idx, te_idx):
        a, q = T(tr_idx), T(te_idx)
        mem = memory(Ti[a], y[a], Ti[q], 6)
        for s in SEEDS:
            h = fit_head(X[a], y[a], X[q], s, 6)
            P[regime]['head'][s][q], P[regime]['memory'][s][q], P[regime]['head+memory'][s][q] = h, mem, LAM * h + (1 - LAM) * mem
    for dt in big:
        te = dates == dt; fill('unseen_date', np.where(~te & ~np.isin(gid, gid[te]))[0], np.where(te)[0]); print(dt, flush=True)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=13).split(yn, yn, gid): fill('ordinary', a, b)
    c = T(np.where(cov)[0]); res = {'rows': int(cov.sum())}
    for r in P:
        res[r] = {}
        for a in arms:
            runs = []
            for s in SEEDS:
                sc = scores(P[r][a][s][c], y[c]); pr = P[r][a][s].argmax(1).cpu().numpy()
                sc['worst_date'] = min(100 * float((pr[dates == dt] == yn[dates == dt]).mean()) for dt in big); runs.append(sc)
            res[r][a] = {**summarise(runs), 'worst_date': float(np.mean([x['worst_date'] for x in runs]))}
            print(f"{r:12s} {a:12s} macro {res[r][a]['macro']:.2f} {[round(x, 2) for x in res[r][a]['macro_per_seed']]} acc {res[r][a]['acc']:.2f} worst {res[r][a]['worst_date']:.2f}")
    rng = np.random.default_rng(13); idx = np.where(cov)[0]; boot = {'unseen_date': [], 'ordinary': []}
    pred = {r: {a: np.stack([P[r][a][s].argmax(1).cpu().numpy() for s in SEEDS]) for a in ('head', 'head+memory')} for r in P}
    for _ in range(2000):
        b = rng.choice(idx, len(idx))
        for r in P:
            boot[r].append(100 * np.mean([f1_score(yn[b], pred[r]['head+memory'][i][b], average='macro') -
                                          f1_score(yn[b], pred[r]['head'][i][b], average='macro') for i in range(3)]))
    res['memory_gain'] = {r: dict(diff=res[r]['head+memory']['macro'] - res[r]['head']['macro'], lo=float(np.percentile(v, 2.5)),
                                  hi=float(np.percentile(v, 97.5))) for r, v in boot.items()}
    print(res['memory_gain'])

elif STAGE == 'met':
    num = lambda s: int(re.search(r'(\d+)', s).group(1))
    Z = np.load(FEAT / 'met_clip_features.npz')
    mm = pd.read_csv(FEAT / 'met_manifest.csv')
    pos = {int(i): n for n, i in enumerate(Z['E_id'])}; sel = np.array([pos[num(n)] for n in mm.file_name])
    I, Tx, y = T(Z['E_image'][sel]), T(Z['E_text'][sel]), T(mm.target.values)
    a, q = T(np.where(mm.partition == 'fit')[0]), T(np.where(mm.partition == 'test')[0])
    mem = memory(I[a], y[a], I[q], 7); X = cat(I, Tx); res = {'train': len(a), 'test': len(q)}
    hs = [fit_head(X[a], y[a], X[q], s, 7) for s in SEEDS]
    res['head'] = summarise([scores(h, y[q]) for h in hs]); res['memory'] = summarise([scores(mem, y[q])])
    res['head+memory'] = summarise([scores(LAM * h + (1 - LAM) * mem, y[q]) for h in hs])
    for k in ('head', 'memory', 'head+memory'): print(f"{k:12s} macro {res[k]['macro']:.2f} {[round(x, 2) for x in res[k]['macro_per_seed']]} acc {res[k]['acc']:.2f} weighted {res[k]['weighted']:.2f}")

elif STAGE == 'test':
    assert not out.exists(), 'The test has already been scored. It is run once.'
    Ti, Tt, dtr = mood('train', CLIPDIR / 'train_features.npz'); Si, St, dte = mood('test', FEAT / 'test_features.npz')
    assert len(dte) == 1502 and not set(dte.group_id) & set(dtr.group_id)
    Ty, Sy = T(dtr.target.values), T(dte.target.values); yn = dte.target.values
    mem = memory(Ti, Ty, Si, 6)
    hs = [fit_head(cat(Ti, Tt), Ty, cat(Si, St), s, 6) for s in SEEDS]
    P = {'head': hs, 'memory': [mem] * 3, 'head+memory': [LAM * h + (1 - LAM) * mem for h in hs], 'ALFRED': []}
    for s in SEEDS:
        z = np.load(RESULTS / f'alfred/seed{s}_test_predictions.npz', allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z['images'])}; o = [pos[i] for i in dte.image]
        assert (z['targets'][o] == yn).all()
        P['ALFRED'].append(T(z['probabilities'][o]))
    res = {k: summarise([scores(p, Sy, NAMES) for p in (v[:1] if k == 'memory' else v)]) for k, v in P.items()}
    pred = {k: np.stack([p.argmax(1).cpu().numpy() for p in v]) for k, v in P.items()}
    rng = np.random.default_rng(13); pairs = [('head+memory', 'ALFRED'), ('head+memory', 'head'), ('head', 'ALFRED')]
    boot = {p: [] for p in pairs}
    for _ in range(2000):
        b = rng.integers(0, 1502, 1502)
        f = {k: np.mean([f1_score(yn[b], pred[k][i][b], average='macro') for i in range(3)]) for k in pred}
        for a_, b_ in pairs: boot[(a_, b_)].append(100 * (f[a_] - f[b_]))
    res['paired_bootstrap'] = {f'{a_} - {b_}': dict(diff=res[a_]['macro'] - res[b_]['macro'], lo=float(np.percentile(v, 2.5)),
                                                    hi=float(np.percentile(v, 97.5)),
                                                    wins_all_seeds=bool(all(x > y_ for x, y_ in zip(res[a_]['macro_per_seed'], res[b_]['macro_per_seed']))))
                               for (a_, b_), v in boot.items()}
    np.savez_compressed(OUTPUTS / 'final_test_predictions.npz', images=dte.image.values.astype(str), targets=yn,
                        **{f"{k.replace('+', '_')}_seed{s}": P[k][i].cpu().numpy() for k in P for i, s in enumerate(SEEDS)})
    for k in P: print(f"{k:12s} macro {res[k]['macro']:.2f} {[round(x, 2) for x in res[k]['macro_per_seed']]} acc {res[k]['acc']:.2f} weighted {res[k]['weighted']:.2f}")
    print(json.dumps(res['paired_bootstrap'], indent=1))

out.write_text(json.dumps(res, indent=2), encoding='utf-8')
