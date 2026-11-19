"""Revision experiments (PROTOCOL.md, Amendment 2). Stages: imgtest | template | met. Each runs once."""
import io, json, re, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from paths import *
DEV = 'cuda'
SEEDS, EPOCHS, BATCH, K, LAM, TAU, D = (13, 29, 47), 40, 256, 16, .5, .05, 384
STAGE = sys.argv[1]
OUT = OUTPUTS / f'revision_{STAGE}.json'
assert not OUT.exists(), f'{OUT.name} exists. This stage is run once.'
T = lambda a: torch.tensor(np.asarray(a), device=DEV)
CLIPDIR = FEAT
man = pd.read_csv(MANIFEST)


def fit_head(X, y, Xq, seed, ncls):
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


def vote(sim, y, ncls):
    s, idx = sim.topk(K, dim=1)
    return torch.zeros(len(sim), ncls, device=DEV).scatter_add_(1, y[idx], F.softmax(s / TAU, 1))


def mood(split, file):
    d = man[man.new_split == split].reset_index(drop=True)
    z = np.load(file, allow_pickle=True)
    p = {str(n): i for i, n in enumerate(z['images'])}; o = np.array([p[i] for i in d.image])
    return T(z['image'][o]), T(z['text'][o]), d


mf = lambda y, p: 100 * f1_score(y, p, average='macro')
cat = lambda a, b: torch.cat([a, b], 1)

if STAGE == 'imgtest':
    Ti, Tt, dtr = mood('train', CLIPDIR / 'train_features.npz'); Si, St, dte = mood('test', FEAT / 'test_features.npz')
    Ty = T(dtr.target.values); yn = dte.target.values
    img = np.stack([fit_head(Ti, Ty, Si, s, 6).argmax(1).cpu().numpy() for s in SEEDS])
    E = np.load(recorded('final_test_predictions.npz'), allow_pickle=True); B = np.load(recorded('baseline_test_predictions.npz'), allow_pickle=True)
    assert (E['images'] == dte.image.values.astype(str)).all() and (B['images'] == E['images']).all()
    pred = {'image-only head': img,
            'EMC': np.stack([E[f'head_memory_seed{s}'].argmax(1) for s in SEEDS]),
            'image+text head': np.stack([E[f'head_seed{s}'].argmax(1) for s in SEEDS]),
            'ALFRED': np.stack([E[f'ALFRED_seed{s}'].argmax(1) for s in SEEDS]),
            'MemeCLIP': np.stack([B[f'MemeCLIP_seed{s}'].argmax(1) for s in SEEDS]),
            'Hate-CLIPper': np.stack([B[f'HateCLIPper_seed{s}'].argmax(1) for s in SEEDS])}
    res = {k: dict(macro=float(np.mean([mf(yn, v[i]) for i in range(3)])), macro_per_seed=[mf(yn, v[i]) for i in range(3)],
                   acc=float(np.mean([100 * (v[i] == yn).mean() for i in range(3)])),
                   weighted=float(np.mean([100 * f1_score(yn, v[i], average='weighted') for i in range(3)]))) for k, v in pred.items()}
    g = dte.group_id.astype('category').cat.codes.values; members = [np.where(g == c)[0] for c in range(g.max() + 1)]
    rng = np.random.default_rng(13); others = [k for k in pred if k != 'EMC']; row, grp = {k: [] for k in others}, {k: [] for k in others}
    for _ in range(2000):
        b1 = rng.integers(0, len(yn), len(yn)); b2 = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        for b, store in ((b1, row), (b2, grp)):
            f = {k: np.mean([f1_score(yn[b], pred[k][i][b], average='macro') for i in range(3)]) for k in pred}
            for k in others: store[k].append(100 * (f['EMC'] - f[k]))
    res['EMC_minus'] = {k: dict(diff=res['EMC']['macro'] - res[k]['macro'],
                                row=[float(np.percentile(row[k], 2.5)), float(np.percentile(row[k], 97.5))],
                                group=[float(np.percentile(grp[k], 2.5)), float(np.percentile(grp[k], 97.5))],
                                emc_wins_all_seeds=bool(all(a > b for a, b in zip(res['EMC']['macro_per_seed'], res[k]['macro_per_seed']))))
                        for k in others}
    res['test_groups'] = len(members)
    for k in pred: print(f"{k:16s} macro {res[k]['macro']:.2f} {[round(x, 2) for x in res[k]['macro_per_seed']]} acc {res[k]['acc']:.2f} weighted {res[k]['weighted']:.2f}")
    print(json.dumps(res['EMC_minus'], indent=1))

elif STAGE == 'template':
    Ti, Tt, dtr = mood('train', CLIPDIR / 'train_features.npz'); Vi, Vt, dva = mood('val', CLIPDIR / 'val_features.npz')
    Ty, Vy = T(dtr.target.values), T(dva.target.values); ytr, yn = dtr.target.values, dva.target.values
    S = Vi @ Ti.T; Sn = S.cpu().numpy()
    heads = {'image+text': [fit_head(cat(Ti, Tt), Ty, cat(Vi, Vt), s, 6) for s in SEEDS], 'image': [fit_head(Ti, Ty, Vi, s, 6) for s in SEEDS]}
    res = {}
    for th in (.90, .85, .95):
        tmpl = Sn >= th; n_ref = tmpl.sum(1); has = n_ref > 0
        rng = np.random.default_rng(13); ctrl = np.zeros_like(tmpl)
        for q in np.where(has)[0]:
            for c in np.unique(ytr[tmpl[q]]):
                need = int((ytr[tmpl[q]] == c).sum()); pool = np.where((ytr == c) & ~tmpl[q])[0]
                ctrl[q, rng.choice(pool, min(need, len(pool)), replace=False)] = True
        mem = {'A_full': vote(S, Ty, 6), 'B_no_template': vote(S.masked_fill(T(tmpl), -2.), Ty, 6), 'C_matched_removal': vote(S.masked_fill(T(ctrl), -2.), Ty, 6)}
        r = {'queries_with_template_refs': int(has.sum()), 'mean_refs_when_present': float(n_ref[has].mean()), 'refs_removed_total': int(tmpl.sum()),
             'control_removed_total': int(ctrl.sum()),
             'template_ref_label_agreement': float(100 * np.mean([(ytr[tmpl[q]] == yn[q]).mean() for q in np.where(has)[0]]))}
        for hname, hp in heads.items():
            for sub, m_ in (('all', np.ones(len(yn), bool)), ('with_refs', has), ('without_refs', ~has)):
                e = {'head': float(np.mean([mf(yn[m_], p.argmax(1).cpu().numpy()[m_]) for p in hp])),
                     'head_acc': float(np.mean([100 * (p.argmax(1).cpu().numpy()[m_] == yn[m_]).mean() for p in hp]))}
                for k, mp in mem.items():
                    pr = [(LAM * p + (1 - LAM) * mp).argmax(1).cpu().numpy() for p in hp]
                    e[k] = float(np.mean([mf(yn[m_], x[m_]) for x in pr])); e[k + '_acc'] = float(np.mean([100 * (x[m_] == yn[m_]).mean() for x in pr]))
                e['memory_only_A'] = float(mf(yn[m_], mem['A_full'].argmax(1).cpu().numpy()[m_]))
                e['memory_only_B'] = float(mf(yn[m_], mem['B_no_template'].argmax(1).cpu().numpy()[m_]))
                r[f'{hname}|{sub}'] = e
            a = r[f'{hname}|all']
            r[f'{hname}|share_due_to_template'] = (a['A_full'] - a['B_no_template']) / (a['A_full'] - a['head']) if a['A_full'] > a['head'] else None
        res[f'threshold_{th}'] = r
        print(th, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})
        for k, v in r.items():
            if isinstance(v, dict): print('   ', f'{k:26s}', {a: round(b, 2) for a, b in v.items() if not a.endswith('_acc')})

elif STAGE == 'met':
    from transformers import CLIPModel, CLIPProcessor
    num = lambda s: int(re.search(r'(\d+)', s).group(1))
        with zipfile.ZipFile(MET_ZIP) as z:
        txt = pd.read_csv(io.BytesIO(z.read('E_text.csv')), encoding='gb18030')
    text = {num(a): ('' if pd.isna(b) else str(b)) for a, b in zip(txt.iloc[:, 0], txt.iloc[:, 1])}
    Z = np.load(FEAT / 'met_clip_features.npz')
    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REV).cuda().half().eval(); proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV)
    ids = Z['E_id']; new = []
    for i in range(0, len(ids), 64):
        tk = proc.tokenizer([text.get(int(n), '') for n in ids[i:i + 64]], padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
        with torch.no_grad(): new.append(F.normalize(model.text_projection(model.text_model(**tk).pooler_output).float(), dim=-1).cpu().numpy())
    new = np.vstack(new); cos = (new * Z['E_text']).sum(1)
    np.savez_compressed(OUTPUTS / 'met_text_gb18030.npz', E_id=ids, E_text=new)
    mm = pd.read_csv(FEAT / 'met_manifest.csv')
    pos = {int(i): n for n, i in enumerate(ids)}; sel = np.array([pos[num(n)] for n in mm.file_name])
    I, y = T(Z['E_image'][sel]), T(mm.target.values); yn = mm.target.values
    a, q = T(np.where(mm.partition == 'fit')[0]), T(np.where(mm.partition == 'test')[0]); yq = yn[mm.partition == 'test']
    mem = vote(I[q] @ I[a].T, y[a], 7)
    res = {'text_rows_changed': int((cos < .9999).sum()), 'min_cosine_old_new': float(cos.min())}
    for name, Tx in (('original_cache', T(Z['E_text'][sel])), ('gb18030_corrected', T(new[sel]))):
        X = cat(I, Tx); hs = [fit_head(X[a], y[a], X[q], s, 7) for s in SEEDS]
        res[name] = {'head': [mf(yq, h.argmax(1).cpu().numpy()) for h in hs], 'memory': mf(yq, mem.argmax(1).cpu().numpy()),
                     'head+memory': [mf(yq, (LAM * h + (1 - LAM) * mem).argmax(1).cpu().numpy()) for h in hs],
                     'head_acc': [100 * float((h.argmax(1).cpu().numpy() == yq).mean()) for h in hs],
                     'head+memory_acc': [100 * float(((LAM * h + (1 - LAM) * mem).argmax(1).cpu().numpy() == yq).mean()) for h in hs]}
        print(name, {k: (round(float(np.mean(v)), 2)) for k, v in res[name].items()})
    print({k: v for k, v in res.items() if not isinstance(v, dict)})

OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
