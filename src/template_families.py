"""Label-free template families for MOOD from sources independent of CLIP.
Source 1: ImageNet-21k ViT-B/16 patch features cached for ALFRED (mean of patch tokens, centred, L2-normalised).
Source 2: 64-bit difference hash of the image. Emotion labels are never loaded here.
Usage: template_families.py explore | build <cosine_threshold>"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from paths import *
idx = pd.read_csv(DATA / 'template_families_0.7.csv', usecols=['row_id', 'image'])  # row order used for clustering
assert len(idx) == 10005 and (idx.row_id.values == np.arange(10005)).all()

feat_file = FEAT / 'vit_in21k_meanpatch.npy'
assert feat_file.exists(), 'Run extract_vit_features.py first.'
X = np.load(feat_file); X = X - X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)

hash_file = FEAT / 'dhash64.npy'
if not hash_file.exists():
    H = np.zeros((10005, 64), bool)
    for i, name in enumerate(idx.image):
        with Image.open(IMAGES / name) as im: g = np.asarray(im.convert('L').resize((9, 8), Image.LANCZOS), dtype=np.int16)
        H[i] = (g[:, 1:] > g[:, :-1]).ravel()
    np.save(hash_file, H)
H = np.load(hash_file)

D = pdist(X, 'cosine'); Z = linkage(D, 'complete')
if sys.argv[1] == 'explore':
    print('pairwise cosine quantiles 50/90/99/99.9:', np.round(1 - np.quantile(D[::97], [.5, .1, .01, .001]), 3))
    for th in (.5, .6, .7, .8, .9):
        lab = fcluster(Z, 1 - th, 'distance'); sz = np.bincount(lab)[1:]
        # agreement with the independent hash: mean Hamming distance within families against random pairs
        rng = np.random.default_rng(0); pairs = []
        for c in np.where(sz >= 2)[0][:4000] + 1:
            m = np.where(lab == c)[0]; a, b = rng.choice(m, 2, replace=False); pairs.append((H[a] != H[b]).sum())
        r = rng.integers(0, 10005, (4000, 2)); rand = (H[r[:, 0]] != H[r[:, 1]]).sum(1).mean()
        print(f'th {th}: families {len(sz)}, in families of size>=2: {int(sz[sz >= 2].sum())}, largest {sz.max()}, top {np.sort(sz)[-5:][::-1]}, '
              f'within-family hash distance {np.mean(pairs):.1f} vs random {rand:.1f}')
else:
    th = float(sys.argv[2]); lab = fcluster(Z, 1 - th, 'distance')
    out = idx.assign(family=lab); out.to_csv(OUTPUTS / f'template_families_{th}.csv', index=False)
    sz = out.family.value_counts()
    # Contact sheets for visual checking: the 6 largest families and 18 random families of size 3 to 8.
    rng = np.random.default_rng(13); mid = sz[(sz >= 3) & (sz <= 8)].index.values
    for tag, fams, per in (('largest', sz.index[:6], 12), ('random', rng.choice(mid, 18, replace=False), 6)):
        sheet = Image.new('RGB', (per * 132, len(fams) * 132), 'white')
        for r, f in enumerate(fams):
            for c, name in enumerate(out[out.family == f].image.head(per)):
                with Image.open(IMAGES / name) as im: t = im.convert('RGB'); t.thumbnail((128, 128)); sheet.paste(t, (c * 132 + 2, r * 132 + 2))
        sheet.save(OUTPUTS / f'families_{th}_{tag}.jpg', quality=85)
    print(json.dumps({'threshold': th, 'families': int(len(sz)), 'in_families_ge2': int(sz[sz >= 2].sum()), 'largest': int(sz.max())}))
