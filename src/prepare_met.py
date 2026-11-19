"""Build features/met_manifest.csv for MET-Meme English: the released partition (data/met_meme_partition.csv) joined with
the seven sentiment categories in label_E.csv inside the Kaggle archive. target = category number - 1
(0 happiness, 1 love, 2 anger, 3 sorrow, 4 fear, 5 hate, 6 surprise). Row order is kept."""
import io, re, zipfile
import pandas as pd
from paths import *

with zipfile.ZipFile(MET_ZIP) as z: lab = pd.read_csv(io.BytesIO(z.read('label_E.csv')), encoding_errors='replace')
lab.columns = [c.strip() for c in lab.columns]
name_col = next(c for c in lab.columns if 'file' in c.lower() or 'name' in c.lower())
cat = lab.set_index(name_col)['sentiment category'].astype(str)
part = pd.read_csv(DATA / 'met_meme_partition.csv')
missing = set(part.file_name) - set(cat.index); assert not missing, f'{len(missing)} files not in label_E.csv'
part['target'] = [int(re.match(r'\s*(\d+)', cat[n]).group(1)) - 1 for n in part.file_name]
assert set(part.target) == set(range(7))
part.to_csv(FEAT / 'met_manifest.csv', index=False)
print('wrote', FEAT / 'met_manifest.csv', part.partition.value_counts().to_dict())
