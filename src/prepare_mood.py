"""Build features/mood_manifest.csv: the released grouped split (data/mood_grouped_split.csv) joined with the MOOD
annotation files (train.csv, val.csv, test.csv with columns image, sentiment, ocr_text, target, emotion).
Row order is kept, because the scripts train on rows in this order. Use --verify-images to check every image's SHA-256."""
import hashlib, sys
import pandas as pd
from paths import *

LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
frames = []
for split in ('train', 'val', 'test'):
    f = pd.read_csv(MOOD_DIR / f'{split}.csv', dtype=str, keep_default_na=False)
    f['original_split'] = split; f['original_row_id'] = range(len(f)); frames.append(f)
src = pd.concat(frames, ignore_index=True)
assert len(src) == 10005 and src.image.is_unique, 'Expected the 10,005-row MOOD release'
assert (src.target.astype(int).map(dict(enumerate(LABELS))) == src.emotion).all()

split = pd.read_csv(DATA / 'mood_grouped_split.csv', dtype=str, keep_default_na=False)
src = src.set_index('image').loc[split.image].reset_index()
for c in ('original_split', 'original_row_id'):
    assert (src[c].astype(str).values == split[c].values).all(), f'MOOD files do not match the released split ({c})'
man = split[['image']].assign(item_id=split.image)
for c in ('sentiment', 'ocr_text', 'target', 'emotion'): man[c] = src[c].values
for c in ('original_split', 'original_row_id', 'file_sha256', 'rgb224_sha256', 'group_id', 'group_size', 'new_split'): man[c] = split[c].values
man = man[['item_id', 'image', 'sentiment', 'ocr_text', 'target', 'emotion', 'original_split', 'original_row_id',
           'file_sha256', 'rgb224_sha256', 'group_id', 'group_size', 'new_split']]

if '--verify-images' in sys.argv:
    bad = [n for n, h in zip(man.image, man.file_sha256) if hashlib.sha256((IMAGES / n).read_bytes()).hexdigest() != h]
    assert not bad, f'{len(bad)} images differ from the release, e.g. {bad[:5]}'
    print('all 10,005 image hashes match')
man.to_csv(MANIFEST, index=False)
print('wrote', MANIFEST, man.new_split.value_counts().to_dict())
