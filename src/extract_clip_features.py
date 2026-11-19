"""Frozen CLIP ViT-L/14 image and OCR-text embeddings (L2-normalised) for the MOOD grouped splits.
Reads images and OCR text only; labels are not loaded. Usage: extract_clip_features.py [train] [val] [test]"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from paths import *
man = pd.read_csv(MANIFEST,
                  usecols=['image', 'ocr_text', 'new_split', 'group_id'], keep_default_na=False)
model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REV).cuda().eval()
proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV)


def embed(frame):
    out = {'image': [], 'text': []}
    for a in range(0, len(frame), 32):
        pics = []
        for name in frame.image.iloc[a:a + 32]:
            with Image.open(IMAGES / name) as im: pics.append(im.convert('RGB'))
        batch = proc(text=[str(t) for t in frame.ocr_text.iloc[a:a + 32]], images=pics, return_tensors='pt', padding=True,
                     truncation=True, max_length=77)
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            o = model(**batch, return_dict=True)
        for k in out: out[k].append(F.normalize(getattr(o, f'{k}_embeds').float(), dim=-1).cpu().numpy())
        print(a, flush=True)
    return {k: np.vstack(v) for k, v in out.items()}


import sys
splits = sys.argv[1:] or ['train', 'val', 'test']
for split in splits:
    rows = man[man.new_split == split].reset_index(drop=True)
    e = embed(rows)
    np.savez_compressed(FEAT / f'{split}_features.npz', images=rows.image.values.astype(str), groups=rows.group_id.values.astype(str), **e)
    print('saved', split, e['image'].shape)
