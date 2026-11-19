"""ImageNet-21k ViT-B/16 features for the label-free template families: the mean of the 196 patch tokens of the last
hidden state, in the row order of data/template_families_0.7.csv. Independent of CLIP. Labels are not loaded."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from paths import *

idx = pd.read_csv(DATA / 'template_families_0.7.csv', usecols=['row_id', 'image'])
model = ViTModel.from_pretrained(VIT_ID, revision=VIT_REV, attn_implementation='sdpa').eval().cuda()
proc = ViTImageProcessor.from_pretrained(VIT_ID, revision=VIT_REV, use_fast=False)
out = np.zeros((len(idx), 768), np.float32)
with torch.inference_mode():
    for a in range(0, len(idx), 64):
        pics = []
        for name in idx.image.iloc[a:a + 64]:
            with Image.open(IMAGES / name) as im: pics.append(im.convert('RGB'))
        px = proc(images=pics, return_tensors='pt').pixel_values.cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(pixel_values=px, return_dict=True).last_hidden_state
        # stored as float16 tokens first, as in the original cache, then averaged in float32
        out[a:a + len(pics)] = h.float().cpu().numpy().astype(np.float16)[:, 1:].astype(np.float32).mean(1)
        if a % 1280 == 0: print(a, flush=True)
np.save(FEAT / 'vit_in21k_meanpatch.npy', out)
print('saved', out.shape)
