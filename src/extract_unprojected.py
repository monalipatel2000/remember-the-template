"""Unprojected CLIP ViT-L/14 features for the MemeCLIP and Hate-CLIPper baselines. Labels are not loaded."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from paths import *
man = pd.read_csv(MANIFEST,
                  usecols=['image', 'ocr_text', 'new_split'], keep_default_na=False)
model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REV).cuda().eval()
proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV)
out = {'vision_pooled': [], 'text_pooled': [], 'image_projected': []}
for a in range(0, len(man), 48):
    pics = []
    for name in man.image.iloc[a:a + 48]:
        with Image.open(IMAGES / name) as im: pics.append(im.convert('RGB'))
    b = proc(text=[str(t) for t in man.ocr_text.iloc[a:a + 48]], images=pics, return_tensors='pt', padding=True, truncation=True, max_length=77)
    b = {k: v.cuda() for k, v in b.items()}
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        v = model.vision_model(pixel_values=b['pixel_values']).pooler_output
        t = model.text_model(input_ids=b['input_ids'], attention_mask=b['attention_mask']).pooler_output
        p = model.visual_projection(v)
    for k, x in zip(out, (v, t, p)): out[k].append(x.float().cpu().numpy())
    if a % 960 == 0: print(a, flush=True)
# Class-prompt initialisation for MemeCLIP: encode_text, normalise, map through the visual projection transpose, normalise.
names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
tk = proc.tokenizer([f'a photo of a {n} meme.' for n in names], padding=True, return_tensors='pt').to('cuda')
with torch.inference_mode():
    e = torch.nn.functional.normalize(model.get_text_features(**tk) if isinstance(model.get_text_features(**tk), torch.Tensor)
                                      else model.text_projection(model.text_model(**tk).pooler_output), dim=-1)
    init = torch.nn.functional.normalize(e.float() @ model.visual_projection.weight.float(), dim=-1)
np.savez_compressed(FEAT / 'unprojected_features.npz', images=man.image.values.astype(str), split=man.new_split.values.astype(str),
                    memeclip_init=init.cpu().numpy(), **{k: np.vstack(v) for k, v in out.items()})
print({k: np.vstack(v).shape for k, v in out.items()}, init.shape)
