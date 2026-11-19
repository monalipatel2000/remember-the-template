"""Frozen CLIP ViT-L/14 features for MET-Meme English, as used in the paper. The text file is decoded as UTF-8 with
replacement, exactly as in the first run (Table MET, "original"); revision.py met recomputes the text with GB18030."""
import hashlib, io, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # some MET-Meme files are truncated
from paths import *
num = lambda s: int(re.search(r'(\d+)', s).group(1))
ZP = MET_ZIP
OUTFILE = FEAT / 'met_clip_features.npz'


def extract():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REV).cuda().half().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV)
    out = {}
    with zipfile.ZipFile(ZP) as z:
        text = pd.read_csv(z.open('E_text.csv'), encoding='utf-8', encoding_errors='replace')
        text = {num(a): ('' if pd.isna(b) else str(b)) for a, b in zip(text.iloc[:, 0], text.iloc[:, 1])}
        for lang in 'E':  # the paper uses the English half only
            names = sorted((n for n in z.namelist() if n.startswith(lang) and n.lower().endswith(('.jpg', '.jpeg', '.png'))),
                           key=lambda n: num(Path(n).name))
            ids, img, txt, md5 = [], [], [], []
            for i in range(0, len(names), 64):
                raw = [z.read(n) for n in names[i:i + 64]]
                pics = [Image.open(io.BytesIO(r)).convert('RGB') for r in raw]
                with torch.no_grad():
                    px = proc(images=pics, return_tensors='pt')['pixel_values'].cuda().half()
                    img.append(F.normalize(model.visual_projection(model.vision_model(pixel_values=px).pooler_output).float(), dim=-1).cpu().numpy())
                    if lang == 'E':
                        tk = proc.tokenizer([text.get(num(Path(n).name), '') for n in names[i:i + 64]], padding=True,
                                            truncation=True, max_length=77, return_tensors='pt').to('cuda')
                        txt.append(F.normalize(model.text_projection(model.text_model(**tk).pooler_output).float(), dim=-1).cpu().numpy())
                ids += [num(Path(n).name) for n in names[i:i + 64]]; md5 += [hashlib.md5(r).hexdigest() for r in raw]
                print(lang, i, flush=True)
            out[f'{lang}_id'] = np.array(ids); out[f'{lang}_image'] = np.vstack(img); out[f'{lang}_md5'] = np.array(md5)
            if txt: out[f'{lang}_text'] = np.vstack(txt)
    np.savez_compressed(OUTFILE, **out)


extract()
print('saved', OUTFILE)
