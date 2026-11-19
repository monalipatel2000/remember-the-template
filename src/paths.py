"""Paths and model identifiers shared by every script. Point MOOD_DIR, MOOD_IMAGES and MET_ZIP at your downloads,
either by editing the defaults below or by setting the environment variables of the same name."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'data'          # released split, template families and folds
RESULTS = REPO / 'results'    # outputs recorded for the paper (read-only)
FEAT = REPO / 'features'      # extracted features and the local manifest (not in git)
OUTPUTS = REPO / 'outputs'    # every new run writes here (not in git)

MOOD_DIR = Path(os.environ.get('MOOD_DIR', REPO / 'external/MOOD'))            # holds train.csv, val.csv, test.csv
IMAGES = Path(os.environ.get('MOOD_IMAGES', MOOD_DIR / 'images'))               # holds the 10,005 MOOD image files
MET_ZIP = Path(os.environ.get('MET_ZIP', REPO / 'external/met-meme-v2.zip'))     # Kaggle liaolianfoka/met-meme, version 2
MANIFEST = FEAT / 'mood_manifest.csv'                                           # written by prepare_mood.py

CLIP_ID, CLIP_REV = 'openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41'
VIT_ID, VIT_REV = 'google/vit-base-patch16-224-in21k', 'b4569560a39a0f1af58e3ddaf17facf20ab919b0'

for _p in (FEAT, OUTPUTS): _p.mkdir(exist_ok=True)


def recorded(name):
    """An earlier stage's output: your own run in outputs/ if it exists, otherwise the copy recorded for the paper."""
    return OUTPUTS / name if (OUTPUTS / name).exists() else RESULTS / name
