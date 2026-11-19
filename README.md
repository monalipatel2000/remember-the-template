# Remember the Template: Exemplar Retrieval for Meme Emotion Recognition

Code, data splits and recorded results for the paper

> Monali Patel and Kaushal Jani. *Remember the Template: Exemplar Retrieval for Meme Emotion Recognition.*
> Department of Computer Engineering, Indus University, Ahmedabad. Manuscript under review, 2026.

Memes reuse a limited stock of image templates, and the template often signals the emotion. **EMC**
(Exemplar-Memory Classifier) uses this reuse directly. A light head on frozen CLIP ViT-L/14 image and text
embeddings is averaged with a parameter-free memory. In that memory, the 16 training memes nearest in image
space vote with similarity-weighted labels (K = 16, temperature 0.05, weight 0.5). All three values were fixed
before testing. Only the head is trained. You can extend the memory with new labelled memes without retraining,
and each prediction can be traced to the training memes that voted for it.

## Main results (MOOD, duplicate-controlled grouped test split, 1,502 memes, macro-F1, mean of 3 seeds)

| Model | All | Template seen in training (790) | Unfamiliar template (712) |
|---|---|---|---|
| ALFRED (local re-training) | 66.96 | 79.99 | 44.69 |
| Hate-CLIPper (adapted) | 74.22 | 80.20 | 63.16 |
| MemeCLIP (adapted) | 76.74 | 81.01 | 68.38 |
| CLIP head, image only | 76.82 | 80.81 | 67.39 |
| CLIP head, image + text | 76.83 | 80.87 | 67.76 |
| Exemplar memory alone | 73.15 | 80.47 | 59.18 |
| **EMC** | **80.63** | **85.35** | **70.95** |

EMC is ahead of every baseline in every seed. About half of the test memes (53%) share a template family with
the training data, and the memory's gain is clearest on these. On unfamiliar templates the margin over the
image-only head is 3.6 points, but the family-bootstrap interval includes zero (-0.9 to 8.1), so the result is
inconclusive. Under template-disjoint cross-validation the margin falls to 0.6 points, which cannot be
distinguished from zero. All models are much weaker on unfamiliar templates. We recommend reporting both columns.

## What is in this repository

```
data/       released split, template families and template-disjoint folds (no images, captions or labels)
results/    every number in the paper as recorded (JSON) plus per-meme test predictions (.npz)
src/        the scripts that produced them, and emc.py, a small reusable implementation
PROTOCOL.md the evaluation protocol, written before each experiment was run
```

`data/README.md` describes every data file. The resource files are:

- `mood_grouped_split.csv`: train / val / test assignment (7,000 / 1,503 / 1,502). Duplicate images
  (the same file, or the same pixels after resizing to 224 x 224) always stay in one split.
- `template_families_{0.5,0.7,0.9}.csv`: label-free template families. They are built from ImageNet-21k ViT-B/16
  features, which are independent of CLIP, using complete-linkage clustering at cosine 0.50, 0.70 (primary) and 0.90.
- `template_disjoint_folds_0.7.csv`: five folds on the 8,503 train and validation memes, where no template
  family spans two folds.
- `mood_test_template_seen.csv`: for each test meme, whether its template family has a member in training.

## Setup

Python 3.10 or newer and a CUDA GPU (8 GB is enough). Install the packages:

```bash
pip install -r requirements.txt
```

Download the datasets from their original authors. They are not redistributed here.

1. **MOOD**: from the ALFRED repository, https://github.com/shiv6891/ALFRED_MemeEmotionDetection (see its data
   link). Put `train.csv`, `val.csv` and `test.csv` in `external/MOOD/`, and the 10,005 images in `external/MOOD/images/`.
2. **MET-Meme** (only for the MET-Meme table): version 2 of the Kaggle dataset `liaolianfoka/met-meme`, saved as
   `external/met-meme-v2.zip`.

To keep the data elsewhere, set the environment variables `MOOD_DIR`, `MOOD_IMAGES` and `MET_ZIP`, or edit
`src/paths.py`. CLIP and ViT weights are downloaded from Hugging Face at the pinned revisions in `src/paths.py`.

## Reproduce

Run from `src/`. Each step writes to `outputs/`. Scripts that need an earlier output read your copy in
`outputs/` if it exists, and otherwise the copy recorded in `results/`. That lets you check any single table
without rerunning the whole pipeline.

```bash
cd src
python prepare_mood.py --verify-images      # joins the released split with the MOOD files and checks image hashes
python extract_clip_features.py             # CLIP ViT-L/14 embeddings, train / val / test
```

| Paper | Command | Output |
|---|---|---|
| Table 1 (test), Table 2 (per class) | `python final_eval.py test`, `python baselines.py`, `python revision.py imgtest` | `final_test.json`, `baselines.json`, `revision_imgtest.json` |
| Table 3 (ablation), Fig. 2 (sensitivity) | `python final_eval.py val` | `final_val.json` |
| Table 4 (learned reader) | `python pilot.py` | `pilot_results.json` |
| Table 5 (removing near-identical images) | `python revision.py template` | `revision_template.json` |
| Table 6 (template-disjoint folds) | `python extract_vit_features.py`, `python template_families.py build 0.7` (also 0.5, 0.9), `python template_split.py` | `template_split.json` |
| Table 7 (seen vs unfamiliar templates) | `python seen_novel_breakdown.py` (saved predictions only, no GPU) | `seen_novel_breakdown.json` |
| Table 8 (designs for unfamiliar templates) | `python novel_template.py`, `python novel_confirm.py` | `novel_template.json`, `novel_confirm.json` |
| Table 9 (archive dates), Fig. 3 | `python final_eval.py date`, `python make_figures.py` | `final_date.json`, `figures/` |
| Table 10 (MET-Meme) | `python prepare_met.py`, `python extract_met_features.py`, `python final_eval.py met`, `python revision.py met` | `final_met.json`, `revision_met.json` |

`baselines.py` needs `python extract_unprojected.py` first. Table numbers follow the submitted manuscript.

Notes:

- **Run once.** Test-set scripts refuse to overwrite their output file. That was how the evaluation was kept
  honest, as `PROTOCOL.md` records. Delete the file in `outputs/` to rerun.
- **ALFRED.** We retrained ALFRED locally on our split with partial fine-tuning, using the public ALFRED code. We
  did not use its published configuration. We include its saved predictions and run metadata (`results/alfred/`),
  not the training pipeline, and the scripts read those predictions.
- **MemeCLIP and Hate-CLIPper** are reimplemented from their official repositories on the same frozen CLIP
  features. The adaptations to six-class emotion are listed in `PROTOCOL.md`, Amendment 1.
- Seeds are 13, 29 and 47. GPU nondeterminism can move individual head scores by a few tenths of a point. In
  our check of the released code, the validation scores of EMC came out identical, and every ablation arm was
  within 0.3 points of the recorded values.

## Using EMC on your own data

```python
from emc import EMC
model = EMC(seed=13).fit(train_image, train_text, train_labels)   # L2-normalised CLIP embeddings
probs, neighbours = model.predict_proba(test_image, test_text, return_neighbours=True)
model.add_exemplars(new_image, new_labels)                        # grow the memory, no retraining
```

## Citation

```bibtex
@unpublished{patel2026remember,
  author = {Patel, Monali and Jani, Kaushal},
  title  = {Remember the Template: Exemplar Retrieval for Meme Emotion Recognition},
  note   = {Manuscript under review},
  year   = {2026}
}
```

Please also cite MOOD and ALFRED, and MET-Meme if you use it.

## License

Code and the derived data files in `data/` and `results/` are released under the MIT License (see `LICENSE`).
MOOD and MET-Meme remain under their owners' terms.
