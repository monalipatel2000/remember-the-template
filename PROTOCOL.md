> Pre-registration record, copied unchanged from the authors' working folder. Folder names in the text refer to that
> folder: `clip_interaction_20261019/prepared` = `features/`, `predictive_architecture_20261018/runs/...alfred` =
> `results/alfred/`, `met_meme_20261020_amp_repair/prepared/membership.csv` = `data/met_meme_partition.csv`,
> `order_leakage_paper_20261112` = the earlier study that defined the archive-date folds. `extract_test.py` is now
> `src/extract_clip_features.py`.

# Exemplar-memory CLIP classifier: frozen protocol for the final evaluation

Frozen 12 November 2026, after the pilot (`pilot.py`, `pilot_results.json`) and before any
script below was run. The pilot's learned attention model failed its own rule and is
reported as a negative result. The model evaluated here is the pilot's control that won.

## Model (all values fixed at the pilot's settings; nothing is tuned after this point)

- Frozen CLIP ViT-L/14 image and OCR-text embeddings, L2-normalised.
- **Head**: the pilot `mlp` arm. 1536-384-6, ReLU, dropout .3, AdamW lr 5e-4, weight decay
  1e-2, one-cycle schedule, 40 epochs, batch 256, label smoothing .1. No early stopping.
- **Exemplar memory**: the K=16 training items nearest to the query in CLIP image cosine
  vote with weights softmax(similarity / 0.05).
- **Prediction**: 0.5 x head probabilities + 0.5 x memory probabilities.
- Seeds 13, 29, 47.

## Evaluations

1. **Known validation (1,503 rows), `final_eval.py val`.** Ablation: image-only, text-only
   and image+text heads; memory keyed on image, on text, and on both; each combination.
   Sensitivity to K, mixing weight and temperature is shown for information only. The
   reported model keeps K=16, 0.5, 0.05 whatever the sensitivity grid shows.
2. **Unseen dates and ordinary folds on the 7,000 training rows, `final_eval.py date`.**
   Same nine dates, purging and folds as `order_leakage_paper_20261112`. Memory is the
   fold's training pool.
3. **MET-Meme English, seven sentiment categories, `final_eval.py met`.** Existing
   partition from `met_meme_20261020_amp_repair/prepared/membership.csv`: train and memory
   on `fit` (2,760), score `test` (591). `stop` is unused because there is no early stopping.
4. **MOOD grouped test (1,502 rows), `final_eval.py test`, run once.** CLIP features for
   test rows are extracted by `extract_test.py` (labels not read there). Train and memory:
   the 7,000 training rows. Comparator: the saved matched ALFRED test predictions
   (`predictive_architecture_20261018/runs/tails_seed{13,29,47}_alfred`), same rows.
   Primary metric macro-F1; also accuracy, weighted F1, per-class F1. Paired row bootstrap
   (2,000 resamples, seed 13) on the seed-mean macro-F1 difference.

## Reading the results, decided in advance

- The model "improves on ALFRED" if test macro-F1 is higher in all three seeds and the
  bootstrap interval of the difference excludes zero.
- The memory "adds to the head" if head+memory beats the head alone on test in all three
  seeds with an interval excluding zero. If not, the paper says the memory does not help.
- Unseen-date results are reported beside the test results whatever they show, including
  if the memory gives no gain or a loss there.
- The test script refuses to run if its output file exists. No rerun, no change of
  hyperparameters, seeds or rows after the test is scored.

## Known limits, to be stated in the paper

The ALFRED comparator is the local partial fine-tuning version, not the published
configuration. The grouped test was used earlier to score ALFRED variants and its labels
were read once in aggregate for metadata baselines. Validation rows were used in many
earlier studies. CLIP ViT-L/14 is a larger pretrained backbone than ALFRED's encoders.

## Amendment 1: two published CLIP-based baselines (12 November 2026, before they were run)

Added after the EMC test was scored. EMC itself is not rerun or changed.

- **MemeCLIP** (Shah et al., EMNLP 2024) and **Hate-CLIPper** (Kumar and Nandakumar, 2022),
  reimplemented from the official repositories on frozen CLIP ViT-L/14, which is how both
  train their heads. `extract_unprojected.py` computes the inputs they use (MemeCLIP:
  projected image feature and unprojected text feature, unnormalised; Hate-CLIPper:
  unprojected pooled outputs of both towers). Labels are not read during extraction.
- Hyperparameters are the repositories' own, not tuned: MemeCLIP map_dim 1024, adapter
  reduction 4, residual ratio 0.2, dropouts .1/.4/.2, cosine classifier scale 30
  initialised from class prompts, AdamW lr 1e-4, weight decay 1e-4, batch 16, 10 epochs.
  Hate-CLIPper cross fusion, map_dim 32, dropouts .2/.4/.1, AdamW lr 1e-4, weight decay
  1e-4, batch 16, 20 epochs, gradient clipping 0.1.
- Necessary adaptations to six-class emotion, stated in the paper: Hate-CLIPper's single
  logit with binary cross-entropy becomes six logits with cross-entropy; MemeCLIP's class
  prompts become "a photo of a {emotion} meme."; both repositories keep the epoch with the
  best validation AUROC, here the best validation macro-F1.
- Seeds 13/29/47. Validation and test macro-F1, weighted F1 and accuracy are reported for
  the selected epoch, with the same paired bootstrap against EMC. One run
  (`baselines.py` refuses to run if its output exists). Reported whatever it shows,
  including if a baseline beats EMC.

## Amendment 2: revision experiments after external review (17 November 2026, before they were run)

EMC is not changed or rerun. Script: `revision.py`. Each stage refuses to run twice.

1. **Image-only CLIP head on the test split (`revision.py imgtest`).** The same head as
   before with the image embedding only, seeds 13/29/47, trained on the 7,000 training
   rows, scored once on the 1,502 test rows. Paired bootstrap of EMC minus image-only
   head. All test comparisons are also re-analysed with a bootstrap that resamples
   exact-image groups (from saved predictions; no model is rerun).
   Reading, decided in advance: EMC "beats the strongest simple baseline" only if it is
   ahead of the image-only head in all three seeds and the group-bootstrap interval
   excludes zero. Otherwise the paper says EMC is not shown to beat an image-only head.
2. **Template-removal memory ablation (`revision.py template`).** Queries: the 1,503
   validation rows. Heads are trained once on the training rows and kept fixed. Template
   references of a query are defined without labels: training memes whose CLIP image
   cosine to the query is at least 0.90 (primary; 0.85 and 0.95 as sensitivity). Label-free
   statistics were inspected to choose this form: connected components chain into one
   giant family below 0.90, so a per-query neighbourhood is used. Memory conditions:
   (A) full memory; (B) template references removed; (C) control: for each query the same
   number of non-template memes with the same class counts removed at random (seed 13).
   Reported for all queries and separately for queries with and without template
   references: head, and head + memory under A, B, C. The share of the memory gain "due to
   template references" is (A - B) / (A - head), reported only if A - head is positive.
   Limit to be stated: the template definition uses the same encoder as retrieval.
3. **MET-Meme with correctly decoded text (`revision.py met`).** The earlier cache decoded
   GB18030 text as UTF-8 with replacement, altering 22 rows (14 train, 3 stop, 5 test).
   Text embeddings are recomputed from GB18030-decoded text; everything else is unchanged.
   The original result stays in `final_met.json` and is reported beside the corrected one.

## Amendment 3: template-disjoint folds (17 November 2026, before any model was scored on them)

**Template families, built without labels (`template_families.py`).** Complete-linkage
clustering of all 10,005 MOOD images on features independent of CLIP: mean patch token of
the ImageNet-21k ViT-B/16 cached for ALFRED, centred and L2-normalised. A family is a
cluster in which every pair has cosine at least 0.70 (primary; 0.50 and 0.90 as
sensitivity). The threshold was chosen from label-free statistics only: 0.70 is just above
the 99.9th percentile of pairwise cosine (0.66), and a 64-bit difference hash, a second
independent signal, differs by 8.9 bits on average within families against 31.8 for random
pairs. Contact sheets of 6 largest and 18 random families were viewed before any scoring;
most are one template under different captions or near-copies of one meme, and a few
contain an unrelated image. Families are merged with exact-duplicate groups.

**Evaluation (`template_split.py`, run once).** Rows: the 8,503 training and validation
rows. The test split is not used. Two 5-fold schemes over the same rows,
`StratifiedGroupKFold(5, shuffle=True, random_state=13)`: (a) ordinary, grouped by
exact-duplicate group; (b) template-disjoint, grouped by family, so no family has members
in both training and evaluation folds, and none in the memory. Models, fixed settings as
before, seeds 13/29/47: image-only head, image+text head, memory alone, EMC. Pooled
macro-F1 and accuracy. Paired bootstrap resampling families (2,000, seed 13) for EMC minus
each head in each scheme.

**Reading, decided in advance.** The memory "helps beyond shared templates" only if, under
template-disjoint folds, EMC beats both heads in all three seeds and both family-bootstrap
intervals exclude zero. Otherwise the paper says the memory's benefit is not shown to
extend beyond shared templates, and the title and abstract are changed to match.

## Amendment 4: three designs for unfamiliar templates (17 November 2026, before any was run)

Goal: beat the image-only CLIP head where it matters, on memes whose template family is
absent from training. Script `novel_template.py`, run once. Folds: the template-disjoint
folds of Amendment 3 (families at 0.70) on the 8,503 training and validation rows. Head
settings, seeds and bootstrap as before. All use frozen CLIP features.

- **A. Language-anchored head.** The head also receives the query's similarities to fixed
  emotion prompts, which carry knowledge that does not depend on any training template:
  CLIP text embeddings of four prompt forms ("a meme expressing {e}", "a person feeling
  {e}", "a reaction image showing {e}", "{e}") for the six emotions; image-to-prompt and
  OCR-text-to-prompt cosines, multiplied by 20, appended to the image+text input.
- **B. Template-balanced training.** The image+text head is trained with each example
  weighted by 1 / (size of its template family within the training fold), normalised to
  mean 1, so large families cannot dominate.
- **C. Template-residual head.** The head receives the image embedding, the centroid of its
  template family (computed without labels over all rows, the query's fold included), the
  residual between the two, and the text embedding, so that what is specific to a meme is
  separated from what it shares with its template.
- **ABC.** All three together.
- Reference arms: image-only head, image+text head, and zero-shot prompt scoring alone.

**Rule, decided in advance.** A design passes only if its macro-F1 beats the image-only
head in all three seeds and the family-bootstrap interval of the difference excludes zero.
If exactly one or more pass, the one with the largest margin is confirmed once on the test
split (`novel_template.py confirm`): trained on the 7,000 training rows, reported on all
test rows and on the test memes whose family has no member in training, against an
image-only head trained the same way. If none passes, no test run is made, the designs are
reported as negative results, and no further designs are tried on these folds in this
round.

### Amendment 4, confirmation step (written after the fold results, before the test run)

Fold results: C template-residual +2.63 over the image-only head (interval 1.55 to 3.67,
all seeds) and ABC +2.50 pass; A, B fail. C has the largest margin and is confirmed.
`novel_confirm.py`, run once: C and the image-only and image+text heads are trained on the
7,000 training rows, seeds 13/29/47. Family centroids are computed without labels over all
10,005 images (families at 0.70 merged with exact-duplicate groups). Reported on all 1,502
test rows and on the template-novel subset (test memes whose family has no training
member). **Confirmed** only if, on the template-novel subset, C beats the image-only head
in all three seeds and the family-bootstrap interval excludes zero. Also reported, as a
diagnostic and not a criterion: the same comparison split by whether a test meme's family
has other members among the unlabelled images, since a singleton's centroid is itself.
C is transductive: it uses other unlabelled memes of the same family at inference. The
paper must say so.
