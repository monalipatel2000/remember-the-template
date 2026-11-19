# Data files

None of these files contains an image, a caption or an emotion label. To get labels and captions, join with the
original MOOD release by image file name (`src/prepare_mood.py` does this).

| File | Rows | Columns |
|---|---|---|
| `mood_grouped_split.csv` | 10,005 | `image`; `original_split` and `original_row_id`, the meme's position in the MOOD release; `file_sha256` of the image file; `rgb224_sha256` of the image after decoding to RGB and resizing to 224 x 224; `group_id`, the duplicate group (same file or same 224 x 224 pixels); `group_size`; `new_split` (train / val / test). Split built with `StratifiedGroupKFold(20, shuffle=True, random_state=20261017)` over duplicate groups, stratified by label: folds 0-13 train, 14-16 val, 17-19 test. |
| `template_families_{0.5,0.7,0.9}.csv` | 10,005 | `row_id`, `image`, `family`. Complete-linkage clusters of ImageNet-21k ViT-B/16 mean-patch features (centred, L2-normalised); every pair in a family has cosine at least the threshold. Labels were never loaded. 0.70 is the primary threshold. |
| `template_disjoint_folds_0.7.csv` | 8,503 | `image`, `template_group` (families at 0.70 merged with duplicate groups), `fold` (0-4). Train and val rows only. No template group spans two folds. |
| `mood_test_template_seen.csv` | 1,502 | `image`, `template_group`, `template_seen_in_training` (the group has a member among the 7,000 training rows). 790 seen, 712 unfamiliar. |
| `mood_archive_dates.csv` | 10,005 | `image`, `date`: the file date recorded in the MOOD archive, used for the held-out-date folds. |
| `met_meme_partition.csv` | 3,942 | MET-Meme English `file_name`, `group_id` (duplicate group), `partition` (fit 2,760 / stop 591 / test 591). |

The fold assignment in `template_disjoint_folds_0.7.csv` is stratified by label, so it is reproduced from the
labels by `src/template_split.py`. It is also written out here so that others can use the folds directly.
