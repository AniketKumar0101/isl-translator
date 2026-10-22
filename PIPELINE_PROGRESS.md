# Real-Data Training Pipeline — Progress Log

**Goal:** Train Aniket's ISL translator on real INCLUDE videos (17 signs), evaluate on a
contamination-free official split, for the Code for a Billion submission.
**Started:** 2026-10-22 ~11:50 UTC (agent 1) · **Resumed:** 2026-10-22 ~12:35 UTC (agent 2, after agent 1 died)

## Stage checklist
- [x] Recon: INCLUDE metadata (parquet) — 17 labels selected (Greetings + Pronouns)
- [x] `data/split_map.json` — 358 unique videos → train=278 / val=25 / test=55, zero cross-split overlap (contaminated dups assigned to train)
- [x] `src/data/build_official_splits.py` — builds dataset.npz from split_map (no random split)
- [x] `DATASET_ATTRIBUTION.md` — CC-BY-4.0 attribution (INCLUDE, AI4Bharat, Zenodo 4010759)
- [~] Downloads: Greetings_1of2.zip ✓ · Greetings_2of2.zip ✓ · Pronouns_2of2.zip ✓ (946,837,765 B, testzip OK) · Pronouns_1of2.zip ~ (fresh re-download in progress — old partial had 52.9 MB resume-misalignment, deleted)
- [x] `unzip -t` integrity: both Greetings zips OK; Pronouns_2of2 OK
- [x] Unzip → `data/raw/<clean-label>/`: 9 greeting labels, 190 videos ✓ (matches split_map)
- [x] `config.yaml` labels updated to 17-label vocab
- [x] Landmark extraction pass 1: 190 greeting videos → .npy ✓ (rewrote extractor for MediaPipe Tasks API; installed libegl1/libgl1/libgles2)
- [x] Landmark extraction pass 2: 168 pronoun videos → .npy ✓ — total 358/358, 0 failures, 0 all-zero sequences
- [x] `build_official_splits.py` → `data/processed/dataset.npz` (train 258 / val 33 / test 67) + `labels.json` ✓
- [x] Train: Transformer won (69.7% val) over LSTM (63.6% val) and aggressive-aug (collapsed to 18% — reverted to gentle aug); `checkpoints/best.pt` = transformer, 17 classes
- [x] Evaluate on official held-out test split: **70.1% accuracy (67 samples, chance = 5.9%)**, zero train/test video overlap verified
- [x] `reports/confusion_matrix.png`, `reports/training_curves.png` saved
- [x] `DATASET_ATTRIBUTION.md` present (CC-BY-4.0, INCLUDE/AI4Bharat/Zenodo 4010759)

## Final results (2026-10-22 ~13:15 UTC)
- TEST ACCURACY: **70.1%** (47/67) on held-out test, 17 real ISL signs
- Per-class recall: hello 100%, thank you 100%, how are you 100%, alright 100%, pleased 100%, she 100%, it 100%, good afternoon 83%, you 75%, good night 75%, they 75%, good morning 67%, you all 67%, we 50%, he 25%, i 33%, **good evening 0% (0/7) — confused with other good-X greetings**
- Note: HF parquet listed 126 videos in >1 split (42 in train+test); split_map dedups to 358 unique videos with greedy per-label balancing; test has zero overlap with train.
- [ ] Pronoun zips → populate → continue extraction (skip-if-exists)
- [ ] `build_official_splits.py` → `data/processed/dataset.npz` + `labels.json`
- [ ] Train (`src/train.py`) → `checkpoints/`
- [ ] Evaluate on official test split → `reports/confusion_matrix.png`, accuracy numbers

## Notes
- Extractor already handles *.MOV (uppercase) and skips existing .npy — no patch needed.
- HF parquet had 126 videos listed in >1 split (42 in train+test); dedup rule keeps test clean.
- Per-label video counts (parquet): good morning 42, hello 42, you all 42, thank you 42, i 42, it 42, good afternoon 22, we/she/good evening/alright/he/how are you/good night/you/pleased/they 21 each.
