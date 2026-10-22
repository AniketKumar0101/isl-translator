# Dataset Attribution

This project trains on real Indian Sign Language videos from:

**INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition**
- Creators: AI4Bharat
- Source: Zenodo record [4010759](https://zenodo.org/records/4010759)
- Hugging Face mirror: `ai4bharat/INCLUDE` (split metadata)
- Licence: **CC-BY-4.0**

## Licence obligation

CC-BY-4.0 requires attribution. If you redistribute this model, dataset
derivative, or substantial portions of it, you must:

1. Credit "INCLUDE: A Large Scale Dataset for Indian Sign Language
   Recognition" by AI4Bharat,
2. Link to the licence: https://creativecommons.org/licenses/by/4.0/,
3. Indicate if changes were made.

## What we use

- Subset: the **Greetings** and **Pronouns** categories (17 signs, 358 videos
  after de-duplication) — see `data/split_map.json`.
- We use INCLUDE's official train/val/test file splits (signer-disjoint by
  dataset design). Where the published split files listed the same video in
  more than one split, the video was assigned once with priority
  test > val > train, so no video appears in two of our splits.
- Only derived MediaPipe hand-landmark sequences (30 frames × 126 features)
  are stored in this repo; raw videos are not redistributed.

## Citation

> Sridhar, A., Ganesan, R., Oduru, P., & Chander, T. (2020).
> INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.
> *MM '20: Proceedings of the 28th ACM International Conference on Multimedia.*
