# Methodology

*(From `Version 1/code_v1/README.md` and `Version 2/CHANGES_v2.md`.)*

## Groups
* Class-0 rows are grouped by their Bonn source segment (found by exact matching).
* Class 1–3 rows (continuous multichannel recordings) are grouped into contiguous 200-row blocks.

## Protocols
* **random** — stratified row-level split (as in prior work).
* **blocked** — whole groups held out; final 20% of each seizure recording as test; 50-row purge between training and
  test rows of the same recording.
* **rolling origin** — five-fold rolling-origin evaluation (primary estimate in Version 2).
* **Nested tuning** — inner three-fold CV follows the same protocol as the outer split.

## Model selection
The primary model is chosen without any test block: best mean macro-F1 in nested purged blocked 5×2 CV → LightGBM on
raw values for both tasks.

## Features and models
73 features (16 raw + 57 engineered); classical models and deep models (BiLSTM and others, tuned on a 12-point grid).

## Uncertainty and statistics
Cluster bootstrap (Bonn segments / 50-row chunks) besides the naive row bootstrap; rolling-origin t-intervals;
corrected t-tests with Holm adjustment across 72 comparisons; Friedman/Nemenyi (descriptive only).

## Further analyses
Feature-group ablation, SHAP (descriptive LightGBM with default settings), shortcut stumps, leakage block-size curve,
nearest-neighbour leakage, recording-fingerprint confound, partial CHB-MIT provenance search, windowed analysis,
CPU footprint profiling.

Pipeline stages: `provenance leakage holdout holdout_extra cv stats ablation shap windowed edge figures`. Seeds fixed
(`SEED = 42`).
