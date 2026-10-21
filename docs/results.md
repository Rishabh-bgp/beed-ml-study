# Results

Numbers are from `Version 2/CHANGES_v2.md` (sections R1–R9) and result files in `Version 1/code_v1/results/`.

## Primary estimate (LightGBM, raw values)
| Evaluation | 4 classes | 3 seizure types |
|---|---|---|
| Rolling origin (5 folds) | 0.748 ± 0.080 (t-interval 0.648–0.847) | 0.677 ± 0.104 (0.548–0.805) |
| Blocked CV | 0.777 ± 0.048 | 0.720 ± 0.053 |
| Final-20% hold-out | 0.810 (CI 0.771–0.845) | 0.752 (0.694–0.806) |
| Fully nested selection | 0.747 | 0.665 |
| Random split | about 0.98 | |

* Test-selected comparator SVM-RBF (raw): 0.842 / 0.776 on hold-out; blocked CV / rolling origin 0.755 / 0.754 and
  0.685 / 0.689.
* Healthy class per-class F1 ≥ 0.99 (0.999 random, 0.995 blocked).

## Recording-fingerprint confound
|x| alone: 0.720 / 0.648 (LightGBM) and 0.702 / 0.627 (LR) in rolling origin. Primary model vs LR on |x|: +0.046 to
+0.054, Holm p = 0.15–0.16 (not significant).

## CHB-MIT provenance search (partial)
246 of 686 files, 282 h, all 24 subjects; positive control 1.00; no file above 0.42 (threshold 0.80).

## Windowed analysis (separate setting)
0.97 (CI 0.90–1.00) vs 0.77 single-row and 0.89 window vote; blocked CV 0.92 / 0.67.

## Deep models
BiLSTM hold-out 0.468 → 0.650 after tuning; no deep model beats LightGBM (raw) in blocked CV.

## Computational footprint (general-purpose CPU)
0.05–0.74 ms per sample (Python), 16 kB–2.3 MB ONNX; tuned LightGBM with 73 features 3.97 ms per sample, 87.5% in
feature extraction. No edge hardware was used.

## Result files (`Version 1/code_v1/results/`)
`holdout_metrics.csv`, `holdout_per_class.csv`, `cv_fold_scores.csv`, `cv_summary.csv`, `stats_*.csv`,
`nemenyi_*.csv`, `ablation_feature_groups.csv`, `shap_*.csv`, `leakage_*.csv`, `provenance_*.csv/json`,
`shortcut_stumps.csv`, `edge_deployment.csv`, `windowed_3class_lobo.csv`; test-set probabilities in `preds/`.
Extended (Version 2) analyses are in `Version 2/supplementary_v2/BEED_code_S1_review_v2.zip`.
