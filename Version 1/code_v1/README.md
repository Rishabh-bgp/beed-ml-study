# BEED provenance audit and leakage-aware benchmark — code and results

Reproducible pipeline for the paper *"Shortcuts and Temporal Leakage in the Bangalore EEG Epilepsy Dataset: a Provenance
Audit and Leakage-Aware Benchmark of Lightweight Seizure-Type Classifiers"* (prepared for NIELIT Bio-AI Conclave 2026, Track 5).

## Contents

| Path | Description |
|---|---|
| `run_all.py` | Single entry point; regenerates every result and figure |
| `src/beed/` | Package: `config`, `data` (loading, provenance matching, grouped/blocked splitters, purge), `features`, `models`, `deep`, `evaluation` (metrics, cluster bootstrap, corrected t-test, Friedman/Nemenyi), `experiments`, `plots` |
| `data/BEED_Data.csv` | BEED file from the UCI repository (doi:10.24432/C5K33B, CC BY 4.0) |
| `data/external/` | Bonn-derived UCI *Epileptic Seizure Recognition* CSV (downloaded automatically from the creators' repository and SHA-256 verified if absent) |
| `results/` | All CSV outputs (hold-out metrics, CV fold scores, statistical tests, ablation, SHAP, provenance, leakage, edge profile); `results/preds/` holds test-set probabilities |
| `figures/` | All figures (PNG, 300 dpi) |
| `analysis.ipynb` | Executed notebook walking through the main pipeline |
| `manuscript/` | Scripts that build the manuscript, supplement and cover letter (.docx) directly from `results/`. The optional `authors_camera_ready.json` holds the camera-ready author block; leave it out of any review copy |

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
python run_all.py                          # all stages (about 1–1.5 h on 8 CPU cores)
python run_all.py provenance leakage       # or individual stages
jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
mkdir -p out && cd manuscript && python build_paper.py ../out && python build_supplement.py ../out
```

Stages: `provenance leakage holdout holdout_extra cv stats ablation shap windowed edge figures`.
Environment used: Linux, Python 3.13.5, 8 virtual CPUs, 16 GB RAM, no GPU. Seeds are fixed (`SEED = 42`).
Set `BEED_N_JOBS` to change parallelism.

## Key design choices

* **Groups.** Class-0 rows are grouped by their Bonn source segment (found by exact matching); class-1–3 rows (continuous
  multichannel recordings) are grouped into contiguous 200-row blocks.
* **Protocols.** *random* = stratified row-level split (as in prior work); *blocked* = whole groups held out, final 20% of
  each seizure recording as test, 50-row purge between training and test rows of the same recording.
* **Nested tuning.** Inner three-fold CV follows the same protocol as the outer split.
* **Uncertainty.** Cluster bootstrap (Bonn segments / 50-row chunks) in addition to the naive row bootstrap.

## Licence

Code: [choose licence, e.g. MIT]. Data remain under their original licences.
