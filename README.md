<div align="center">

# 🧠 BEED ML Study

### Shortcuts and Temporal Leakage in the Bangalore EEG Epilepsy Dataset: a Provenance Audit and Leakage-Aware Benchmark of Lightweight Seizure-Type Classifiers

![Python](https://img.shields.io/badge/Python-3.13.5-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-02569B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%20CPU-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/data-UCI%20BEED%20(included)-success)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

> Research work carried out as part of the **6-Month Industrial Training & Certification Program in Artificial Intelligence and Machine Learning**, jointly offered by [IIT Ropar](https://www.iitrpr.ac.in/) and [NIELIT Ropar](https://www.nielit.gov.in/ropar/), under MeitY, Govt. of India.

---

## 📌 Overview
The Bangalore EEG Epilepsy Dataset (BEED, UCI doi:10.24432/C5K33B) is used to report very high seizure-type
classification scores. This study audits the dataset's **provenance** (its healthy class appears to be taken from the
Bonn EEG data), measures **temporal leakage** from row-level random splits, and benchmarks lightweight classifiers
under a **leakage-aware protocol** (grouped/blocked splits with purging, nested tuning, rolling-origin evaluation,
cluster bootstrap).

## 🔑 Key findings
*(numbers from `Version 2/CHANGES_v2.md` and `Version 1/code_v1/results/`)*

| Item | Result |
|---|---|
| Dataset structure | 8,000 rows, 2,000 per class, 0 duplicate rows; class-0 rows match Bonn segments (16–23 rows per segment) |
| Primary model (pre-specified) | LightGBM on raw values |
| Rolling-origin macro-F1 | **0.748 ± 0.080** (4 classes); **0.677 ± 0.104** (3 seizure types) |
| Blocked CV | **0.777 ± 0.048** / **0.720 ± 0.053** |
| Final-20% hold-out | **0.810** (CI 0.771–0.845) / **0.752** (0.694–0.806) |
| Random row-level split | about **0.98** — inflated by temporal leakage |
| Recording fingerprint (\|x\| only) | 0.720 / 0.648 (LightGBM), about 85–95% of the primary model's above-chance performance |
| Windowed setting (separate) | 0.97 hold-out (CI 0.90–1.00); blocked CV 0.92 / 0.67 |
| CPU footprint | 0.05–0.74 ms per sample (Python); 16 kB–2.3 MB ONNX |

See [`docs/results.md`](docs/results.md).

## 📂 Repository structure
```
IJCRT/                        journal-format paper, supplement, cover letter, figures, notes
Version 1/
  code_v1/                    full pipeline: run_all.py, src/beed/, data/, results/, figures/, analysis.ipynb, manuscript/
  paper_v1/                   review (anonymised) and camera-ready manuscripts
  supplementary_v1/           supplement and code archive S1
  presentation_v1/            conference deck and speaker notes
  cover_letter_v1.*, BEED_simple_explainer_v1.*, submission_checklist_v1.md
Version 2/
  paper_v2/, supplementary_v2/ (incl. extended-analysis code archive), presentation_v2/,
  cover_letter/, explainer_v2/, CHANGES_v2.md, submission_checklist.md
docs/                         extended documentation
```

## 🚀 Quick start
```bash
git clone https://github.com/Rishabh-bgp/beed-ml-study.git
cd "beed-ml-study/Version 1/code_v1"
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # torch CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
python run_all.py                   # all stages (~1–1.5 h on 8 CPU cores)
```
Full details: [`docs/reproducing.md`](docs/reproducing.md).

## 📚 Documentation
| Page | Contents |
|---|---|
| [docs/data.md](docs/data.md) | BEED and Bonn data, structure, provenance, licences |
| [docs/methodology.md](docs/methodology.md) | Groups, protocols, models, statistics |
| [docs/results.md](docs/results.md) | Headline results and result files |
| [docs/reproducing.md](docs/reproducing.md) | Environment, stages, commands |
| [docs/versions.md](docs/versions.md) | What Version 1, Version 2 and IJCRT contain |
| [docs/faq.md](docs/faq.md) | FAQ and limitations |

## 📝 Citation
See [`CITATION.cff`](CITATION.cff).

## ⚖️ Licence
Code and documentation: MIT © 2026 Rishabh Aryan. Datasets remain under their original licences (see `docs/data.md`).
