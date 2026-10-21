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

## 🔗 Other repositories by Rishabh Aryan

| Repository | Description | Live |
|---|---|---|
| [100-days-of-python](https://github.com/Rishabh-bgp/100-days-of-python) | My original, offline, no-API-key solutions to all 100 projects of the 100 Days of Code Python bootcamp curriculum… | — |
| [Agentic-Engineering-Ebook](https://github.com/Rishabh-bgp/Agentic-Engineering-Ebook) | Agentic Engineering is a practical notebook-book on specifying, generating, verifying, and refining multi-tenant… | — |
| [Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction](https://github.com/Rishabh-bgp/Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction) | Brain–computer interfaces (BCIs) link neural activity to external devices, translating brain signals via algorithms… | — |
| [building-a-virtual-cell-in-python](https://github.com/Rishabh-bgp/building-a-virtual-cell-in-python) | A cell has a normal gene-activity pattern. Knock out one gene and that pattern changes. This project trains a small… | — |
| [Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML](https://github.com/Rishabh-bgp/Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML) | ML is transforming Chronic Kidney Disease (CKD) detection in HIV patients, catching early asymptomatic stages to… | — |
| [cobratate](https://github.com/Rishabh-bgp/cobratate) | Cobratate, an esoteric imperative programming language. Tate-flavored syntax, ordinary semantics, and a Python… | [open](https://rishabh-bgp.github.io/cobratate/) |
| [coda-synth](https://github.com/Rishabh-bgp/coda-synth) | CoDA-Synth: stylometric detection of LLM-rewritten dark-web documents with sparse TF-IDF + LinearSVC. Research lab,… | [open](https://shale-ember-delta-kite.grok.me/) |
| [IIT-ROPAR-GENAI-FOUNDATIONS](https://github.com/Rishabh-bgp/IIT-ROPAR-GENAI-FOUNDATIONS) | Free open-source Generative AI foundations ebooks (2026) covering Claude, Grok, Gemini, Cursor, OpenAI and more —… | — |
| [kbc-game](https://github.com/Rishabh-bgp/kbc-game) | Kaun Banega Crorepati (KBC) quiz game for the terminal in Python: 16-question prize ladder, safe levels and 4… | — |
| [knn-classifier](https://github.com/Rishabh-bgp/knn-classifier) | Simple K-Nearest Neighbors classifier on the classic Iris dataset using scikit-learn. Loads data, trains the model,… | — |
| [linear-regression-model-](https://github.com/Rishabh-bgp/linear-regression-model-) | Supervised linear regression on the scikit-learn diabetes dataset. A single feature (BMI) is used to predict… | — |
| [logistic-regression](https://github.com/Rishabh-bgp/logistic-regression) | This graduate capstone applies multi-class logistic regression to the Iris dataset. It includes EDA, stratified… | — |
| [logistics-website-capstone](https://github.com/Rishabh-bgp/logistics-website-capstone) | Logistics website capstone: Kargonaut Logistics, a premium freight site mixing claymorphism, glassmorphism and… | [open](https://rishabh-bgp.github.io/logistics-website-capstone/) |
| [lovelight-leap](https://github.com/Rishabh-bgp/lovelight-leap) | 💖 Lovelight Leap: a love-themed 2D platformer in vanilla JS canvas. 4 worlds, a boss you cheer up, original… | [open](https://rishabh-bgp.github.io/lovelight-leap/) |
| [machine-learning-part-1](https://github.com/Rishabh-bgp/machine-learning-part-1) | Machine learning in Python with scikit-learn: heart disease classification and medical insurance cost regression… | — |
| [Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail](https://github.com/Rishabh-bgp/Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail) | Agentic commerce lets shopping and seller agents place orders without showing operational truth. This IS case argues… | — |
| [morocco-city-aware-load-forecasting](https://github.com/Rishabh-bgp/morocco-city-aware-load-forecasting) | City-aware short-term load forecasting and peak detection on high-resolution Moroccan smart-meter data (5 zones).… | — |
| [NIELIT-DATA-SCIENCE](https://github.com/Rishabh-bgp/NIELIT-DATA-SCIENCE) | Hands-on NumPy, Pandas, and Matplotlib Jupyter notebooks for the NIELIT Data Science curriculum — practical,… | — |
| [NIELIT-MACHINE-LEARNING-PROJECTS](https://github.com/Rishabh-bgp/NIELIT-MACHINE-LEARNING-PROJECTS) | Nineteen end-to-end machine learning projects: classification, regression, clustering, and recommendation, with… | — |
| [NIELIT-PYTHON-](https://github.com/Rishabh-bgp/NIELIT-PYTHON-) | Open-source Python teaching notebooks: a language reference and five capstone programs, with detailed documentation.… | — |
| [pan-cancer-classification](https://github.com/Rishabh-bgp/pan-cancer-classification) | Leakage, equivalence, conformal reliability and data efficiency on the UCI TCGA Pan-Cancer RNA-Seq dataset — IIT… | — |
| [Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks](https://github.com/Rishabh-bgp/Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks) | This notebook details the implementation and evaluation of a Bayesian perspective Early stage Event Prediction… | — |
| [prediction-as-power](https://github.com/Rishabh-bgp/prediction-as-power) | Laboratory and working paper on prediction as power in European electricity markets. A DE-like hourly simulator… | — |
| [snakes-and-ladders](https://github.com/Rishabh-bgp/snakes-and-ladders) | 🐍🪜 Original Snakes & Ladders browser game: hand-drawn SVG snakes and ladders, 3D dice, 2-4 humans or bots, sound,… | [open](https://rishabh-bgp.github.io/snakes-and-ladders/) |
| [spam-scan](https://github.com/Rishabh-bgp/spam-scan) | Explainable spam, phishing & Indian-scam detector for SMS/email (English, Hindi, Hinglish): TF-IDF + Logistic… | [open](https://spam-scan.onrender.com) |
| [vedic-math-hybrid-kt](https://github.com/Rishabh-bgp/vedic-math-hybrid-kt) | Hybrid Vedic Math AI: rule-based sutras (Urdhva-Tiryagbhyam, Nikhilam, Yavadunam) plus RF classifier that cleans… | — |
| [wind-turbine-power-model](https://github.com/Rishabh-bgp/wind-turbine-power-model) | Single-turbine power curves + wind-farm simulation. Weibull wind stats, hourly NASA-style speed data, and… | — |
| [zoo-website-capstone](https://github.com/Rishabh-bgp/zoo-website-capstone) | Capstone project: a fictional zoo website built twice — a beginner-style simple version and a cinematic advanced… | [open](https://rishabh-bgp.github.io/zoo-website-capstone/) |
