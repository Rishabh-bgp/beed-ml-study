"""Build the manuscript (.docx) in review (anonymised) and camera-ready versions."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from docx_utils import Doc, Refs
from bib import BIB
from results_loader import load, f3, ci, hrow, DEEP

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
CLASSICAL = ["LR", "kNN", "SVM-RBF", "RF", "XGBoost", "LightGBM"]
TITLE = ("Shortcuts and Temporal Leakage in the Bangalore EEG Epilepsy Dataset: a Provenance Audit "
         "and Leakage-Aware Benchmark of Lightweight Seizure-Type Classifiers")


def authors():
    """Camera-ready author block; kept in a separate file so the review code archive can omit it."""
    import json
    f = Path(__file__).with_name("authors_camera_ready.json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"authors": "[Author Name]^{1,*}, [Co-author Name]^{1}", "affiliation": "^{1}[Department], [Institution], [City], [Country]",
            "corresponding": "^{*}Corresponding Author: [Author Name], [Email]. ORCID: [ORCID]",
            "cover_corresponding": "Corresponding author: [Author Name]^{*}, [Department], [Institution], [City], [Country]; [Email]; ORCID [ORCID]."}


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m} × 10^{{{int(e)}}}"


def pct(x):
    return f"{100 * x:.1f}%"


def build(anonymous: bool, out: Path):
    R = load()
    rf = Refs(BIB)
    c = rf.cite
    D = Doc()
    P = R["prov"]; NN = R["nn"]; B = R["best"]; E = R["edge"]
    bl = R["block"]

    def blk(model, B_, gap):
        g = bl[(bl.model == model) & (bl.block_size == B_) & (bl.gap == gap)]
        return g.iloc[0]
    lg_r, lg_b = blk("LightGBM (raw+eng)", 1, 0), blk("LightGBM (raw+eng)", 200, 50)
    nn_r, nn_b = blk("1-NN (raw)", 1, 0), blk("1-NN (raw)", 200, 50)
    b4r, b4b = B[("4class", "random")], B[("4class", "blocked")]
    b3r, b3b = B[("3class", "random")], B[("3class", "blocked")]
    bbr, bbb = B[("binary", "random")], B[("binary", "blocked")]
    knn_r = hrow(R, "4class", "random", "kNN", "raw"); knn_b = hrow(R, "4class", "blocked", "kNN", "raw")
    cvs = R["cv"]
    def cvbest(task, prot):
        g = cvs[(cvs.task == task) & (cvs.protocol == prot)].sort_values("macro_f1_mean", ascending=False)
        return g.iloc[0], g
    cv4b, cv4b_all = cvbest("4class", "blocked"); cv4r, cv4r_all = cvbest("4class", "random")
    cv3b, cv3b_all = cvbest("3class", "blocked"); cv3r, cv3r_all = cvbest("3class", "random")
    fr = R["friedman"].set_index(["task", "protocol"])
    tt = R["ttest"]
    win = R["windowed"]
    sh3 = R["shap"][("3class", "blocked")]; sh4b = R["shap"][("4class", "blocked")]; sh4r = R["shap"][("4class", "random")]
    top3 = list(sh3.index[:5]); top4b = list(sh4b.index[:5]); top4r = list(sh4r.index[:5])
    abl = R["ablation"]
    def ab(task, prot, cfg, col="cv_macro_f1_mean"):
        return abl[(abl.task == task) & (abl.protocol == prot) & (abl.config == cfg)][col].iloc[0]
    drop_r = b4r.macro_f1 - b4b.macro_f1
    hh = R["holdout"]
    def _f(prot, m):
        return hh[(hh.task == "4class") & (hh.protocol == prot) & (hh.model == m) & (hh.features == "raw")].macro_f1.iloc[0]
    drops = pd.Series({m: _f("random", m) - _f("blocked", m) for m in CLASSICAL}).sort_values(ascending=False)
    nn_adj = [NN.loc[k, "nn_within_1_row"] for k in (1, 2, 3)]
    cnn_e = E.loc["1D-CNN"]; mlp_e = E.loc["MLP"]; lgb_e = E.loc["LightGBM"]

    # ---------------------------------------------------------------- title page
    D.title(TITLE)
    if anonymous:
        D.para("[Author names and affiliations removed for double-blind review]", center=True, italic=True)
    else:
        A = authors()
        D.para(A["authors"], center=True)
        D.para(A["affiliation"], center=True, size=11)
        D.para(A["corresponding"], center=True, size=11)
    D.para("Track 5: Digital Life Sciences, Health, Agriculture & Biodiversity (healthcare artificial intelligence)", center=True, size=11, italic=True)

    # ---------------------------------------------------------------- abstract
    D.h1("Abstract")
    D.para(
        "Public electroencephalography datasets are widely used to develop automated seizure classifiers, and the recently "
        "released Bangalore EEG Epilepsy Dataset has quickly attracted reports of 95–99% accuracy. We asked whether these "
        "figures reflect physiological discrimination between healthy, generalised-seizure, focal-seizure and seizure-event "
        "recordings, or properties of how the public file was assembled. We audited the 8000-row, 16-column file using "
        "autocorrelation analysis and exact matching against an older public benchmark, and then benchmarked seven classical "
        "and four small deep-learning models under two evaluation protocols: the conventional random row-level split and a "
        "leakage-aware protocol that holds out contiguous time blocks with a temporal purge gap. Hyper-parameters were tuned "
        "by nested cross-validation; we report bootstrap confidence intervals, corrected resampled t-tests, Friedman tests, "
        "calibration and Shapley-value explanations. The audit showed that the columns mean different things in different "
        f"classes: every one of the 2000 'healthy' rows is an exact 16-sample excerpt of the Bonn eyes-closed recordings "
        f"(173.61 Hz, single channel), whereas the three seizure classes are 16-channel time points of continuous recordings, "
        f"so consecutive rows are near-duplicates. Under a random split, the best model reached a macro-averaged F1 score of "
        f"{f3(b4r.macro_f1)}; under the leakage-aware protocol, the best reached {f3(b4b.macro_f1)} (95% confidence interval "
        f"{ci(b4b.f1_cci_lo, b4b.f1_cci_hi)}). Healthy-versus-seizure detection remained near-perfect under both protocols "
        f"(macro-F1 ≥ {f3(min(bbr.macro_f1, bbb.macro_f1))}), consistent with a source shortcut rather than a clinical signal. "
        f"For the three seizure types alone, macro-F1 fell from {f3(b3r.macro_f1)} to {f3(b3b.macro_f1)}. Results on this "
        "file should therefore be reported with leakage-aware splits, and the healthy class should not be used for "
        "clinical claims. We release a reproducible pipeline and a lightweight model profile for edge deployment.")
    D.para("**Keywords:** electroencephalography; epileptic seizure classification; data leakage; dataset provenance; "
           "blocked cross-validation; probability calibration; explainable machine learning; edge deployment")

    # ---------------------------------------------------------------- 1 introduction
    D.h1("1 Introduction")
    D.para(
        f"Epilepsy diagnosis relies heavily on visual review of the electroencephalogram (EEG), which is slow and requires "
        f"specialist expertise; machine-learning (ML) classifiers are therefore an active research area "
        f"{c('shoeb', 'acharya', 'tsiouris', 'truong')}. Progress depends on public datasets such as the Bonn recordings "
        f"{c('andrzejak')} and the CHB-MIT scalp database {c('shoeb', 'physionet')}. The Bangalore EEG Epilepsy Dataset (BEED), "
        f"released through the UCI Machine Learning Repository in 2024 {c('beed', 'seqboost')}, adds a four-class problem: "
        f"healthy subjects (class 0), generalised seizures (1), focal seizures (2) and seizure events accompanied by "
        f"behaviours such as blinking or staring (3). Distinguishing focal from generalised onset matters clinically "
        f"because it guides treatment choices, and seizure-type classification is markedly harder than seizure detection "
        f"{c('roy')}.")
    D.para(
        f"Within two years, several studies reported 95–99% accuracy on BEED {c('seqboost', 'paneru', 'nandatama', 'papernet', 'obaido', 'epilepsyformer')}. "
        f"However, the public BEED file contains only 8000 rows of 16 integers with no subject or recording identifiers, and "
        f"its documentation does not say whether a row is 16 channels at one instant or 16 samples of one channel. Most "
        f"reported evaluations split rows at random. In EEG and other autocorrelated data, random splits can place almost "
        f"identical neighbouring samples in training and test sets, inflating performance {c('kapoor', 'brookshire', 'roberts')}. "
        f"Whether this applies to BEED had not been examined.")
    D.runin("Research questions",
        "We addressed four questions. (RQ1) What do the rows and columns of the public BEED file represent, and is the "
        "representation the same across classes? (RQ2) How much of the reported performance survives an evaluation that "
        "prevents temporal leakage? (RQ3) Once shortcuts are controlled, which signal properties separate generalised, focal "
        "and seizure-event recordings, and are the predicted probabilities calibrated? (RQ4) Can a model small enough for "
        "wearable or edge hardware retain the leakage-safe performance?")
    D.runin("Contributions",
        "(i) A structural and provenance audit showing that the healthy class is an exact excerpt of the Bonn eyes-closed set "
        "whereas the seizure classes are multichannel time points, which creates a source shortcut and strong temporal "
        "autocorrelation between rows. (ii) A leakage-aware protocol (grouped, temporally blocked splits with a purge gap and "
        "nested tuning) and a quantification of the gap between random and blocked evaluation. (iii) A statistically "
        "grounded benchmark of eleven models with cluster-bootstrap confidence intervals, corrected resampled t-tests, "
        "Friedman/Nemenyi analysis and calibration. (iv) Shapley-value interpretation of seizure-type discrimination and a "
        "parameter and latency profile of lightweight models. (v) An open, fully reproducible pipeline (Supplementary File S1).")

    # ---------------------------------------------------------------- 2 related work
    D.h1("2 Related Work")
    D.h2("2.1 Seizure detection and classification benchmarks")
    D.para(
        f"The Bonn dataset {c('andrzejak')} contains five sets of 100 single-channel, 23.6 s segments sampled at 173.61 Hz; a "
        f"widely redistributed UCI version cuts each segment into 1 s chunks of 178 samples and shuffles them {c('wu_uci')}. "
        f"Deep convolutional networks report near-perfect accuracy on Bonn {c('acharya')}; CHB-MIT provides long multichannel "
        f"paediatric recordings at 256 Hz {c('shoeb', 'physionet')} used for detection and prediction {c('tsiouris', 'truong')}. "
        f"Classic hand-crafted EEG descriptors include Hjorth parameters {c('hjorth')}, line length {c('esteller')}, sample and permutation entropy "
        f"{c('richman', 'bandt')} and discrete wavelet energies {c('subasi')}. For seizure-type classification on the TUH corpus, "
        f"weighted F1 fell from 0.901 under seizure-wise to 0.561 under patient-wise cross-validation {c('roy')}, showing how "
        f"strongly the evaluation protocol shapes conclusions.")
    D.h2("2.2 Prior work on BEED")
    D.para(f"Table 1 summarises published BEED studies. Most use row-level random or stratified splits; one preprint "
           f"described its protocol as subject-independent, but its methods section reports a stratified 70/15/15 sample-level "
           f"split, because the file has no subject identifiers {c('papernet')}. One study used 20 s, 14-channel segments "
           f"{c('epilepsyformer')}, which suggests that a richer version of BEED exists than the public CSV file.")
    D.table("Published studies on the public BEED file and their evaluation protocols",
            ["Study", "Method", "Evaluation protocol (as reported)", "Reported result"],
            [[f"Najmusseher and Nizar Banu {c('seqboost')}", "UMAP + FFT features, stacked LSTM/XGBoost/GB (SeqBoostNet)", "Train/test split of rows", "Accuracy 96.71%"],
             [f"Paneru {c('paneru')}", "UMAP + FFT, two-level stacking, LIME", "Hold-out; 10-fold nested CV", "Accuracy 97.06% (nested CV 91.51%)"],
             [f"Nandatama {c('nandatama')}", "Nine classical models, noise robustness", "Row-level split", "LightGBM F1 0.9831"],
             [f"Sajid et al. {c('papernet')}", "CNN + residual attention + BiLSTM (PaperNet)", "Stratified 70/15/15 sample-level split", "Macro-F1 0.9576"],
             [f"Obaido and Esenogho {c('obaido')}", "Tabular foundation models vs ensembles", "Cross-validation on rows", "High accuracy (see source)"],
             [f"Heba and Atya {c('epilepsyformer')}", "CNN–vision transformer (EpilepsyFormer)", "20 s segments, described as subject-independent", "Accuracy 98.3%"]],
            note="Abbreviations: UMAP, uniform manifold approximation and projection; FFT, fast Fourier transform; LSTM, long "
                 "short-term memory; GB, gradient boosting; LIME, local interpretable model-agnostic explanations; CNN, "
                 "convolutional neural network; CV, cross-validation. Protocols are as stated by the original authors.",
            col_widths=[3.4, 4.6, 4.6, 3.4])
    D.h2("2.3 Leakage in machine-learning evaluation")
    D.para(f"Data leakage, in which information about the test set reaches the model during training, is a leading cause of "
           f"irreproducible ML results {c('kapoor')}. In EEG, splitting segments rather than subjects inflated deep-learning "
           f"accuracy substantially {c('brookshire')}. For autocorrelated data, blocked cross-validation with buffers is "
           f"recommended {c('roberts')}. To our knowledge, no BEED study has examined row adjacency, the meaning of the columns "
           f"or the origin of the classes.")

    # ---------------------------------------------------------------- 3 methods
    D.h1("3 Materials and Methods")
    D.para("The study design (Supplementary Fig. S9) comprises a structural and provenance audit, provenance-based grouping, "
           "and two evaluation protocols sharing identical features, models, tuning and metrics.")
    D.h2("3.1 Dataset")
    D.para(f"We used the BEED CSV file from the UCI repository {c('beed')} (8000 rows; integer columns X1–X16; label y). "
           f"According to its documentation, BEED was recorded at a neurological centre in Bangalore with the 10–20 system at "
           f"256 Hz, from 20 adults per class (21–55 years) {c('beed', 'seqboost')}. Each class has 2000 rows stored contiguously "
           f"(label changes at rows 2000, 4000 and 6000). There were no missing values or duplicated rows. No filtering, "
           f"resampling or artefact removal was applied, so that the file was analysed exactly as distributed; standardisation "
           f"was fitted on training data only.")
    D.h2("3.2 Structure and provenance audit")
    D.para("For each class we computed (i) the mean correlation between columns k and k+l (within-row lag l), (ii) the mean "
           "correlation of each column between rows i and i+l (across-row lag l) and (iii) whether the within-row correlation "
           "depends only on lag (a Toeplitz structure expected for time samples) or on position (expected for electrodes). "
           f"Because class-0 rows were smooth and lacked across-row continuity, we searched every 16-sample window of every row "
           f"of the UCI Bonn file {c('wu_uci')} for exact integer matches with BEED rows (SHA-256 of the file is recorded in the code).")
    D.h2("3.3 Evaluation protocols")
    D.para("**Random protocol.** A stratified 80/20 row-level hold-out split and repeated (5 × 2) stratified cross-validation (CV), "
           "as in most prior work. **Blocked protocol.** Rows were first grouped: class-0 rows by their Bonn source segment "
           "(100 groups, from the audit) and class-1–3 rows into contiguous blocks of 200 rows (0.78 s at 256 Hz). The hold-out "
           "test set comprised the final 20% of each seizure recording and 20 randomly chosen Bonn segments. CV folds were "
           "formed by assigning whole groups to folds within each class. Training rows within 50 rows (0.195 s) of any "
           "test row of the same class were purged; across-row autocorrelation is close to zero at this lag (Table 2). "
           "In both protocols, hyper-parameters were chosen by an inner three-fold CV that followed the same protocol "
           "as the outer split (nested CV), optimising macro-F1. The test set was used once.")
    D.para("We studied three tasks: the four-class problem; a binary healthy-versus-seizure task; and a three-class seizure-type "
           "task (generalised, focal, seizure events) that excludes the healthy class and so removes the source shortcut.")
    D.h2("3.4 Feature engineering")
    D.para(f"Each 16-value row was described by 57 engineered features in four groups. Time domain (24): moments, quantiles, "
           f"root mean square, Hjorth activity, mobility and complexity {c('hjorth')}, line length {c('esteller')}, mean absolute "
           f"first and second differences, zero crossings, number of extrema and lag-1 autocorrelation. Frequency domain (14): "
           f"relative power in the eight non-DC bins of a Hann-windowed 16-point FFT, spectral entropy, centroid, 80% edge, "
           f"peak bin and a low/high power ratio. With only 16 points, conventional band powers (delta to gamma) cannot be "
           f"resolved, so normalised bins were used. Nonlinear (7): permutation {c('bandt')}, sample and approximate entropy "
           f"{c('richman')}, singular-value-decomposition entropy and Petrosian, Katz and Higuchi fractal dimensions. Wavelet (12): "
           f"log energy, relative energy, standard deviation and mean absolute value of the Daubechies-2 level-2 sub-bands "
           f"{c('subasi', 'pywt')}. Feature sets were raw (16), engineered (57) and raw+engineered (73). For class 0 these features "
           f"are temporal; for classes 1–3 they describe the spatial profile across electrodes (Section 4.1).")
    D.h2("3.5 Models and training")
    D.para(f"Classical models {c('sklearn')}: multinomial logistic regression (LR), k-nearest neighbours (kNN), radial-basis-function "
           f"support vector machine (SVM-RBF), random forest (RF), XGBoost {c('xgboost')} and LightGBM {c('lightgbm')}, plus a "
           f"majority-class baseline. Compact grids are listed in Supplementary Table S1. Deep models {c('pytorch')} took the "
           f"16 standardised raw values as input: a multilayer perceptron (MLP), a three-layer one-dimensional convolutional "
           f"network (1D-CNN), a bidirectional long short-term memory network (BiLSTM) and a two-layer transformer encoder. Each had "
           f"11–21 thousand parameters (Section 4.8). They were trained with AdamW (learning rate 0.002, weight decay 10^{{-4}}), batch "
           f"size 128, up to 60 epochs, with early stopping (patience 8) on the macro-F1 of a validation split that respected the protocol. "
           f"All random seeds were fixed (seed 42).")
    D.h2("3.6 Metrics, uncertainty and statistical testing")
    D.para(f"We report accuracy, macro-F1, Cohen's kappa, one-vs-rest macro area under the receiver operating characteristic curve "
           f"(AUC), per-class precision and recall, Brier score and expected calibration error (ECE, 15 bins) {c('guo')}. Hold-out "
           f"95% confidence intervals (CIs) were obtained from 2000 bootstrap resamples. Because test rows are autocorrelated, the "
           f"primary CI is a class-stratified cluster bootstrap that resamples whole Bonn segments (class 0) or 50-row chunks "
           f"(classes 1–3); naive row-level CIs are given in Supplementary Table S3. Models were compared on the 10 outer CV folds "
           f"with the Nadeau–Bengio corrected resampled t-test {c('nadeau', 'bouckaert')} against the best model, with Holm "
           f"correction, and with the Friedman test followed by Nemenyi post-hoc analysis {c('demsar')}.")
    D.h2("3.7 Interpretability, ablation and deployment profiling")
    D.para(f"For LightGBM on raw+engineered features we computed TreeSHAP values {c('shap')} on the hold-out test set and gain "
           f"importance. A feature-group ablation retrained LightGBM (fixed hyper-parameters) on each group alone and on all "
           f"engineered features minus each group. To check whether seizure types are separable given true temporal context, "
           f"we also built 0.25 s windows (64 rows) of classes 1–3 and computed per-channel log band powers (Welch, 256 Hz) "
           f"and line length, evaluated by leave-one-block-out CV with neighbouring blocks purged. For deployment we counted "
           f"parameters and measured median single-sample CPU latency (one thread, 500 repetitions).")
    D.h2("3.8 Software and hardware environment")
    D.para("Experiments ran on a Linux virtual machine (8 virtual Intel Xeon CPUs, 16 GB RAM, no GPU) with Python 3.13, "
           "scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, PyTorch 2.14.1 (CPU), SHAP 0.52.0, SciPy 1.18.1, PyWavelets 1.10.0 "
           "and AntroPy 0.2.2. A single command (run_all.py) regenerates every table and figure (Supplementary File S1).")

    # ---------------------------------------------------------------- 4 results
    D.h1("4 Results")
    D.h2("4.1 The columns mean different things in different classes")
    D.para(f"Class 0 showed the signature of time samples: the within-row lag-1 correlation was {f3(P.loc[0, 'within_row_lag1_corr'])} "
           f"at every position (standard deviation across positions {P.loc[0, 'lag1_corr_sd_across_positions']:.3f}), decaying "
           f"smoothly with lag, whereas consecutive rows were uncorrelated (r = {P.loc[0, 'across_row_lag1_corr']:.3f}) (Fig. 1, Table 2). "
           f"Classes 1–3 showed the opposite pattern: across-row lag-1 correlations of {f3(P.loc[1, 'across_row_lag1_corr'])}, "
           f"{f3(P.loc[2, 'across_row_lag1_corr'])} and {f3(P.loc[3, 'across_row_lag1_corr'])}, falling to about zero by 50 rows, "
           f"with position-dependent, non-Toeplitz correlations between columns. The strongest correlations occurred at lags 4, 8 and 12, "
           f"a pattern consistent with four chains of four electrodes, as in a longitudinal bipolar montage. Thus, in classes 1–3, "
           f"each row is one time point of a 16-channel recording, and each class is a contiguous recording of about "
           f"{2000 / 256:.1f} s at the documented 256 Hz. Classes also differed sharply in amplitude (row standard deviation "
           f"{P.loc[0, 'amp_sd']:.1f} for class 0 versus {P.loc[1, 'amp_sd']:.1f}, {P.loc[2, 'amp_sd']:.1f} and {P.loc[3, 'amp_sd']:.1f}).")
    D.figure(FIG / "fig_structure.png", "Correlation structure reveals different data layouts across classes",
             "Top: column-by-column correlation matrices per class. Class 0 shows a Toeplitz band (time samples); classes 1–3 "
             "show position-specific structure with strong correlations at lags 4, 8 and 12 (electrode chains). Bottom: (i) mean "
             "within-row correlation versus column lag; (ii) mean across-row correlation versus row lag; (iii) lag-1 "
             "correlation for each adjacent column pair; (iv) distribution of row standard deviation (log scale). Classes: "
             "H, healthy; GS, generalised seizure; FS, focal seizure; SE, seizure events.")
    rows = []
    for k in range(4):
        rows.append([["H (0)", "GS (1)", "FS (2)", "SE (3)"][k], f"{P.loc[k, 'amp_sd']:.1f}", f3(P.loc[k, "within_row_lag1_corr"]),
                     f"{P.loc[k, 'lag1_corr_sd_across_positions']:.3f}", f3(P.loc[k, "across_row_lag1_corr"]),
                     f3(P.loc[k, "across_row_lag50_corr"]), str(int(P.loc[k, "bonn_exact_matches"])),
                     "16 samples × 1 channel" if k == 0 else "1 time point × 16 channels"])
    D.table("Structural audit of the public BEED file by class",
            ["Class", "Row SD", "Within-row lag-1 r", "SD of lag-1 r over positions", "Across-row lag-1 r", "Across-row lag-50 r",
             "Exact Bonn matches", "Inferred row layout"], rows,
            note="SD, standard deviation (dataset units); r, Pearson correlation. Exact matches: rows identical to a 16-sample "
                 "window of the UCI Bonn file (2000 rows per class).", font_size=8.5, col_widths=[1.5, 1.4, 1.9, 2.1, 1.9, 1.9, 1.7, 3.6])
    D.h2("4.2 The healthy class is an excerpt of the Bonn dataset")
    bm_off = int(list(eval(P.loc[0, "bonn_offsets"]).keys())[0])
    D.para(f"All {int(P.loc[0, 'bonn_exact_matches'])} class-0 rows matched a 16-sample window of the UCI Bonn file exactly and "
           f"uniquely; no row from classes 1–3 matched (Fig. 2). Every match came from Bonn label 4 (set B, healthy volunteers, "
           f"eyes closed {c('andrzejak', 'wu_uci')}), always at the same position (samples {bm_off + 1}–{bm_off + 16} of the 178-sample chunk), "
           f"from {int(P.loc[0, 'bonn_unique_rows'])} distinct chunks of {int(P.loc[0, 'bonn_unique_segments'])} distinct segments, "
           f"in the same order as in the UCI file. Class 0 is therefore single-channel scalp EEG sampled at 173.61 Hz from another "
           f"centre, not 16-channel EEG from Bangalore. The oscillation visible in its within-row correlation (Fig. 1, minimum at lag 8, "
           f"recovery by lag 15) matches the eyes-closed alpha rhythm at that sampling rate.")
    D.figure(FIG / "fig_provenance.png", width_in=5.4, title="BEED healthy rows are exact excerpts of Bonn eyes-closed recordings",
             legend="(a) A Bonn set-B one-second chunk (grey, 173.61 Hz) with the identical BEED class-0 row overlaid (blue). (b) Number "
             "of rows per BEED class with an exact match anywhere in the 11 500 × 178 UCI Bonn file.")
    D.h2("4.3 Quantifying temporal leakage")
    D.para(f"Under the random split, the nearest training neighbour of a test row in classes 1–3 was the immediately adjacent row "
           f"of the same recording in {pct(min(nn_adj))}–{pct(max(nn_adj))} of cases (median distance one row), versus "
           f"{pct(NN.loc[0, 'nn_within_1_row'])} for class 0. Fig. 3 shows how macro-F1 declines as rows are allocated to folds in "
           f"increasingly long contiguous blocks: LightGBM fell from {f3(lg_r.f1)} (random) to {f3(lg_b.f1)} (200-row blocks with "
           f"purge) and 1-nearest-neighbour from {f3(nn_r.f1)} to {f3(nn_b.f1)}. The decline tracks the decay of the across-row "
           f"autocorrelation, so near-perfect random-split scores depend largely on near-duplicate neighbours.")
    D.figure(FIG / "fig_block_curve.png", width_in=4.6, title="Performance falls as folds are separated in time", legend=
             "Five-fold cross-validation macro-F1 (mean ± SD) when rows of classes 1–3 are assigned to folds in contiguous "
             "blocks of the given size (block size 1 is a random split). Dashed lines: no purge; solid lines: 50-row purge "
             "(shown for blocks of at least 100 rows, because smaller blocks leave almost no purged training data).")
    D.h2("4.4 Hold-out benchmark under both protocols")
    D.para(f"Table 3 compares all models (bar chart in Supplementary Fig. S10). Under the random protocol, the best four-class model was {b4r.model} on "
           f"{b4r.features} features (macro-F1 {f3(b4r.macro_f1)}, 95% CI {ci(b4r.f1_cci_lo, b4r.f1_cci_hi)}; accuracy "
           f"{f3(b4r.accuracy)}), and kNN on raw values reached {f3(knn_r.macro_f1)}. Under the blocked protocol, the best model "
           f"was {b4b.model} on {b4b.features} features (macro-F1 {f3(b4b.macro_f1)}, CI {ci(b4b.f1_cci_lo, b4b.f1_cci_hi)}; "
           f"accuracy {f3(b4b.accuracy)}, CI {ci(b4b.acc_cci_lo, b4b.acc_cci_hi)}; kappa {f3(b4b.kappa)}), and kNN fell to "
           f"{f3(knn_b.macro_f1)}. The ranking of models changed between protocols. On raw values, the largest random-to-blocked drop was for "
           f"{drops.index[0]} ({f3(drops.iloc[0])}) and the smallest for {drops.index[-1]} ({f3(drops.iloc[-1])}), so models that interpolate "
           f"between near neighbours gained the most from leakage. "
           f"The binary healthy-versus-seizure task was almost perfectly solved under both protocols (best macro-F1 "
           f"{f3(bbr.macro_f1)} random, {f3(bbb.macro_f1)} blocked)." + (
           f" A single engineered feature with one threshold (within-row lag-1 autocorrelation) achieved a blocked-protocol accuracy of "
           f"{f3(R['stumps'].query('protocol==\"blocked\" and feature==\"t_lag1_ac\"').accuracy.iloc[0])}." if R["stumps"] is not None else ""))
    D.para(f"Errors under the blocked protocol were concentrated among the three seizure classes (Fig. 4). In the three-class "
           f"seizure-type task the best model was {b3r.model} ({b3r.features}) with macro-F1 {f3(b3r.macro_f1)} under the random "
           f"protocol, and {b3b.model} ({b3b.features}) with {f3(b3b.macro_f1)} (CI {ci(b3b.f1_cci_lo, b3b.f1_cci_hi)}) under the blocked protocol.")
    trows = []
    h = R["holdout"]
    for m, fs in [(m, "raw") for m in CLASSICAL] + [(m, "raw+eng") for m in CLASSICAL] + [(m, "raw") for m in DEEP]:
        a, b = hrow(R, "4class", "random", m, fs), hrow(R, "4class", "blocked", m, fs)
        if a is None or b is None:
            continue
        trows.append([m, fs, f3(a.macro_f1), f"{f3(b.macro_f1)} ({ci(b.f1_cci_lo, b.f1_cci_hi)})", f3(b.accuracy),
                      f3(b.kappa), f3(b.auc), f3(b.ece)])
    D.table("Four-class hold-out results under random and blocked protocols",
            ["Model", "Features", "Macro-F1 random", "Macro-F1 blocked (95% CI)", "Accuracy blocked", "Kappa blocked",
             "AUC blocked", "ECE blocked"], trows,
            note="Random test set: 1600 rows; blocked test set: 1599 rows. CI: class-stratified cluster bootstrap (2000 resamples). "
                 "AUC, one-vs-rest macro area under the ROC curve; ECE, expected calibration error. Engineered-only results, "
                 "majority baseline (macro-F1 0.100) and per-class metrics are in Supplementary Tables S2–S4.", font_size=8.5,
            col_widths=[2.0, 1.6, 1.6, 3.4, 1.8, 1.8, 1.8, 2.0])
    D.figure(FIG / "fig_confusion.png", "Confusion matrices of the best models per task and protocol",
             "Counts with row-normalised proportions in parentheses. From left: four-class random, four-class blocked, "
             "three-class random and three-class blocked. Grey intensity encodes the proportion; numbers are printed.")
    D.h2("4.5 Repeated cross-validation and statistical comparison")
    def ttline(task, prot):
        g = tt[(tt.task == task) & (tt.protocol == prot)]
        ns = g[g.p_holm >= 0.05].other.tolist()
        return g.best.iloc[0], ns
    b4, ns4 = ttline("4class", "blocked"); b3, ns3 = ttline("3class", "blocked")
    D.para(f"Table 4 summarises nested 5 × 2 CV on the development sets (raw+engineered features for classical models, raw "
           f"values for deep models). Under the blocked protocol, {b4} had the highest mean four-class macro-F1 "
           f"({f3(cv4b.macro_f1_mean)} ± {f3(cv4b.macro_f1_sd)}); after Holm correction, the corrected t-test did not separate it from "
           f"{', '.join(ns4) if ns4 else 'any other model'}{'' if ns4 else ' (all differences significant)'}. For the three-class task, "
           f"{b3} was best ({f3(cv3b.macro_f1_mean)} ± {f3(cv3b.macro_f1_sd)}), statistically indistinguishable from "
           f"{', '.join(ns3) if ns3 else 'no other model'}. Friedman tests rejected equal performance in all settings "
           f"(four-class blocked p = {sci(fr.loc[('4class', 'blocked'), 'friedman_p'])}; three-class blocked p = "
           f"{sci(fr.loc[('3class', 'blocked'), 'friedman_p'])}); mean ranks are shown in Supplementary Fig. S5. Fold-to-fold "
           f"standard deviations were much larger under the blocked protocol ({f3(cv4b_all.macro_f1_sd.min())}–{f3(cv4b_all.macro_f1_sd.max())} versus "
           f"{f3(cv4r_all.macro_f1_sd.min())}–{f3(cv4r_all.macro_f1_sd.max())}). The hold-out winner under the blocked protocol ({b4b.model} on {b4b.features} values) differs from the CV winner on raw+engineered "
           f"features ({b4}); SVM-RBF on raw+engineered features reached only {f3(cvs[(cvs.task=='4class')&(cvs.protocol=='blocked')&(cvs.model=='SVM-RBF')].macro_f1_mean.iloc[0])} in CV. "
           f"A single leakage-safe split therefore cannot establish a definitive ranking; the CV comparison is more reliable.")
    crow = []
    for m in CLASSICAL + ["MLP", "1D-CNN"]:
        def g(task, prot, col):
            r = cvs[(cvs.task == task) & (cvs.protocol == prot) & (cvs.model == m)]
            return r[col].iloc[0] if len(r) else np.nan
        def pv(task, prot):
            r = tt[(tt.task == task) & (tt.protocol == prot) & (tt.other == m)]
            return "best" if not len(r) else f"{r.p_holm.iloc[0]:.3f}"
        crow.append([m, f"{f3(g('4class', 'random', 'macro_f1_mean'))} ± {f3(g('4class', 'random', 'macro_f1_sd'))}",
                     f"{f3(g('4class', 'blocked', 'macro_f1_mean'))} ± {f3(g('4class', 'blocked', 'macro_f1_sd'))}", pv("4class", "blocked"),
                     f"{f3(g('3class', 'random', 'macro_f1_mean'))} ± {f3(g('3class', 'random', 'macro_f1_sd'))}",
                     f"{f3(g('3class', 'blocked', 'macro_f1_mean'))} ± {f3(g('3class', 'blocked', 'macro_f1_sd'))}", pv("3class", "blocked")])
    D.table("Nested 5 × 2 cross-validation macro-F1 and corrected t-tests",
            ["Model", "4-class random", "4-class blocked", "p (Holm), 4-class blocked", "3-class random", "3-class blocked",
             "p (Holm), 3-class blocked"], crow,
            note="Mean ± SD over 10 outer folds. p: Nadeau–Bengio corrected resampled t-test against the best model under the "
                 "blocked protocol, Holm-adjusted; 'best' marks the reference model. Per-fold scores: Supplementary Table S6.", font_size=8.5,
            col_widths=[2.0, 2.4, 2.4, 2.0, 2.4, 2.4, 2.4])
    D.h2("4.6 Calibration")
    cal = h[(h.task == "4class") & (h.protocol == "blocked") & (h.model != "Majority")]
    cal_best = cal.sort_values("ece").iloc[0]
    calr = h[(h.task == "4class") & (h.protocol == "random") & (h.model != "Majority")]
    lgr, lgb_ = hrow(R, "4class", "random", "LightGBM", "raw+eng"), hrow(R, "4class", "blocked", "LightGBM", "raw+eng")
    D.para(f"Calibration deteriorated under the blocked protocol: the median ECE across the ten non-baseline models on the four-class "
           f"task rose from {f3(calr.ece.median())} (random) to {f3(cal.ece.median())} (blocked; computed over all feature sets). "
           f"LightGBM on raw+engineered features, well calibrated under the random protocol (ECE {f3(lgr.ece)}), became overconfident on "
           f"the temporally distant test blocks (ECE {f3(lgb_.ece)}). For the best blocked model, ECE was {f3(b4b.ece)} and the Brier score "
           f"{f3(b4b.brier)}; the lowest blocked ECE was {f3(cal_best.ece)} ({cal_best.model}, {cal_best.features} features), at the cost of "
           f"lower discrimination (Supplementary Fig. S4). Probabilities should therefore be recalibrated on held-out recordings before any clinical use.")
    D.h2("4.7 Interpretability and ablation")
    D.para(f"Under the random protocol, the most influential LightGBM features for the four-class task were {', '.join(top4r[:4])}; "
           f"under the blocked protocol they were {', '.join(top4b[:4])} (Fig. 5). For the three seizure types (blocked), the leading "
           f"features were {', '.join(top3)}. The healthy class was separated almost entirely by smoothness descriptors (Hjorth mobility, "
           f"Higuchi fractal dimension, singular-value-decomposition entropy and lag-1 autocorrelation), which is exactly the source "
           f"shortcut identified in Section 4.2, whereas the seizure classes were separated by raw values of particular columns "
           f"(notably X11, X8 and X15): the model relies on the spatial distribution of voltage across the 16 channels at each time point. "
           f"In the ablation (Supplementary Fig. S6), raw values alone outperformed all engineered features combined under the "
           f"blocked protocol for the four-class task ({f3(ab('4class', 'blocked', 'raw only'))} vs "
           f"{f3(ab('4class', 'blocked', 'engineered only'))} CV macro-F1), because summary statistics computed across electrodes "
           f"discard electrode identity. With genuine temporal context, the seizure types were more separable: 0.25 s multichannel "
           f"band-power windows gave macro-F1 {f3(win.loc['LR', 'macro_f1'])} (LR) under leave-one-block-out CV, although only "
           f"{int(win.loc['LR', 'n_windows'])} windows were available.")
    D.figure(FIG / "fig_shap.png", "Mean absolute SHAP values of the leading LightGBM features",
             "Twelve most influential features (raw+engineered set), stacked by class, for the four-class task under random and "
             "blocked protocols and for the three-class seizure-type task under the blocked protocol. X1–X16 are raw column "
             "values; prefixes t_, f_, n_ and w_ denote time, frequency, nonlinear and wavelet features. Hatching distinguishes classes.")
    D.h2("4.8 Lightweight deployment")
    erows = []
    for m in ["MLP", "1D-CNN", "BiLSTM", "Transformer", "LightGBM"]:
        e = E.loc[m]
        erows.append([m, e.input, f"{int(e.n_params):,}" + (" nodes" if m == "LightGBM" else ""), f"{e.size_kB_fp32:.1f}",
                      f"{e.latency_ms:.2f}", f3(e.macro_f1_blocked_test)])
    D.para(f"Table 5 lists model sizes and latencies. The MLP ({int(mlp_e.n_params):,} parameters, {mlp_e.size_kB_fp32:.0f} kB) "
           f"classified one 16-value sample in {mlp_e.latency_ms:.2f} ms on a single CPU thread with blocked macro-F1 "
           f"{f3(mlp_e.macro_f1_blocked_test)}; all four deep models needed less than {E.loc[DEEP, 'latency_ms'].max() + 0.005:.2f} ms. LightGBM "
           f"achieved a higher blocked macro-F1 ({f3(lgb_e.macro_f1_blocked_test)}) but its pipeline took {lgb_e.latency_ms:.2f} ms per sample, "
           f"dominated by Python feature extraction, which exceeds the 3.9 ms sampling interval at 256 Hz unless samples are batched. "
           f"The MLP therefore offers the best accuracy–cost trade-off for continuous on-device inference.")
    D.table("Size, latency and blocked-protocol performance of deployable models",
            ["Model", "Input", "Parameters", "Size (kB, 32-bit)", "Latency (ms)", "Macro-F1 blocked"], erows,
            note="Latency: median of 500 single-sample predictions on one CPU thread (Intel Xeon virtual CPU), including "
                 "standardisation and, for LightGBM, the 57-feature extraction. LightGBM size approximated from tree nodes. "
                 "Deep models were retrained for profiling, so F1 may differ slightly from Table 3.", font_size=8.5,
            col_widths=[2.2, 3.6, 2.6, 2.4, 2.4, 2.8])

    # ---------------------------------------------------------------- 5 discussion
    D.h1("5 Discussion")
    D.para(f"The main finding is that the public BEED file combines two incompatible data layouts and two data sources. The "
           f"'healthy' class consists of single-channel Bonn snippets, while the seizure classes are multichannel time points "
           f"from continuous recordings. Any classifier can therefore separate healthy from seizure rows from amplitude, smoothness "
           f"or spectral shape, which explains the near-perfect binary results under both protocols and the near-perfect class-0 "
           f"recall in all confusion matrices. This is a dataset artefact; it should not be read as evidence that seizures can be "
           f"detected from 62 ms of EEG.")
    D.para(f"The second finding concerns evaluation. Consecutive seizure rows are 3.9 ms apart and correlated at r ≈ 0.85–0.98, so a "
           f"random split evaluates interpolation between neighbouring samples. Accuracies of 95–99% reported for row-level splits "
           f"{c('seqboost', 'paneru', 'nandatama', 'papernet')} agree with our random-protocol results; under the blocked protocol "
           f"the best four-class macro-F1 fell by {f3(drop_r)} and the ranking of models changed. This agrees with reports of leakage-inflated EEG results {c('brookshire', 'kapoor')}. "
           f"The blocked protocol is still optimistic, because training and test blocks come from the same recordings; since the "
           f"file seems to contain about 7.8 s per seizure class, recording identity and seizure type are confounded, and "
           f"subject-independent evaluation is impossible with the public file. Claims of subject-independent evaluation on this "
           f"file should be treated with caution.")
    D.para(f"Clinically, the features that remained informative under the blocked protocol were the spatial voltage pattern "
           f"across electrodes and its amplitude (Fig. 5). This is plausible (focal seizures involve a restricted region, generalised "
           f"seizures both hemispheres), but the confound with recording identity prevents a physiological interpretation. The windowed analysis showed that adding temporal context improves seizure-type discrimination, "
           f"which suggests that future BEED releases should provide segment-level multichannel data with subject identifiers. "
           f"Finally, the deployment profile shows that an MLP of about 11 000 parameters (43 kB) runs in well under a millisecond and is "
           f"statistically indistinguishable from the best model in blocked CV, so performance on this file is limited by data and "
           f"evaluation rather than by model capacity.")

    # ---------------------------------------------------------------- 6 conclusion
    D.h1("6 Conclusion")
    D.para(f"The public BEED file is not a homogeneous 16-channel dataset: its healthy class is copied from the Bonn eyes-closed set, "
           f"and its seizure classes are autocorrelated multichannel time points. Row-level random splits therefore inflate "
           f"performance substantially. Under a leakage-aware protocol, four-class macro-F1 was {f3(b4b.macro_f1)} rather than "
           f"{f3(b4r.macro_f1)}, and seizure-type macro-F1 was {f3(b3b.macro_f1)} rather than {f3(b3r.macro_f1)}. We recommend that "
           f"BEED studies (i) exclude or replace the healthy class for clinical claims, (ii) use grouped, temporally blocked "
           f"evaluation with purging, (iii) report cluster-bootstrap confidence intervals and calibration, and (iv) state the "
           f"evaluation protocol explicitly.")
    D.h1("7 Limitations and Future Work")
    D.para("The public file contains no subject identifiers, so neither our protocol nor any other can assess generalisation to new "
           "patients; the blocked protocol measures generalisation to unseen time within the same recordings. Our structural "
           "inferences about classes 1–3 (time points of continuous 256 Hz recordings, possibly a bipolar montage) rest on "
           "statistical evidence rather than documentation, and we could not determine whether each seizure class comes from one "
           "or several recordings. The provenance match covers only the Bonn file; classes 1–3 were not searched against other "
           "public databases. Hyper-parameter grids were deliberately compact, and the deep models were small and not exhaustively "
           "tuned. Future work will apply the protocol to the full segment-level BEED recordings if released, test cross-dataset "
           "transfer (for example, to CHB-MIT) and evaluate calibrated, patient-independent seizure-type classifiers.")

    # ---------------------------------------------------------------- back matter (unnumbered)
    D.h1("Data Availability Statement")
    D.para(f"BEED is publicly available from the UCI Machine Learning Repository (doi:10.24432/C5K33B) {c('beed')}. The Bonn-derived "
           f"Epileptic Seizure Recognition file used for the provenance check is available from its creators {c('wu_uci')}. All derived "
           "results, figures and code are provided in Supplementary File S1" +
           (" and will be deposited in a public repository upon acceptance (link withheld for double-blind review)." if anonymous
            else " and at [Repository URL / DOI]. Code licence: [MIT or other licence]."))
    D.h1("Ethics Statement")
    D.para("This study analysed only publicly available, de-identified EEG data released for research use; no new data were "
           "collected and the authors had no access to identifying information. Ethics approval and consent to participate were "
           "therefore not required for this secondary analysis; consent for the original recordings is described by the data "
           "creators. Consent for publication: not applicable.")
    D.h1("Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process")
    D.para("[Placeholder for the authors to finalise.] During this work, the authors used an AI assistant to help write analysis "
           "code, run experiments and draft and edit the manuscript text. The authors reviewed, verified and edited all content "
           "and take full responsibility for the publication. All numerical results were produced by the released code.")
    D.h1("Declaration of Conflict of Interest")
    D.para("The authors declare no conflict of interest. [Authors to confirm.]")
    D.h1("Acknowledgements")
    if anonymous:
        D.para("[Removed for double-blind review.]")
    else:
        D.para("We thank the creators of BEED and of the Bonn EEG database for making their data publicly available. "
               "[Funding information, if any: Funder name, grant number.]")
    # references
    D.h1("References")
    for i, k in enumerate(rf.order, 1):
        D.para(f"[{i}] {BIB[k]}", align="left", size=11)
    D.save(out, title=TITLE)
    return rf.order


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    build(True, out / "BEED_paper_review_anonymised.docx")
    build(False, out / "BEED_paper_camera_ready.docx")
