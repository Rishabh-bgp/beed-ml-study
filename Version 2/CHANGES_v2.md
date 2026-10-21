# BEED paper: changes from version 1 to version 2 (internal note — do not upload)

This file is for Rishabh only. The paper and supplement themselves are written as a **first submission**: they contain no
reference to an earlier version.

Files in this folder:
- `paper_v2/BEED_paper_camera_ready_v2.docx` / `.pdf` (26 pages)
- `paper_v2/BEED_paper_review_anonymised_v2.docx` / `.pdf` (26 pages; for CMT double-blind upload)
- `supplementary_v2/BEED_supplementary_v2.docx` / `.pdf` (29 pages)
- `supplementary_v2/BEED_code_S1_review_v2.zip` (Supplementary File S1, code and results archive, 14.6 MB)

Version 1 in `deliverables/` was not modified. Every number is read from the result files (`revision_v2/results/`,
`revision_v2/REVISION_RESULTS.md`) or from the v1 results by the build scripts in `revision_v2/manuscript_v2/`
(`build_paper_v2.py`, `build_supplement_v2.py`, `v2data.py`, `bib_v2.py`; neutral figures from `make_figs_neutral.py`,
confusion matrices from `make_confusion_v2.py`). Supplementary numbering is generated: the supplement build writes
`supp_map.json` and the paper reads every "Supplementary Table/Fig./Section S…" number from it, so all cross-references
resolve (checked automatically after the final build).

## Final delivery fixes (first-submission clean-up, 8 Oct 2026)
1. **No version history in the manuscript.** Removed every mention of a "first version", "version 1", "original
   headline/models/runs", "superseded", "mislabelled 77" and "Table 5 of the first paper" from the paper and the supplement.
   Current facts only, e.g. "73 features (16 raw + 57 engineered)". The test-selected comparator is now described as
   "SVM-RBF (raw), the best of about 16 classical configurations on the final-20% block, i.e. selected on the test block".
   Limitations now say: "the primary-model selection rule was fixed after an initial exploratory analysis that included
   hold-out results; a fully nested selection check inside the rolling origin gave almost the same estimates".
   Figures with old labels were re-plotted with neutral labels (`manuscript_v2/figs/`).
2. **Supplement.** "Changes from version 1" note deleted. Removed v1 Table S12 (uncorrected t-tests; replaced by the
   Holm table, now S33) and v1 Table S16 (leave-one-block-out windowed analysis; replaced by S29). Remaining v1 tables are
   presented neutrally ("fixed default hyper-parameters" instead of "version-1"). Section S7 renamed "Additional analyses".
   Everything renumbered; new map: Tables S1–S33, Figures S1–S20, Sections S1–S7 (S7.1–S7.9).
3. **Length.** No page or word limit exists in the NIELIT Bio-AI Conclave 2026 guidelines (see below). The paper was
   tightened from 34 to 26 pages (double spacing, references included; body ≈ 5,400 words; abstract 245 words, was 333).
   Moved to the supplement (no result dropped):
   - full Table 4 (all 38 configurations) → Table S25; the paper keeps a compact Table 4 (10 raw-value models ×
     blocked CV / rolling origin / Holm p, both tasks);
   - detailed Table 1 (data described per study) → Table S18; the paper keeps a condensed Table 1 (study, task, protocol,
     result);
   - Fig. 4 (CV vs rolling origin vs hold-out, all configurations) → Fig. S13;
   - Fig. 5 (confusion matrices) → Fig. S14;
   - Fig. 6 (fingerprint baselines) → Fig. S15;
   - Fig. 7 (SHAP) → Fig. S8;
   - Table 6 (computational footprint) → Table S31 (§4.10 keeps the key numbers);
   - verbatim UCI quotations, change-point details, windowed (R6) and deep-tuning (R7) details shortened in the text and
     referenced to S7.1, S7.2, S7.6 and S7.7.
   Also shortened: abstract, contributions, §2.2 counter-example discussion, §2.3, §3.3–3.8, §4.1, §4.5–4.10,
   Discussion, RQ answers and Limitations.
   Paper now has Figures 1–3 and Tables 1–5.
4. **Code archive S1** rebuilt as `BEED_code_S1_review_v2.zip` (109 files, 14.6 MB; old v1 zip 24 MB):
   - `code/` = v1 pipeline (run_all.py, src/, results/, figures/, data/, analysis.ipynb, README.md, requirements.txt);
   - `code/extended_analyses/` = the additional-analysis scripts (r1, r2, r5, r6, r7, r8, summarise, figures,
     figures_final, figure_confusion, common, deep2, edf), result CSV/JSON files, publication figures and notes
     (UCI documentation snapshot, CHB-MIT record lists);
   - excluded: `manuscript/` (v1 paper builders, including `authors_camera_ready.json`), `results/preds/` (both
     levels), `_ctx_cache.pkl`, logs, `_r7_locks`, `_*` scratch files (incl. `_R7_first3_superseded.csv`,
     `_r5_scan_v1*`), `__pycache__`;
   - hard-coded `/workspace/...` paths replaced by paths relative to the archive; script docstrings/comments neutralised;
     the v1 label "raw+engineered (77)" corrected to 73 in `src/beed/experiments.py` and `results/edge_deployment.csv`;
   - README: manuscript row and build command removed; new section "Extended analyses" with rerun commands, extra
     requirements and a note on column names (`original_default_*`, `original_headline_test_selected`, role labels);
   - identity check: no name, IIIT/Bhagalpur, roll number, email of the author, ORCID, `/Users/` or `/home/` path anywhere in
     the archive (only dataset creators' and UCI contact emails inside the saved UCI page).
   The paper cites it as "Supplementary File S1" (no file name), and §3.9 now says that run_all.py regenerates the core
   analyses and the scripts in `extended_analyses/` the rest.
5. Rebuilt DOCX + PDF of both paper versions and the supplement; every page checked visually; anonymisation re-checked
   (text, DOCX XML, core.xml creator/lastModifiedBy empty, PDF Author empty, zip).

**Guideline check.** `/workspace/beed/guidelines.txt` (from SubmissionGuidelines.pdf) gives formatting rules only
(§6 "Use double-line spacing", "Include page numbers"; §9 a concise abstract; §13/14 titles ≤ 15 words and legends
≤ 300 words) and no page or word limit; the conference website (https://nielitbioai.org/) also states none. The
"not exceeding 6000 words" rule found online belongs to a different event (SBBS "BIOAI-2026",
sbbsindia.in/docs/BioAi-2026__compressed.pdf). The paper body is ≈ 5,400 words in any case.

## Scientific content added in version 2 (supplementary numbers as in the final build)

### R1. Pre-specified, leakage-aware headline
- Primary model chosen without any test block (best mean macro-F1 in nested purged blocked 5×2 CV): LightGBM on raw
  values for both tasks.
- Primary estimate (five-fold rolling origin): four classes 0.748 ± 0.080 (t-interval 0.648–0.847); three seizure types
  0.677 ± 0.104 (0.548–0.805). Blocked CV 0.777 ± 0.048 / 0.720 ± 0.053; final-20% hold-out 0.810 (CI 0.771–0.845) /
  0.752 (0.694–0.806). Random split about 0.98. Fully nested selection: 0.747 / 0.665.
- The v1 headline 0.842 / 0.776 (SVM-RBF raw) is now only a comparator, described as test-selected (blocked CV / rolling
  origin 0.755 / 0.754 and 0.685 / 0.689).
- Table 3 (primary model), compact Table 4 + Table S25 (all configurations, Holm p), Fig. S13, Fig. S14, Tables S23–S24.
- Healthy class: per-class F1 ≥ 0.99 (0.999 random, 0.995 blocked).

### R2. Documentation and splice analysis
- §3.1 summarises the UCI documentation and its contradictions with the file (verbatim quotations in Table S20).
- Change-point analysis (§3.2, §4.1; Tables S21–S22, Fig. S12): no support for 200 rows per subject; one candidate
  discontinuity (class 1, row 1900) cannot be excluded.
- Bonn wording neutral ("identical to", "appears to be taken from").

### R3 / R10. Literature and references
- Table 1: 8 studies with protocol as implemented (detailed version Table S18); AISI 2024 [15], Shah & Toma [16], IJIEI
  2024 [17] added; counter-examples [13], [14], [16] discussed in §2.2.
- Corrected entries (Roy et al. SPMB 2020, PaperNet IDAA 2025, EpilepsyFormer authors, UCI Bonn DOI, Lundberg 2020 SHAP,
  preprints marked, SeqBoostNet 96.71% = mean of six binary tasks). 36 references, numbered by first appearance.
- Further 2024–2026 BEED reports: Table S19.

### R4. Recording-fingerprint confound
- §3.6, §4.6, Table 5; Tables S26–S27, Fig. S15.
- |x| alone: 0.720 / 0.648 (LightGBM) and 0.702 / 0.627 (LR) in the rolling origin, about 85–95% of the primary model's
  above-chance performance. Pre-specified test vs **logistic regression on |x|**: +0.046 to +0.054, Holm p = 0.15–0.16
  (not significant). Per-recording z-scoring costs the 3-class task 0.10–0.13; per-row normalisation costs nothing.
  "Artefact" (non-de-quantised) rows excluded everywhere.

### R5. CHB-MIT provenance search
- §3.2, §4.2, Limitations; §S7.5, Table S28, Fig. S16. 246 of 686 files, 282 h, all 24 subjects; positive control 1.00;
  no file above 0.42 (threshold 0.80); described as partial.

### R6. Windowed analysis
- §4.7 as a separate, non-comparable setting; 0.97 (CI 0.90–1.00) vs 0.77 single-row and 0.89 window vote; blocked CV
  0.92 / 0.67; advantage over window vote not significant. Table S29, Fig. S17.

### R7. Deep models tuned
- 12-point grid, inner purged blocked hold-out (§3.5); shortcuts disclosed. BiLSTM hold-out 0.468 → 0.650; no deep model
  beats LightGBM (raw) in blocked CV. Table S30, Fig. S18.

### R8. Computational footprint on a general-purpose CPU
- Edge/wearable claims removed. §4.10 (Table S31, Fig. S19): 0.05–0.74 ms per sample (Python), 16 kB–2.3 MB ONNX;
  profiled tuned LightGBM 73 features, 3.97 ms per sample, 87.5% feature extraction; no edge hardware used.

### R9. Uncertainty and multiplicity
- Cluster bootstrap (chunk-size insensitive) plus rolling-origin t-interval and range (Table S32, Fig. S20); Holm across
  all 72 comparisons (Table S33); Friedman/Nemenyi descriptive only (Table S12, Fig. S5).

### Other
- Conclusion recommendation (v): pre-specify the model and report rolling-origin or CV estimates.
- Limitations: selection-rule timing, integer-lattice artefact, partial CHB-MIT search, deep-model shortcuts.
- SHAP/ablation labelled as a descriptive LightGBM with default settings.
- Camera-ready byline: Rishabh Aryan's name, affiliation, email, ORCID; co-author/supervisor placeholders remain.
- Anonymised version: author block and acknowledgements removed, repository link withheld, DOCX creator/lastModifiedBy
  and PDF Author empty, identity search clean (DOCX, PDF, supplement, zip).

## Still to do (Rishabh)
- **Placeholders:** co-author and supervisor names and ORCIDs; repository URL / DOI and code licence (paper and archive
  README "[choose licence, e.g. MIT]"); funding; AI-use declaration ("[Placeholder…]"); COI ("[Authors to confirm.]").
- **Title** still contains "Lightweight" (kept to match the CMT registration).
- **File names** contain "_v2" (e.g. `BEED_code_S1_review_v2.zip`). Rename before uploading to CMT if you do not want a
  version suffix visible to reviewers; the documents themselves do not refer to the file names.
- **Archive identifiers:** some CSV column names / role labels and a few code identifiers still read "original…" (e.g.
  `original_default_holdout`, "original headline (test-selected)", "LightGBM (original default, Table 5)"); renaming
  them would break the scripts, so the README explains them instead. The archive also keeps the v1 pipeline outputs
  `stats_corrected_ttest.csv` and `windowed_3class_lobo.csv` (regenerated by run_all.py; not cited).


## Submission-guideline audit (16 Oct 2026)
- Checked against every section of the NIELIT Bio-AI Conclave 2026 full-paper guidelines; see submission_checklist.md.
- Data Availability Statement now gives the software details asked for in guideline section 15.
- Added separate Funding and Author Contributions sections (placeholders; removed in the anonymised copy).
- AI-use declaration placeholder reworded to state accurately that AI agents wrote the code, ran the experiments and drafted the text.
- Table notes are now double-spaced. Added cover_letter/ (DOCX and PDF). No scientific number changed.

## Plain-language explainer rebuilt for version 2 (17 Oct 2026)
- New `explainer_v2/BEED_simple_explainer_v2.docx` and `.pdf` (4 pages) replace the version 1 explainer, which still gave the old figures. Every number now comes from the V2 paper and result files: the main result is the fair (rolling-origin) score of 0.75 ± 0.08 for four classes and 0.68 ± 0.10 for three classes. The 0.842/0.776 figures appear only as test-selected numbers. The section on small devices (43 KB, 0.08 ms, wearable) is removed. The windowed result is given as 0.97 on the hold-out and 0.92 ± 0.10 in blocked CV, as a separate setting. The recording-fingerprint result (0.72/0.65 using only |x|) is added, and the summary and Q&A are updated. There is no author name or affiliation in the text or the document properties. The explainer is not a submission file.
