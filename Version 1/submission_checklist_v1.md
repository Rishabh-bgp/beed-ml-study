# Submission checklist: NIELIT Bio-AI Conclave 2026, Track 5

Manuscript: *Shortcuts and Temporal Leakage in the Bangalore EEG Epilepsy Dataset: a Provenance Audit and Leakage-Aware
Benchmark of Lightweight Seizure-Type Classifiers*

Each requirement in `guidelines.txt` (sections 1–23) is mapped to where it is met. ✅ = done; ⚠️ = needs an author
decision or action before submission.

## Files to upload

| Item | File(s) | Status |
|---|---|---|
| Manuscript, editable + identical PDF (§1, §5) | `paper/BEED_paper_review_anonymised.docx` + `.pdf` (upload these for review) | ✅ |
| Camera-ready manuscript (after acceptance) | `paper/BEED_paper_camera_ready.docx` + `.pdf` (placeholders to fill in) | ⚠️ fill in placeholders |
| Supplementary document, one DOCX + identical PDF (§1, §19) | `supplementary/BEED_supplementary.docx` + `.pdf` | ✅ |
| Code/results archive (Supplementary File S1) | `supplementary/BEED_code_S1_review.zip` (anonymised: the camera-ready author file is left out) | ⚠️ §1 allows only one supplementary DOCX+PDF pair, so upload the zip only if CMT accepts an extra file; otherwise deposit it at acceptance |
| Cover letter / author declaration (§22) | `cover_letter.docx` (+ `cover_letter.pdf`) | ⚠️ fill in date, names, signature |

The PDFs were produced from the DOCX files with LibreOffice (`soffice --headless --convert-to pdf`), so their content is identical.

## Requirement mapping

| § | Requirement | Where / how it is met | Status |
|---|---|---|---|
| 1 | Original full research paper; one DOCX + one PDF with identical content; the same for the supplement | See the files table above | ✅ |
| 2 | Primary track identified; any secondary area given in the cover note | Track 5 (healthcare AI) is stated under the title. The cover letter gives Track 3 (reproducibility) as a secondary area | ✅ |
| 3 | Originality, authorship, acknowledgement of sources, COI | The contribution is a provenance audit plus a leakage-aware protocol, positioned against prior BEED studies (Section 2.2, Table 1). All datasets, software and methods are cited [refs 1–33]. A COI declaration is included | ⚠️ authors to confirm authorship and originality |
| 4 | At least one author registers if accepted | Confirmed in the cover letter | ⚠️ author action |
| 5 | Editable Word file | `.docx` built with python-docx; the editable source is kept in `code/manuscript/` | ✅ |
| 6 | English (British spelling), double spacing, page numbers | Body text uses 2.0 line spacing. A PAGE field in the footer numbers every page | ✅ |
| 6 | Only two heading levels numbered (1, 1.1), flush left; third level as bold run-in headings | Headings run 1–7 and 2.1–4.8. "Research questions", "Contributions", "Random protocol" and "Blocked protocol" are bold run-in headings. Back-matter headings are unnumbered | ✅ |
| 6 | No unnecessary manual page breaks | None inserted | ✅ |
| 6 | SI units; symbols embedded correctly | Units are Hz, s, ms and kB. Superscripts (e.g. 10⁻¹⁰) are true superscripts | ✅ |
| 6 | Reference manager | References come from a single scripted bibliography (`code/manuscript/bib.py`), numbered automatically by first citation | ✅ (export to Zotero if preferred) |
| 7 | Recommended structure | Title → (Authors) → Abstract → Keywords → 1 Introduction (with research questions RQ1–RQ4, objectives and contributions) → 2 Related Work → 3 Materials and Methods → 4 Results → 5 Discussion → 6 Conclusion → 7 Limitations and Future Work → Data Availability → Ethics → Declaration of Generative AI → Declaration of Conflict of Interest → Acknowledgements → References | ✅ |
| 8 | Title page: full title, author names, affiliation, department, city/country, corresponding author marked with * plus email, ORCID | **Camera-ready copy:** [Author Name]^1,*, [Co-author Name]^1, [Supervisor Name]^1; [Department of … (M.Tech, AI and Data Science)], [IIIT Bhagalpur], [Bhagalpur], [India]; "*Corresponding Author" line with [Email] and [ORCID]. **Review copy:** author block replaced by "[Author names and affiliations removed for double-blind review]" | ⚠️ fill in placeholders. Note: the double-blind/CMT instruction comes from the call for papers, not `guidelines.txt` (§8 asks for author details on page 1), so confirm which copy CMT expects |
| 9 | Abstract covers context, problem, method, results and conclusion, with no citations or unexplained abbreviations; keywords follow | Abstract is 265 words with no citations; every term is spelled out (macro-averaged F1 score; Hz is an SI unit). Eight keywords follow it | ✅ |
| 10 | Introduction: background, gap, previous work, research questions, objectives/contributions | Section 1 (two paragraphs, plus run-in "Research questions" (RQ1–RQ4) and "Contributions" (i)–(v)) | ✅ |
| 11 | Methodology: design, data, preprocessing, models, training, validation, environment, hardware, statistics, metrics | 3.1 Dataset; 3.2 Audit; 3.3 Evaluation protocols (nested CV, purge); 3.4 Features; 3.5 Models and training; 3.6 Metrics and statistics; 3.7 Interpretability, ablation and deployment profiling; 3.8 Software and hardware. Grids are in Supplementary Table S1 | ✅ |
| 12 | Results numbered and referenced; observation kept separate from interpretation; statistics given | Section 4 (results with CIs, corrected t-tests, Friedman) is kept apart from Section 5 (interpretation, comparison with prior work, implications) | ✅ |
| 13 | Figures numbered in order of first mention; titles ≤15 words, legends ≤300 words; labelled axes; ~300 dpi | Main Figs. 1–5 and Supplementary Figs. S1–S10. The build script checks the word limits (assertions). All PNGs are rendered at 300 dpi and shown at ≤6.2 in width | ✅ |
| 13 | Permission for copyrighted figures | All figures were generated by the authors from public data | ✅ |
| 14 | Tables numbered and cited in order; titles above and ≤15 words; notes ≤300 words; editable; no colour-only encoding | Tables 1–5 in the main text and S1–S19 in the supplement are native Word tables in black and white | ✅ |
| 15 | Data Availability statement; software details (name, repository, version, language, OS, dependencies, licence) | Data Availability Statement (UCI DOI 10.24432/C5K33B; Bonn-derived file). Section 3.8 gives versions and hardware. `code/README.md` and `code/requirements.txt` cover the rest. The repository link is withheld in the review copy; in the camera-ready copy it is "[Repository URL / DOI]" | ⚠️ choose a licence and add the repository DOI at camera-ready |
| 16 | Ethics: approval, consent, privacy | Ethics Statement: secondary analysis of public, de-identified data, so approval and consent were not required; consent for publication not applicable | ✅ (authors to confirm the institution's policy) |
| 17 | Declarations: generative AI and conflict of interest | Both are present. The generative-AI text is an honest placeholder saying an AI assistant helped with code, experiments and drafting | ⚠️ authors to finalise the wording |
| 18 | Numeric citations [1], [2,3] in order of first appearance; "Author AA. Title. Journal. Year;Vol:pages." format; no bare URLs in text | `Refs` class numbers citations by first appearance. All 33 references were verified (Crossref, publisher or repository pages). No URLs appear in the running text | ✅ |
| 19 | Supplementary files named and cited in the text | "Supplementary Table S…/Fig. S…/File S1" are cited at the relevant points | ✅ |
| 20 | Peer review / revisions | All results regenerate from `code/run_all.py`; the documents rebuild from `code/manuscript/*.py` | ✅ |
| 21 | Copyright/permissions | No third-party material reproduced. BEED (UCI) and the Bonn-derived file are cited | ✅ |
| 22 | Cover letter declarations (originality, no dual submission, approval, consent, COI, permissions, registration) | `cover_letter.docx` covers all seven points, plus AI disclosure | ⚠️ sign and date |
| 23 | Paper ID | Assigned after acceptance (format BioAi-2026_Paper_YYY) | n/a |

## Double-blind checks on the review copy

- No author names, affiliations, emails, ORCIDs or identifying acknowledgements ("[Removed for double-blind review]").
- Repository link withheld. The DOCX core properties (author, last-modified-by) are blank.
- Searches of the review PDF, the supplement and the review code zip for the institution and author names return no hits.
- The camera-ready author block lives only in `code/manuscript/authors_camera_ready.json`, which is excluded from `supplementary/BEED_code_S1_review.zip`.

## Before submitting (author actions)

1. Fill in all placeholders in the camera-ready copy and the cover letter: [Author Name], [Co-author Name], [Supervisor Name], department, [Email], [ORCID], [Date], [Signature], licence, repository DOI.
2. Finalise the generative-AI declaration and confirm the conflict-of-interest statement.
3. Confirm the double-blind requirement and the CMT upload rules (in particular whether the code zip may be uploaded).
4. Proof-read once in Microsoft Word: the PDFs were rendered with LibreOffice, so pagination may differ slightly in Word. Re-export the PDF from Word if the pages differ.
5. Length: about 17 double-spaced pages up to the end of Limitations (title page and abstract included), 18 up to the back matter, and 21 with references. If the organisers set a page limit, Table 1 or Section 4.6 can move to the supplement.
6. Deadline: 30 October 2026 (from the call for papers).
