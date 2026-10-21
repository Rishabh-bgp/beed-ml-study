# FAQ and limitations

**Why is the random-split score (~0.98) so different from the headline (~0.75)?**  Rows from the same recording are
temporally correlated; a row-level random split puts near-identical neighbours in training and test sets.

**Is the data included?**  Yes — BEED and the Bonn-derived CSV used for the provenance audit.

**Was any edge/wearable hardware used?**  No. Footprint numbers are from a general-purpose CPU.

## Limitations (from `Version 2/CHANGES_v2.md`)
* The primary-model selection rule was fixed after an initial exploratory analysis that included hold-out results; a
  fully nested selection check gave almost the same estimates.
* Integer-lattice artefact in the data.
* The CHB-MIT provenance search is partial (246 of 686 files).
* Deep models may exploit shortcuts.
* Change-point analysis does not support 200 rows per subject; one candidate discontinuity cannot be excluded.
