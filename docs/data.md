# Data

Both datasets are **included** in this repository.

| File | Description |
|---|---|
| `Version 1/code_v1/data/BEED_Data.csv` | BEED file from the UCI repository (doi:[10.24432/C5K33B](https://doi.org/10.24432/C5K33B), CC BY 4.0) |
| `Version 1/code_v1/data/external/bonn_uci_epileptic_seizure_recognition.csv` | Bonn-derived UCI *Epileptic Seizure Recognition* CSV, used for the provenance audit (the pipeline downloads and SHA-256-verifies it if absent) |

## Structure (from `results/provenance_summary.json`)
* 4 classes × 2,000 rows = 8,000 rows; label changes at rows 1999, 3999, 5999.
* 0 duplicate rows and 0 duplicate feature vectors.
* Class-0 rows match Bonn segments, 16–23 rows per segment.

## Licences
The datasets keep their original licences and are not covered by this repository's MIT licence.
