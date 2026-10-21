# Reproducing the results

Environment used: Linux, Python 3.13.5, 8 virtual CPUs, 16 GB RAM, no GPU. Pinned packages in
`Version 1/code_v1/requirements.txt`.

```bash
cd "Version 1/code_v1"
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
python run_all.py                          # all stages (about 1–1.5 h on 8 CPU cores)
python run_all.py provenance leakage       # or individual stages
jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
mkdir -p out && cd manuscript && python build_paper.py ../out && python build_supplement.py ../out
```

Stages: `provenance leakage holdout holdout_extra cv stats ablation shap windowed edge figures`.
Set `BEED_N_JOBS` to change parallelism. Seeds are fixed (`SEED = 42`).

## Version 2 extended analyses
Unzip `Version 2/supplementary_v2/BEED_code_S1_review_v2.zip`; `code/extended_analyses/` holds the additional scripts,
results and figures, with rerun commands in the archive README.
