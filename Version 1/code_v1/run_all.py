#!/usr/bin/env python
"""Reproduce the full BEED study: python run_all.py [stage ...]

Stages: provenance leakage holdout cv stats ablation shap windowed edge figures
(default: all, in that order). Outputs go to results/ and figures/."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from beed import experiments as X, plots, config as C

ALL = ["provenance", "leakage", "holdout", "holdout_extra", "cv", "stats", "ablation", "shap", "windowed", "edge", "figures"]


def main(stages):
    np.random.seed(C.SEED)
    ctx = None
    for st in stages:
        t = time.time()
        print(f"=== stage {st} ===", flush=True)
        if st not in ("stats", "figures") and ctx is None:
            ctx = X.Ctx()
        if st == "provenance": X.stage_provenance(ctx)
        elif st == "leakage": X.stage_leakage(ctx)
        elif st == "holdout": X.stage_holdout(ctx)
        elif st == "holdout_extra": X.stage_holdout_extra(ctx)
        elif st == "cv": X.stage_cv(ctx)
        elif st == "stats": X.stage_stats()
        elif st == "ablation": X.stage_ablation(ctx)
        elif st == "shap": X.stage_shap(ctx)
        elif st == "windowed": X.stage_windowed(ctx)
        elif st == "edge": X.stage_edge(ctx)
        elif st == "figures":
            ctx = ctx or X.Ctx()
            plots.make_all(ctx)
        print(f"=== stage {st} finished in {time.time() - t:.1f} s ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ALL)
