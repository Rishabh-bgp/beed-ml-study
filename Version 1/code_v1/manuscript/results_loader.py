"""Collect every number quoted in the manuscript directly from results/ CSV files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
DEEP = ["MLP", "1D-CNN", "BiLSTM", "Transformer"]


def f3(x):
    return f"{x:.3f}"


def ci(lo, hi):
    return f"{lo:.3f}–{hi:.3f}"


def load():
    R = {}
    R["prov"] = pd.read_csv(RES / "provenance_structure.csv").set_index("class")
    R["prov_json"] = json.load(open(RES / "provenance_summary.json"))
    R["nn"] = pd.read_csv(RES / "leakage_nearest_neighbour.csv").set_index("class")
    bc = pd.read_csv(RES / "leakage_block_size_curve.csv")
    R["block"] = bc.groupby(["model", "block_size", "gap"]).agg(
        f1=("macro_f1", "mean"), sd=("macro_f1", "std"), acc=("accuracy", "mean")).reset_index()
    h = pd.read_csv(RES / "holdout_metrics.csv")
    R["holdout"] = h
    R["perclass"] = pd.read_csv(RES / "holdout_per_class.csv")
    R["params"] = pd.read_csv(RES / "holdout_best_params.csv")

    def best(task, prot):
        g = h[(h.task == task) & (h.protocol == prot) & (h.model != "Majority")]
        return g.sort_values("macro_f1", ascending=False).iloc[0]
    R["best"] = {(t, p): best(t, p) for t in ["4class", "3class", "binary"] for p in ["random", "blocked"]}
    R["cv"] = pd.read_csv(RES / "cv_summary.csv")
    R["cvfold"] = pd.read_csv(RES / "cv_fold_scores.csv")
    R["ttest"] = pd.read_csv(RES / "stats_corrected_ttest.csv")
    R["friedman"] = pd.read_csv(RES / "stats_friedman.csv")
    R["ablation"] = pd.read_csv(RES / "ablation_feature_groups.csv")
    R["shap"] = {(t, p): pd.read_csv(RES / f"shap_{t}_{p}.csv", index_col=0)
                 for t in ["4class", "3class"] for p in ["random", "blocked"]}
    R["windowed"] = pd.read_csv(RES / "windowed_3class_lobo.csv").set_index("model")
    R["edge"] = pd.read_csv(RES / "edge_deployment.csv").set_index("model")
    sp = RES / "shortcut_stumps.csv"
    R["stumps"] = pd.read_csv(sp) if sp.exists() else None
    return R


def hrow(R, task, prot, model, feats):
    h = R["holdout"]
    g = h[(h.task == task) & (h.protocol == prot) & (h.model == model) & (h.features == feats)]
    return g.iloc[0] if len(g) else None
