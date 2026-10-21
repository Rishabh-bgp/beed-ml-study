"""All experiments of the study. Each stage writes CSV/NPZ outputs to results/."""
import json
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from . import config as C, data, features, models, deep, evaluation as E

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
PRED = C.RES / "preds"
PRED.mkdir(exist_ok=True, parents=True)

TASKS = {  # name -> (classes included, label mapping)
    "4class": ([0, 1, 2, 3], {0: 0, 1: 1, 2: 2, 3: 3}),
    "binary": ([0, 1, 2, 3], {0: 0, 1: 1, 2: 1, 3: 1}),
    "3class": ([1, 2, 3], {1: 0, 2: 1, 3: 2}),
}
TASK_NAMES = {"4class": ["H", "GS", "FS", "SE"], "binary": ["Healthy", "Seizure"],
              "3class": ["GS", "FS", "SE"]}


class Ctx:
    """Loaded data, groups and features shared by all stages."""
    def __init__(self):
        self.df, self.X, self.y, self.pos = data.load_beed()
        data.ensure_bonn()
        self.bonn_match, _ = data.match_to_bonn(self.X, self.y)
        self.groups = data.make_groups(self.y, self.pos, self.bonn_match)
        self.fsets = features.build_feature_sets(self.X)
        # clusters for the cluster bootstrap: Bonn segment (class 0) / 50-row chunks (classes 1-3)
        self.clusters = np.array([g if c == 0 else f"c{c}_k{p // 50}"
                                  for g, c, p in zip(self.groups, self.y, self.pos)], dtype=object)

    def task(self, name):
        cls, mp = TASKS[name]
        idx = np.where(np.isin(self.y, cls))[0]
        yt = np.full(len(self.y), -1)
        yt[idx] = [mp[v] for v in self.y[idx]]
        return idx, yt


def val_split(protocol, y, pos, groups, tr, frac=0.15, seed=C.SEED):
    """Validation subset (positions within tr) for early stopping of deep models."""
    rng = np.random.RandomState(seed)
    if protocol == "random":
        return rng.choice(len(tr), int(frac * len(tr)), replace=False)
    g = np.unique(groups[tr])
    vg = rng.choice(g, max(1, int(frac * len(g))), replace=False)
    v = tr[np.isin(groups[tr], vg)]
    rest = data.purge(np.setdiff1d(tr, v), v, y, pos)
    loc = np.full(len(y), -1); loc[tr] = np.arange(len(tr))
    # rows purged from 'rest' are simply not used for training: mark them as validation-excluded
    keep_train = np.isin(tr, rest)
    return loc[v], keep_train


def fit_classical(name, F, y, tr, inner, n_jobs=C.N_JOBS):
    est, grid = models.get_models()[name]
    if grid:
        gs = GridSearchCV(clone(est), grid, cv=inner, scoring="f1_macro", n_jobs=n_jobs, refit=True)
        gs.fit(F[tr], y[tr])
        return gs.best_estimator_, gs.best_params_
    return clone(est).fit(F[tr], y[tr]), {}


def fit_deep(arch, X, yt, tr, protocol, ctx, n_cls, threads=4, seed=C.SEED):
    vs = val_split(protocol, yt, ctx.pos, ctx.groups, tr, seed=seed)
    if protocol == "random":
        m = deep.DeepClassifier(arch, n_cls=n_cls, threads=threads, seed=seed).fit(X[tr], yt[tr], vs)
    else:
        v, keep = vs
        use = np.where(keep | np.isin(np.arange(len(tr)), v))[0]
        loc = np.full(len(tr), -1); loc[use] = np.arange(len(use))
        m = deep.DeepClassifier(arch, n_cls=n_cls, threads=threads, seed=seed).fit(
            X[tr][use], yt[tr][use], loc[v])
    return m


# ----------------------------------------------------------------------------------------------
def stage_provenance(ctx):
    X, y = ctx.X, ctx.y
    bm = ctx.bonn_match
    bm.to_csv(C.RES / "provenance_bonn_matches.csv", index=False)
    rows = []
    for c in range(4):
        Xc = X[y == c]
        Cc = np.corrcoef(Xc.T)
        diag_means = [np.mean(np.diag(Cc, k)) for k in range(1, 16)]
        diag_sd1 = np.std(np.diag(Cc, 1))
        row_lag = {k: np.mean([np.corrcoef(Xc[:-k, j], Xc[k:, j])[0, 1] for j in range(16)])
                   for k in (1, 5, 10, 50)}
        d = np.linalg.norm(np.diff(Xc, axis=0), axis=1)
        m = bm[bm.beed_class == c] if len(bm) else bm
        rows.append({"class": c, "n_rows": len(Xc), "amp_sd": Xc.std(),
                     "within_row_lag1_corr": diag_means[0], "within_row_lag4_corr": diag_means[3],
                     "within_row_lag8_corr": diag_means[7], "within_row_lag15_corr": diag_means[14],
                     "lag1_corr_sd_across_positions": diag_sd1,
                     "across_row_lag1_corr": row_lag[1], "across_row_lag5_corr": row_lag[5],
                     "across_row_lag10_corr": row_lag[10], "across_row_lag50_corr": row_lag[50],
                     "large_jumps_gt8xmedian": int((d > 8 * np.median(d)).sum()),
                     "bonn_exact_matches": len(m),
                     "bonn_labels": json.dumps(m.bonn_label.value_counts().to_dict()) if len(m) else "{}",
                     "bonn_offsets": json.dumps({int(k): int(v) for k, v in m.offset.value_counts().items()}) if len(m) else "{}",
                     "bonn_unique_rows": m.bonn_row.nunique() if len(m) else 0,
                     "bonn_unique_segments": m.bonn_segment.nunique() if len(m) else 0,
                     "bonn_order_preserved": bool(np.all(np.diff(m.bonn_row.to_numpy()) > 0)) if len(m) > 1 else False})
    pd.DataFrame(rows).to_csv(C.RES / "provenance_structure.csv", index=False)
    dup = {"duplicate_rows": int(ctx.df.duplicated().sum()),
           "duplicate_feature_vectors": int(ctx.df[data.COLS].duplicated().sum()),
           "label_changes_at_rows": np.where(np.diff(y) != 0)[0].tolist(),
           "class_counts": np.bincount(y).tolist(),
           "rows_per_bonn_segment_min": int(bm.groupby("bonn_segment").size().min()) if len(bm) else None,
           "rows_per_bonn_segment_max": int(bm.groupby("bonn_segment").size().max()) if len(bm) else None}
    json.dump(dup, open(C.RES / "provenance_summary.json", "w"), indent=2)
    return pd.DataFrame(rows)


def stage_leakage(ctx):
    """(i) nearest-neighbour temporal adjacency under a random split; (ii) block-size curve."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    X, y, pos = ctx.X, ctx.y, ctx.pos
    tr, te = data.random_holdout(y)
    sc = StandardScaler().fit(X[tr])
    nn_ = NearestNeighbors(n_neighbors=1).fit(sc.transform(X[tr]))
    _, j = nn_.kneighbors(sc.transform(X[te]))
    nbr = tr[j[:, 0]]
    rows = []
    for c in range(4):
        m = y[te] == c
        same = y[nbr[m]] == c
        dist = np.abs(pos[nbr[m]] - pos[te][m])
        rows.append({"class": c, "nn_same_class": same.mean(),
                     "nn_within_1_row": np.mean(same & (dist <= 1)),
                     "nn_within_5_rows": np.mean(same & (dist <= 5)),
                     "median_row_distance": float(np.median(dist[same])) if same.any() else np.nan})
    pd.DataFrame(rows).to_csv(C.RES / "leakage_nearest_neighbour.csv", index=False)

    # block-size curve: 5-fold CV where classes 1-3 are split into contiguous blocks of size B
    F = ctx.fsets["raw+eng"].values
    out = []
    from lightgbm import LGBMClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    mdl = {"LightGBM (raw+eng)": (lambda: LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                         n_jobs=C.N_JOBS, random_state=C.SEED, verbose=-1), F),
           "1-NN (raw)": (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(1)), X)}
    idx = np.arange(len(y))
    for B in [1, 5, 25, 100, 200, 400]:
        for gap in ([0] if B == 1 else [0, C.GAP]):
            g = np.array([f"c{c}_b{p // B}" if c != 0 else ctx.groups[i]
                          for i, (c, p) in enumerate(zip(y, pos))], dtype=object)
            if B == 1:
                sp = list(data.random_cv(y, idx, 5, 1))
            else:
                sp = list(data.blocked_cv(y, pos, g, idx, 5, 1, gap=gap))
            for name, (mk, Fm) in mdl.items():
                for k, (a, b) in enumerate(sp):
                    p = mk().fit(Fm[a], y[a]).predict(Fm[b])
                    mm = E.metrics(y[b], p)
                    out.append({"block_size": B, "gap": gap, "model": name, "fold": k, **mm})
    pd.DataFrame(out).to_csv(C.RES / "leakage_block_size_curve.csv", index=False)


def stage_holdout(ctx, tasks=("4class", "binary", "3class")):
    """Hold-out evaluation with protocol-respecting inner-CV tuning on the development set."""
    rows, pcs, params = [], [], []
    for task in tasks:
        idx, yt = ctx.task(task)
        n_cls = len(TASK_NAMES[task])
        fsets = ["raw", "eng", "raw+eng"] if task == "4class" else ["raw+eng"]
        for prot in ["random", "blocked"]:
            tr, te = data.holdout(prot, ctx.y, ctx.pos, ctx.groups, idx)
            inner = data.inner_cv(prot, ctx.y, ctx.pos, ctx.groups, tr)
            for fs in fsets:
                F = ctx.fsets[fs].values
                for name in models.CLASSICAL:
                    t0 = time.time()
                    est, bp = fit_classical(name, F, yt, tr, inner)
                    fit_s = time.time() - t0
                    proba = est.predict_proba(F[te]); pred = proba.argmax(1)
                    rows.append(_row(task, prot, fs, name, yt[te], pred, proba, ctx.clusters[te], fit_s))
                    params.append({"task": task, "protocol": prot, "features": fs, "model": name,
                                   "best_params": json.dumps(bp)})
                    np.savez_compressed(PRED / f"{task}_{prot}_{fs}_{name}.npz", idx=te, y=yt[te],
                                        proba=proba)
                    pcs.append(E.per_class(yt[te], pred, TASK_NAMES[task]).assign(
                        task=task, protocol=prot, features=fs, model=name))
                    print(task, prot, fs, name, round(rows[-1]["macro_f1"], 4), flush=True)
            deep_archs = list(deep.DEEP) if task == "4class" else ["MLP", "1D-CNN"]
            for arch in deep_archs:
                t0 = time.time()
                m = fit_deep(arch, ctx.X, yt, tr, prot, ctx, n_cls, threads=C.N_JOBS)
                fit_s = time.time() - t0
                proba = m.predict_proba(ctx.X[te]); pred = proba.argmax(1)
                rows.append(_row(task, prot, "raw", arch, yt[te], pred, proba, ctx.clusters[te], fit_s))
                rows[-1]["n_params"] = deep.n_params(m.model); rows[-1]["epochs"] = m.epochs_
                np.savez_compressed(PRED / f"{task}_{prot}_raw_{arch}.npz", idx=te, y=yt[te], proba=proba)
                pcs.append(E.per_class(yt[te], pred, TASK_NAMES[task]).assign(
                    task=task, protocol=prot, features="raw", model=arch))
                print(task, prot, "raw", arch, round(rows[-1]["macro_f1"], 4), flush=True)
            pd.DataFrame(rows).to_csv(C.RES / "holdout_metrics.csv", index=False)
    pd.concat(pcs).to_csv(C.RES / "holdout_per_class.csv", index=False)
    pd.DataFrame(params).to_csv(C.RES / "holdout_best_params.csv", index=False)


def _row(task, prot, fs, name, y, pred, proba, clusters, fit_s):
    labels = np.arange(proba.shape[1])
    mm = E.metrics(y, pred, proba, labels)
    ci = E.bootstrap_ci(y, pred)
    cci = E.bootstrap_ci(y, pred, clusters=clusters)
    return {"task": task, "protocol": prot, "features": fs, "model": name, **mm,
            "acc_ci_lo": ci["acc_lo"], "acc_ci_hi": ci["acc_hi"], "f1_ci_lo": ci["f1_lo"], "f1_ci_hi": ci["f1_hi"],
            "acc_cci_lo": cci["acc_lo"], "acc_cci_hi": cci["acc_hi"], "f1_cci_lo": cci["f1_lo"],
            "f1_cci_hi": cci["f1_hi"], "fit_seconds": fit_s, "n_test": len(y)}


def _cv_fold(ctx, task, prot, k, a, b, name, F, yt, n_cls):
    if name in deep.DEEP:
        m = fit_deep(name, ctx.X, yt, a, prot, ctx, n_cls, threads=1)
        proba = m.predict_proba(ctx.X[b]); bp = {}
    else:
        inner = data.inner_cv(prot, ctx.y, ctx.pos, ctx.groups, a)
        est, bp = fit_classical(name, F, yt, a, inner, n_jobs=1)
        proba = est.predict_proba(F[b])
    mm = E.metrics(yt[b], proba.argmax(1), proba, np.arange(n_cls))
    return {"task": task, "protocol": prot, "model": name, "fold": k, "n_train": len(a),
            "n_test": len(b), "best_params": json.dumps(bp), **mm}


def stage_cv(ctx, tasks=("4class", "3class")):
    """Repeated (5x2) CV on the development set with nested tuning; used for statistical tests."""
    out = []
    for task in tasks:
        idx, yt = ctx.task(task)
        n_cls = len(TASK_NAMES[task])
        F = ctx.fsets["raw+eng"].values
        for prot in ["random", "blocked"]:
            dev, _ = data.holdout(prot, ctx.y, ctx.pos, ctx.groups, idx)
            sp = data.splits(prot, ctx.y, ctx.pos, ctx.groups, dev)
            names = [m for m in models.CLASSICAL if m != "Majority"] + ["MLP", "1D-CNN"]
            jobs = [(k, a, b, n) for n in names for k, (a, b) in enumerate(sp)]
            res = Parallel(n_jobs=C.N_JOBS, verbose=0)(
                delayed(_cv_fold)(ctx, task, prot, k, a, b, n, F, yt, n_cls) for k, a, b, n in jobs)
            out.extend(res)
            print("cv", task, prot, "done", flush=True)
            pd.DataFrame(out).to_csv(C.RES / "cv_fold_scores.csv", index=False)


def stage_stats():
    cv = pd.read_csv(C.RES / "cv_fold_scores.csv")
    rows, fr = [], []
    for (task, prot), d in cv.groupby(["task", "protocol"]):
        tab = d.pivot(index="fold", columns="model", values="macro_f1")
        best = tab.mean().idxmax()
        ntr, nte = d.n_train.mean(), d.n_test.mean()
        tp = []
        for m in tab.columns:
            if m == best:
                continue
            t, p = E.corrected_ttest(tab[best], tab[m], ntr, nte)
            tp.append((m, t, p))
        adj = E.holm([p for _, _, p in tp])
        for (m, t, p), pa in zip(tp, adj):
            rows.append({"task": task, "protocol": prot, "best": best, "other": m,
                         "best_mean_f1": tab[best].mean(), "other_mean_f1": tab[m].mean(),
                         "t": t, "p": p, "p_holm": pa})
        stat, p, ranks, nem, cd = E.friedman_nemenyi(tab)
        fr.append({"task": task, "protocol": prot, "friedman_chi2": stat, "friedman_p": p,
                   "critical_difference": cd, "n_folds": len(tab), "n_models": tab.shape[1],
                   **{f"rank_{k}": v for k, v in ranks.items()}})
        nem.to_csv(C.RES / f"nemenyi_{task}_{prot}.csv")
    pd.DataFrame(rows).to_csv(C.RES / "stats_corrected_ttest.csv", index=False)
    pd.DataFrame(fr).to_csv(C.RES / "stats_friedman.csv", index=False)
    summ = cv.groupby(["task", "protocol", "model"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        kappa_mean=("kappa", "mean"), auc_mean=("auc", "mean"), ece_mean=("ece", "mean")).reset_index()
    summ.to_csv(C.RES / "cv_summary.csv", index=False)


def stage_ablation(ctx):
    from lightgbm import LGBMClassifier
    eng = ctx.fsets["eng"]
    raw = ctx.fsets["raw"]
    grp = {g: [c for c in eng.columns if features.group_of(c) == g] for g in features.GROUPS}
    configs = {"all (raw+eng)": list(raw.columns) + list(eng.columns), "raw only": list(raw.columns),
               "engineered only": list(eng.columns)}
    for g, cols in grp.items():
        configs[f"only {g}"] = cols
        configs[f"eng minus {g}"] = [c for c in eng.columns if c not in cols]
    full = pd.concat([raw, eng], axis=1)
    out = []
    for task in ["4class", "3class"]:
        idx, yt = ctx.task(task)
        for prot in ["random", "blocked"]:
            dev, te = data.holdout(prot, ctx.y, ctx.pos, ctx.groups, idx)
            sp = data.splits(prot, ctx.y, ctx.pos, ctx.groups, dev, 5, 1)
            for cname, cols in configs.items():
                F = full[cols].values
                mk = lambda: LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8,
                                            subsample_freq=1, colsample_bytree=0.8, n_jobs=C.N_JOBS,
                                            random_state=C.SEED, verbose=-1)
                cvf = [E.metrics(yt[b], mk().fit(F[a], yt[a]).predict(F[b]))["macro_f1"] for a, b in sp]
                p = mk().fit(F[dev], yt[dev]).predict(F[te])
                out.append({"task": task, "protocol": prot, "config": cname, "n_features": len(cols),
                            "cv_macro_f1_mean": np.mean(cvf), "cv_macro_f1_sd": np.std(cvf, ddof=1),
                            "test_macro_f1": E.metrics(yt[te], p)["macro_f1"]})
                print("ablation", task, prot, cname, round(out[-1]["test_macro_f1"], 4), flush=True)
    pd.DataFrame(out).to_csv(C.RES / "ablation_feature_groups.csv", index=False)


def stage_shap(ctx):
    import shap
    from lightgbm import LGBMClassifier
    full = ctx.fsets["raw+eng"]
    out = {}
    for task in ["4class", "3class"]:
        idx, yt = ctx.task(task)
        for prot in ["random", "blocked"]:
            dev, te = data.holdout(prot, ctx.y, ctx.pos, ctx.groups, idx)
            m = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=C.N_JOBS, random_state=C.SEED, verbose=-1)
            m.fit(full.iloc[dev], yt[dev])
            expl = shap.TreeExplainer(m)
            sv = expl.shap_values(full.iloc[te])
            sv = np.stack(sv, -1) if isinstance(sv, list) else sv     # (n, f, classes)
            imp = np.abs(sv).mean(0)                                   # (f, classes)
            df = pd.DataFrame(imp, index=full.columns, columns=TASK_NAMES[task])
            df["mean_abs_shap"] = imp.mean(1)
            df["gain_importance"] = m.booster_.feature_importance("gain")
            df.sort_values("mean_abs_shap", ascending=False).to_csv(C.RES / f"shap_{task}_{prot}.csv")
            np.savez_compressed(PRED / f"shap_{task}_{prot}.npz", sv=sv, X=full.iloc[te].values,
                                cols=np.array(full.columns))
            out[(task, prot)] = df
    return out


def stage_windowed(ctx, win=64):
    """Classes 1-3 only: physically meaningful multichannel temporal features on 0.25 s windows."""
    from scipy.signal import welch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 70)}
    feats, labs, blocks = [], [], []
    for c in [1, 2, 3]:
        Xc = ctx.X[ctx.y == c]
        for s in range(0, len(Xc) - win + 1, win):
            w = Xc[s:s + win]
            f, P = welch(w, fs=C.FS_BEED, nperseg=win, axis=0)
            row = {}
            for ch in range(16):
                for bn, (lo, hi) in bands.items():
                    row[f"ch{ch+1}_{bn}"] = np.log(P[(f >= lo) & (f < hi), ch].sum() + 1e-6)
                row[f"ch{ch+1}_ll"] = np.log(np.abs(np.diff(w[:, ch])).sum() + 1e-6)
            feats.append(row); labs.append(c - 1); blocks.append(s // C.BLOCK)
    F = pd.DataFrame(feats); yy = np.array(labs); bb = np.array(blocks)
    # leave-one-block-out (blocks of 200 rows = 0.78 s; windows never straddle the held-out block
    # boundary by construction because window starts are multiples of 64 and blocks of 200 rows are
    # matched on window start -> we drop windows adjacent to the held-out block as a purge)
    out = []
    preds = {"LR": np.zeros(len(yy), int), "LightGBM": np.zeros(len(yy), int)}
    for blk in np.unique(bb):
        te = bb == blk
        trm = np.abs(bb - blk) > 1        # purge neighbouring blocks
        for name, mk in {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
                         "LightGBM": lambda: LGBMClassifier(n_estimators=200, num_leaves=7, min_child_samples=5,
                                                            n_jobs=C.N_JOBS, random_state=C.SEED, verbose=-1)}.items():
            preds[name][te] = mk().fit(F.values[trm], yy[trm]).predict(F.values[te])
    for name, p in preds.items():
        out.append({"model": name, "n_windows": len(yy), "window_s": win / C.FS_BEED, **E.metrics(yy, p)})
    pd.DataFrame(out).to_csv(C.RES / "windowed_3class_lobo.csv", index=False)
    # also report how many windows per class
    return pd.DataFrame(out)


def stage_edge(ctx):
    """Parameter counts and single-sample CPU latency for deployable candidates (blocked protocol)."""
    import torch
    from lightgbm import LGBMClassifier
    idx, yt = ctx.task("4class")
    dev, te = data.holdout("blocked", ctx.y, ctx.pos, ctx.groups, idx)
    rows = []
    x1 = ctx.X[te[:1]]
    torch.set_num_threads(1)
    for arch in deep.DEEP:
        m = fit_deep(arch, ctx.X, yt, dev, "blocked", ctx, 4, threads=1)
        m.threads = 1
        p = m.predict(ctx.X[te])
        rows.append({"model": arch, "input": "raw 16 values", "n_params": deep.n_params(m.model),
                     "size_kB_fp32": deep.n_params(m.model) * 4 / 1024,
                     "latency_ms": deep.latency_ms(m.predict_proba, x1),
                     "macro_f1_blocked_test": E.metrics(yt[te], p)["macro_f1"]})
    F = ctx.fsets["raw+eng"].values
    lg = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, n_jobs=1, random_state=C.SEED, verbose=-1).fit(F[dev], yt[dev])
    nodes = lg.booster_.trees_to_dataframe().shape[0]

    def lg_pipeline(x):
        f = pd.concat([features.raw_features(x), features.engineered(x)], axis=1).values
        return lg.predict_proba(f)
    rows.append({"model": "LightGBM", "input": "raw+engineered (77)", "n_params": nodes,
                 "size_kB_fp32": nodes * 4 * 3 / 1024,
                 "latency_ms": deep.latency_ms(lg_pipeline, x1, n=200),
                 "macro_f1_blocked_test": E.metrics(yt[te], lg.predict(F[te]))["macro_f1"]})
    pd.DataFrame(rows).to_csv(C.RES / "edge_deployment.csv", index=False)


def stage_holdout_extra(ctx):
    """3-class task on raw values (all classical models) and single-feature shortcut stumps."""
    rows, pcs = [], []
    task = "3class"
    idx, yt = ctx.task(task)
    for prot in ["random", "blocked"]:
        tr, te = data.holdout(prot, ctx.y, ctx.pos, ctx.groups, idx)
        inner = data.inner_cv(prot, ctx.y, ctx.pos, ctx.groups, tr)
        F = ctx.fsets["raw"].values
        for name in models.CLASSICAL:
            t0 = time.time()
            est, bp = fit_classical(name, F, yt, tr, inner)
            proba = est.predict_proba(F[te]); pred = proba.argmax(1)
            rows.append(_row(task, prot, "raw", name, yt[te], pred, proba, ctx.clusters[te], time.time() - t0))
            np.savez_compressed(PRED / f"{task}_{prot}_raw_{name}.npz", idx=te, y=yt[te], proba=proba)
            pcs.append(E.per_class(yt[te], pred, TASK_NAMES[task]).assign(task=task, protocol=prot, features="raw", model=name))
            print(task, prot, "raw", name, round(rows[-1]["macro_f1"], 4), flush=True)
    h = pd.read_csv(C.RES / "holdout_metrics.csv")
    h = h[~((h.task == task) & (h.features == "raw") & h.model.isin(models.CLASSICAL))]
    pd.concat([h, pd.DataFrame(rows)]).to_csv(C.RES / "holdout_metrics.csv", index=False)
    pc = pd.read_csv(C.RES / "holdout_per_class.csv")
    pc = pc[~((pc.task == task) & (pc.features == "raw") & pc.model.isin(models.CLASSICAL))]
    pd.concat([pc] + pcs).to_csv(C.RES / "holdout_per_class.csv", index=False)
    # shortcut: one engineered feature + depth-1 decision tree, binary healthy vs seizure
    from sklearn.tree import DecisionTreeClassifier
    out = []
    idx, yb = ctx.task("binary")
    for prot in ["random", "blocked"]:
        tr, te = data.holdout(prot, ctx.y, ctx.pos, ctx.groups, idx)
        for feat in ["t_lag1_ac", "t_std", "f_low_high_ratio", "t_zc_mean"]:
            f = ctx.fsets["eng"][[feat]].values
            st = DecisionTreeClassifier(max_depth=1, random_state=C.SEED).fit(f[tr], yb[tr])
            mm = E.metrics(yb[te], st.predict(f[te]))
            out.append({"protocol": prot, "feature": feat, "threshold": st.tree_.threshold[0], **mm})
    pd.DataFrame(out).to_csv(C.RES / "shortcut_stumps.csv", index=False)
