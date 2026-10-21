"""All figures (300 dpi PNG, colour-blind-safe palette, legible at publication size)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
from . import config as C

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
PAL = sns.color_palette("colorblind")
CLS_COL = {0: PAL[0], 1: PAL[1], 2: PAL[2], 3: PAL[3]}
MARK = {0: "o", 1: "s", 2: "^", 3: "D"}
DPI = 300


def save(fig, name):
    fig.savefig(C.FIG / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_examples(ctx):
    X, y = ctx.X, ctx.y
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for c in range(4):
        i = np.where(y == c)[0][3]
        ax.plot(range(1, 17), X[i], marker=MARK[c], color=CLS_COL[c], label=C.CLASS_NAMES[c])
    ax.set_xlabel("Column index (X1–X16)"); ax.set_ylabel("Value (dataset units)")
    ax.set_title("(a) One row per class"); ax.legend(fontsize=9)
    ax = axes[1]
    Xc = X[y == 2][:512]
    t = np.arange(len(Xc)) / C.FS_BEED
    for ch in range(16):
        ax.plot(t, Xc[:, ch] / 40 - ch, lw=0.6, color="k")
    ax.set_yticks(-np.arange(16)); ax.set_yticklabels([f"X{i}" for i in range(1, 17)], fontsize=7)
    ax.set_xlabel("Time (s), assuming 256 Hz along rows"); ax.set_title("(b) Class 2: first 512 rows as 16 traces")
    save(fig, "fig_examples")


def fig_structure(ctx):
    X, y = ctx.X, ctx.y
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for c in range(4):
        ax = axes[0, c]
        sns.heatmap(np.corrcoef(X[y == c].T), vmin=-1, vmax=1, cmap="vlag", ax=ax, square=True,
                    cbar=c == 3, xticklabels=4, yticklabels=4)
        ax.set_title(C.CLASS_NAMES[c]); ax.set_xlabel("column"); ax.set_ylabel("column")
    lags = np.arange(0, 16)
    ax = axes[1, 0]; ax2 = axes[1, 1]
    for c in range(4):
        Cc = np.corrcoef(X[y == c].T)
        ax.plot(lags, [np.mean(np.diag(Cc, k)) for k in lags], marker=MARK[c], color=CLS_COL[c], label=C.SHORT_NAMES[c])
        Xc = X[y == c]
        rl = [1.0] + [np.mean([np.corrcoef(Xc[:-k, j], Xc[k:, j])[0, 1] for j in range(16)]) for k in range(1, 61)]
        ax2.plot(range(61), rl, color=CLS_COL[c], marker=MARK[c], markevery=10, label=C.SHORT_NAMES[c])
    ax.set_xlabel("Lag along columns"); ax.set_ylabel("Mean correlation"); ax.set_title("(i) Within-row (column) lag"); ax.legend()
    ax2.set_xlabel("Lag along rows"); ax2.set_ylabel("Mean correlation"); ax2.set_title("(ii) Across-row lag"); ax2.legend()
    ax = axes[1, 2]
    for c in range(4):
        Cc = np.corrcoef(X[y == c].T)
        ax.plot(range(1, 16), np.diag(Cc, 1), marker=MARK[c], color=CLS_COL[c], label=C.SHORT_NAMES[c])
    ax.set_xlabel("Column pair (k, k+1)"); ax.set_ylabel("Correlation"); ax.set_title("(iii) Lag-1 by position"); ax.legend(fontsize=8)
    ax = axes[1, 3]
    sd = X.std(1)
    sns.boxplot(x=[C.SHORT_NAMES[v] for v in y], y=np.log10(sd + 1), ax=ax, color="0.8",
                order=C.SHORT_NAMES)
    ax.set_ylabel("log10(1 + row SD)"); ax.set_title("(iv) Row amplitude")
    fig.tight_layout()
    save(fig, "fig_structure")


def fig_provenance(ctx):
    bm = ctx.bonn_match
    if not len(bm):
        return
    b = pd.read_csv(C.BONN_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    r = bm.iloc[0]
    seg = b.iloc[int(r.bonn_row)][[f"X{i}" for i in range(1, 179)]].to_numpy(float)
    t = np.arange(178) / C.FS_BONN
    ax = axes[0]
    ax.plot(t, seg, color="0.5", lw=1, label=f"Bonn set B chunk ({r.bonn_name})")
    s = int(r.offset)
    ax.plot(t[s:s + 16], ctx.X[int(r["index"])], color=PAL[0], lw=2.5, marker="o", ms=3, label="BEED class-0 row 1")
    ax.axvspan(t[s], t[s + 15], color=PAL[0], alpha=0.12)
    ax.set_xlabel("Time (s) at 173.61 Hz"); ax.set_ylabel("Amplitude (dataset units)")
    ax.set_title("(a) Exact match of a BEED 'healthy' row inside a Bonn segment"); ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    cnt = bm.groupby("beed_class").size().reindex(range(4), fill_value=0)
    ax.bar([C.SHORT_NAMES[i] for i in range(4)], cnt.values, color=[CLS_COL[i] for i in range(4)], edgecolor="k")
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 30, str(v), ha="center")
    ax.set_ylim(0, 2300); ax.set_ylabel("Rows with exact Bonn match"); ax.set_title("(b) Matches per class")
    fig.tight_layout()
    save(fig, "fig_provenance")


def fig_psd(ctx):
    X, y = ctx.X, ctx.y
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for c in [1, 2, 3]:
        f, P = welch(X[y == c], fs=C.FS_BEED, nperseg=256, axis=0)
        m = P.mean(1)
        ax.semilogy(f, m, color=CLS_COL[c], marker=MARK[c], markevery=8, label=C.CLASS_NAMES[c])
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (units$^2$/Hz)"); ax.set_xlim(0, 100)
    ax.set_title("(a) Classes 1–3: Welch PSD along rows (256 Hz)"); ax.legend()
    ax = axes[1]
    for c in range(4):
        xc = X[y == c] - X[y == c].mean(1, keepdims=True)
        P = (np.abs(np.fft.rfft(xc * np.hanning(16), axis=1)) ** 2).mean(0)[1:]
        ax.semilogy(range(1, 9), P / P.sum(), marker=MARK[c], color=CLS_COL[c], label=C.SHORT_NAMES[c])
    ax.set_xlabel("FFT bin of the 16-value row (bin k = k/16 cycles per column)")
    ax.set_ylabel("Relative power"); ax.set_title("(b) Within-row spectrum (all classes)"); ax.legend()
    fig.tight_layout()
    save(fig, "fig_psd")


def fig_block_curve():
    d = pd.read_csv(C.RES / "leakage_block_size_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, (m, g) in enumerate(d.groupby("model")):
        for gap, ls in [(0, "--"), (C.GAP, "-")]:
            gg = g[(g.gap == gap) | (g.block_size == 1)]
            if gap:
                gg = gg[gg.block_size >= 100]   # purge would empty small-block folds
            s = gg.groupby("block_size").macro_f1.agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="os"[j], ls=ls, capsize=3, color=PAL[j],
                        label=f"{m}, gap {gap if gap else 0} rows")
    ax.set_xscale("log"); ax.set_xlabel("Contiguous block size (rows) assigned to folds; 1 = random split")
    ax.set_ylabel("Macro-F1 (5-fold CV)"); ax.set_title("Performance versus temporal separation of folds")
    ax.legend(fontsize=8)
    save(fig, "fig_block_curve")


def fig_holdout_bars():
    d = pd.read_csv(C.RES / "holdout_metrics.csv")
    d = d[(d.task == "4class") & (((d.features == "raw+eng") & ~d.model.isin(["MLP", "1D-CNN", "BiLSTM", "Transformer"]))
                                  | d.model.isin(["MLP", "1D-CNN", "BiLSTM", "Transformer"]))]
    order = d[d.protocol == "random"].sort_values("macro_f1").model.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    yy = np.arange(len(order))
    for j, (prot, hatch) in enumerate([("random", ""), ("blocked", "//")]):
        g = d[d.protocol == prot].set_index("model").loc[order]
        ax.barh(yy + (j - 0.5) * 0.38, g.macro_f1, height=0.38, color=PAL[j], hatch=hatch, edgecolor="k",
                label=f"{prot} split", xerr=[g.macro_f1 - g.f1_cci_lo, g.f1_cci_hi - g.macro_f1], capsize=2)
    ax.set_yticks(yy); ax.set_yticklabels(order); ax.set_xlabel("Hold-out macro-F1 (95% cluster-bootstrap CI)")
    ax.set_xlim(0, 1.0); ax.legend(loc="lower right"); ax.set_title("Four-class hold-out performance by protocol")
    save(fig, "fig_holdout_bars")


def _best(d, task, prot, feats=None):
    g = d[(d.task == task) & (d.protocol == prot) & (d.model != "Majority")]
    if feats:
        g = g[g.features.isin(feats)]
    return g.sort_values("macro_f1", ascending=False).iloc[0]


def fig_confusion():
    d = pd.read_csv(C.RES / "holdout_metrics.csv")
    from .experiments import TASK_NAMES
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    k = 0
    for task in ["4class", "3class"]:
        for prot in ["random", "blocked"]:
            b = _best(d, task, prot)
            z = np.load(C.RES / "preds" / f"{task}_{prot}_{b.features}_{b.model}.npz")
            cm = confusion_matrix(z["y"], z["proba"].argmax(1), labels=range(len(TASK_NAMES[task])))
            cmn = cm / cm.sum(1, keepdims=True)
            ax = axes[k]; k += 1
            annot = np.array([[f"{cm[i, j]}\n({cmn[i, j]:.2f})" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
            sns.heatmap(cmn, annot=annot, fmt="", cmap="Greys", vmin=0, vmax=1, cbar=False, ax=ax,
                        xticklabels=TASK_NAMES[task], yticklabels=TASK_NAMES[task], annot_kws={"size": 9})
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{task}, {prot}: {b.model}\n({b.features})", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_confusion")


def fig_roc_pr():
    d = pd.read_csv(C.RES / "holdout_metrics.csv")
    from .experiments import TASK_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for j, prot in enumerate(["random", "blocked"]):
        b = _best(d, "4class", prot)
        z = np.load(C.RES / "preds" / f"4class_{prot}_{b.features}_{b.model}.npz")
        yv, P = z["y"], z["proba"]
        for c in range(4):
            fpr, tpr, _ = roc_curve(yv == c, P[:, c])
            axes[0, j].plot(fpr, tpr, color=CLS_COL[c], label=f"{C.SHORT_NAMES[c]} (AUC {auc(fpr, tpr):.3f})")
            pr, rc, _ = precision_recall_curve(yv == c, P[:, c])
            axes[1, j].plot(rc, pr, color=CLS_COL[c], label=f"{C.SHORT_NAMES[c]} (AP {average_precision_score(yv == c, P[:, c]):.3f})")
        axes[0, j].plot([0, 1], [0, 1], "k:", lw=0.8)
        axes[0, j].set_xlabel("False-positive rate"); axes[0, j].set_ylabel("True-positive rate")
        axes[0, j].set_title(f"ROC, {prot} split: {b.model}"); axes[0, j].legend(fontsize=9, loc="lower right")
        axes[1, j].set_xlabel("Recall"); axes[1, j].set_ylabel("Precision")
        axes[1, j].set_title(f"Precision–recall, {prot} split"); axes[1, j].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_roc_pr")


def fig_calibration():
    d = pd.read_csv(C.RES / "holdout_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sel = [("LightGBM", "raw+eng"), ("SVM-RBF", "raw+eng"), ("LR", "raw+eng"), ("1D-CNN", "raw"), ("kNN", "raw+eng")]
    for j, prot in enumerate(["random", "blocked"]):
        ax = axes[j]
        for k, (m, fs) in enumerate(sel):
            f = C.RES / "preds" / f"4class_{prot}_{fs}_{m}.npz"
            if not f.exists():
                continue
            z = np.load(f); P = z["proba"]; yv = z["y"]
            conf, acc = P.max(1), (P.argmax(1) == yv)
            bins = np.linspace(0, 1, 11); mids, accs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mm = (conf > lo) & (conf <= hi)
                if mm.sum() >= 10:
                    mids.append(conf[mm].mean()); accs.append(acc[mm].mean())
            e = d[(d.task == "4class") & (d.protocol == prot) & (d.model == m) & (d.features == fs)].ece.values
            ax.plot(mids, accs, marker="osD^v"[k], color=PAL[k], label=f"{m} (ECE {e[0]:.3f})" if len(e) else m)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
        ax.set_title(f"Reliability, {prot} split"); ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_calibration")


def fig_ranks():
    f = pd.read_csv(C.RES / "stats_friedman.csv")
    fig, axes = plt.subplots(1, len(f), figsize=(4.2 * len(f), 4.0), squeeze=False)
    for k, (_, r) in enumerate(f.iterrows()):
        ranks = {c[5:]: r[c] for c in f.columns if c.startswith("rank_") and pd.notna(r[c])}
        s = pd.Series(ranks).sort_values()
        ax = axes[0, k]
        ax.errorbar(s.values, range(len(s)), xerr=r.critical_difference / 2, fmt="o", color="k", capsize=3)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index)
        ax.invert_yaxis(); ax.set_xlabel("Mean rank (1 = best)")
        ax.set_title(f"{r.task}, {r.protocol}\nFriedman p = {r.friedman_p:.1e}", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_ranks")


def fig_shap():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for k, (task, prot) in enumerate([("4class", "random"), ("4class", "blocked"), ("3class", "blocked")]):
        d = pd.read_csv(C.RES / f"shap_{task}_{prot}.csv", index_col=0).head(12)
        cls = [c for c in d.columns if c not in ("mean_abs_shap", "gain_importance")]
        left = np.zeros(len(d))
        for j, c in enumerate(cls):
            axes[k].barh(range(len(d)), d[c].values, left=left, color=PAL[j + (1 if task == "3class" else 0)],
                         hatch=["", "//", "..", "xx"][j], edgecolor="k", lw=0.4, label=c)
            left += d[c].values
        axes[k].set_yticks(range(len(d))); axes[k].set_yticklabels(d.index, fontsize=9); axes[k].invert_yaxis()
        axes[k].set_xlabel("Mean |SHAP| (summed over classes)"); axes[k].set_title(f"LightGBM, {task}, {prot} split", fontsize=11)
        axes[k].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_shap")


def fig_shap_beeswarm():
    import shap
    z = np.load(C.RES / "preds" / "shap_3class_blocked.npz", allow_pickle=True)
    sv, Xv, cols = z["sv"], z["X"], list(z["cols"])
    fig = plt.figure(figsize=(7, 6))
    shap.summary_plot(sv[:, :, 0], Xv, feature_names=cols, max_display=12, show=False, plot_size=None)
    plt.title("SHAP values for 'generalised' (3-class, blocked)")
    save(plt.gcf(), "fig_shap_beeswarm_GS")


def fig_ablation():
    d = pd.read_csv(C.RES / "ablation_feature_groups.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    order = d.config.unique().tolist()
    for k, task in enumerate(["4class", "3class"]):
        ax = axes[k]
        for j, (prot, hatch) in enumerate([("random", ""), ("blocked", "//")]):
            g = d[(d.task == task) & (d.protocol == prot)].set_index("config").loc[order]
            ax.barh(np.arange(len(order)) + (j - 0.5) * 0.38, g.cv_macro_f1_mean, xerr=g.cv_macro_f1_sd, height=0.38,
                    color=PAL[j], hatch=hatch, edgecolor="k", capsize=2, label=f"{prot} split")
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
        ax.set_xlabel("Macro-F1 (5-fold CV on development set, mean ± SD)"); ax.set_title(f"Feature-group ablation, {task}")
        ax.set_xlim(0, 1); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_embedding(ctx):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(C.SEED)
    idx = np.concatenate([rng.choice(np.where(ctx.y == c)[0], 600, replace=False) for c in range(4)])
    Z = TSNE(n_components=2, random_state=C.SEED, perplexity=40, init="pca").fit_transform(
        StandardScaler().fit_transform(ctx.fsets["eng"].values[idx]))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in range(4):
        m = ctx.y[idx] == c
        ax.scatter(Z[m, 0], Z[m, 1], s=6, marker=MARK[c], color=CLS_COL[c], label=C.CLASS_NAMES[c], alpha=0.7)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(markerscale=2.5, fontsize=9)
    ax.set_title("t-SNE of engineered features (600 rows per class)")
    save(fig, "fig_tsne")


def fig_pipeline(ctx=None):
    fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis("off")
    boxes = ["BEED CSV\n8000 rows × 16 columns\n4 classes", "Structure & provenance\naudit (autocorrelation,\nexact Bonn matching)",
             "Provenance groups\nBonn segment (class 0)\n200-row blocks\n(classes 1–3)",
             "Two protocols\nrandom rows vs\nblocked groups\n+ 50-row purge", "57 features + raw values\n7 classical + 4 deep models\nnested tuning",
             "Metrics, bootstrap CIs\nstatistical tests,\ncalibration, SHAP,\nablation, latency"]
    pos = [(0.02, 0.56), (0.35, 0.56), (0.68, 0.56), (0.68, 0.06), (0.35, 0.06), (0.02, 0.06)]
    for (x, y0), t in zip(pos, boxes):
        ax.add_patch(plt.Rectangle((x, y0), 0.30, 0.36, fill=False, lw=1.8, edgecolor="k"))
        ax.text(x + 0.15, y0 + 0.18, t, ha="center", va="center", fontsize=10.5)
    arrows = [((0.32, 0.74), (0.35, 0.74)), ((0.65, 0.74), (0.68, 0.74)), ((0.83, 0.56), (0.83, 0.42)),
              ((0.68, 0.24), (0.65, 0.24)), ((0.35, 0.24), (0.32, 0.24))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_pipeline")


def make_all(ctx):
    for fn in [fig_pipeline, fig_examples, fig_structure, fig_provenance, fig_psd, fig_embedding]:
        try:
            fn(ctx); print("figure", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, e)
    for fn in [fig_block_curve, fig_holdout_bars, fig_confusion, fig_roc_pr, fig_calibration, fig_ranks,
               fig_shap, fig_shap_beeswarm, fig_ablation]:
        try:
            fn(); print("figure", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, e)
