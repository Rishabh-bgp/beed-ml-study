"""Metrics, bootstrap confidence intervals, calibration and statistical tests."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix, log_loss,
                             balanced_accuracy_score)


def metrics(y, pred, proba=None, labels=None):
    labels = np.unique(y) if labels is None else labels
    out = {"accuracy": accuracy_score(y, pred), "balanced_acc": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=labels),
           "kappa": cohen_kappa_score(y, pred)}
    if proba is not None:
        try:
            if len(labels) == 2:
                out["auc"] = roc_auc_score(y, proba[:, 1])
            else:
                out["auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            out["auc"] = np.nan
        out["brier"] = brier(y, proba, labels)
        out["ece"] = ece(y, proba)
        out["logloss"] = log_loss(y, np.clip(proba, 1e-7, 1), labels=labels)
    return out


def per_class(y, pred, names):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=np.arange(len(names)), zero_division=0)
    return pd.DataFrame({"class": names, "precision": p, "recall": r, "f1": f, "support": s})


def brier(y, proba, labels):
    Y = (np.asarray(y)[:, None] == np.asarray(labels)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - Y) ** 2, 1)))


def ece(y, proba, n_bins=15):
    conf, pred = proba.max(1), proba.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def bootstrap_ci(y, pred, clusters=None, n_boot=2000, seed=42, alpha=0.05):
    """Percentile CI for accuracy and macro-F1. If `clusters` is given, a stratified cluster
    bootstrap is used (whole clusters resampled within class) to respect autocorrelation."""
    rng = np.random.RandomState(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    labels = np.unique(y)
    if clusters is None:
        clusters = np.arange(len(y))
    units = {}
    for c in labels:
        cl = np.unique(clusters[y == c])
        units[c] = [np.where((clusters == u) & (y == c))[0] for u in cl]
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = np.concatenate([np.concatenate([units[c][k] for k in rng.randint(0, len(units[c]), len(units[c]))])
                              for c in labels])
        accs.append(accuracy_score(y[idx], pred[idx]))
        f1s.append(f1_score(y[idx], pred[idx], average="macro", labels=labels))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"acc_lo": np.percentile(accs, q[0]), "acc_hi": np.percentile(accs, q[1]),
            "f1_lo": np.percentile(f1s, q[0]), "f1_hi": np.percentile(f1s, q[1])}


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for J paired CV scores."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(p)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def friedman_nemenyi(score_table: pd.DataFrame):
    """score_table: rows = folds, columns = models (higher is better)."""
    import scikit_posthocs as sp
    stat, p = stats.friedmanchisquare(*[score_table[c] for c in score_table.columns])
    ranks = score_table.rank(axis=1, ascending=False).mean()
    nem = sp.posthoc_nemenyi_friedman(score_table.to_numpy())
    nem.index = nem.columns = score_table.columns
    k, n = score_table.shape[1], score_table.shape[0]
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
         10: 3.164, 11: 3.219, 12: 3.268}.get(k, np.nan)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return stat, p, ranks.sort_values(), nem, cd
