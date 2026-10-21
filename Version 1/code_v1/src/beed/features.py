"""Row-wise feature extraction for 16-value BEED vectors.

Each BEED row is treated as a short 16-point sequence. For class 0 the sequence is temporal
(16 consecutive samples); for classes 1-3 it is the 16-channel spatial profile at one time point
(see provenance analysis), so the same descriptors are temporal for one class and spatial for
the others. Groups: time-domain, frequency-domain (normalised FFT bins; physical band powers are
not resolvable with 16 points), nonlinear and discrete-wavelet features."""
import numpy as np
import pandas as pd
import pywt
import antropy as ant
from scipy import stats

EPS = 1e-12


def _hjorth(x):
    d1 = np.diff(x, axis=1); d2 = np.diff(d1, axis=1)
    v0, v1, v2 = x.var(1) + EPS, d1.var(1) + EPS, d2.var(1) + EPS
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return v0, mob, comp


def time_features(X):
    xc = X - X.mean(1, keepdims=True)
    d1 = np.diff(X, axis=1); d2 = np.diff(d1, axis=1)
    act, mob, comp = _hjorth(X)
    sgn = np.sign(xc); sgn[sgn == 0] = 1
    raw_sgn = np.sign(X); raw_sgn[raw_sgn == 0] = 1
    ext = np.sum(np.diff(np.sign(d1), axis=1) != 0, axis=1)
    q75, q25 = np.percentile(X, [75, 25], axis=1)
    f = {
        "t_mean": X.mean(1), "t_std": X.std(1), "t_min": X.min(1), "t_max": X.max(1),
        "t_range": np.ptp(X, axis=1), "t_median": np.median(X, 1), "t_iqr": q75 - q25,
        "t_skew": np.nan_to_num(stats.skew(X, axis=1)),
        "t_kurt": np.nan_to_num(stats.kurtosis(X, axis=1)),
        "t_rms": np.sqrt((X ** 2).mean(1)), "t_meanabs": np.abs(X).mean(1),
        "t_logenergy": np.log((X ** 2).sum(1) + 1.0),
        "t_hjorth_act": np.log(act), "t_hjorth_mob": mob, "t_hjorth_comp": comp,
        "t_linelength": np.abs(d1).sum(1), "t_mad1": np.abs(d1).mean(1),
        "t_mad2": np.abs(d2).mean(1), "t_maxabsdiff": np.abs(d1).max(1),
        "t_zc_mean": np.sum(np.diff(sgn, axis=1) != 0, axis=1),
        "t_zc_raw": np.sum(np.diff(raw_sgn, axis=1) != 0, axis=1),
        "t_n_extrema": ext,
        "t_lag1_ac": np.nan_to_num(np.sum(xc[:, :-1] * xc[:, 1:], 1) / (np.sum(xc ** 2, 1) + EPS)),
        "t_norm_ll": np.abs(d1).sum(1) / (X.std(1) + 1.0),
    }
    return pd.DataFrame(f)


def freq_features(X):
    xc = X - X.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(xc * np.hanning(X.shape[1]), axis=1)) ** 2
    P = P[:, 1:]                                   # drop DC -> 8 bins
    tot = P.sum(1, keepdims=True) + EPS
    Pn = P / tot
    k = np.arange(1, P.shape[1] + 1)
    cum = np.cumsum(Pn, 1)
    f = {f"f_bin{j}": Pn[:, j - 1] for j in k}
    f.update({
        "f_logpower": np.log(tot[:, 0] + 1.0),
        "f_spec_entropy": -(Pn * np.log(Pn + EPS)).sum(1) / np.log(P.shape[1]),
        "f_centroid": (Pn * k).sum(1),
        "f_edge80": (cum < 0.8).sum(1) + 1,
        "f_peak_bin": P.argmax(1) + 1,
        "f_low_high_ratio": np.log((Pn[:, :2].sum(1) + EPS) / (Pn[:, 4:].sum(1) + EPS)),
    })
    return pd.DataFrame(f)


def nonlinear_features(X):
    rows = []
    for x in np.ascontiguousarray(X, dtype=float):
        x = np.ascontiguousarray(x)
        sd = x.std()
        se = ant.sample_entropy(x) if sd > 0 else 0.0
        if not np.isfinite(se):
            se = np.log(len(x))                     # undefined -> maximal irregularity cap
        rows.append({
            "n_perm_entropy": ant.perm_entropy(x, order=3, delay=1, normalize=True),
            "n_sample_entropy": se,
            "n_app_entropy": ant.app_entropy(x, order=2) if sd > 0 else 0.0,
            "n_svd_entropy": ant.svd_entropy(x, order=3, delay=1, normalize=True) if sd > 0 else 0.0,
            "n_petrosian_fd": ant.petrosian_fd(x),
            "n_katz_fd": ant.katz_fd(x) if sd > 0 else 1.0,
            "n_higuchi_fd": ant.higuchi_fd(x.astype(float), kmax=4) if sd > 0 else 1.0,
        })
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def wavelet_features(X, wavelet="db2", level=2):
    coeffs = pywt.wavedec(X, wavelet, level=level, mode="periodization", axis=1)
    names = [f"A{level}"] + [f"D{l}" for l in range(level, 0, -1)]
    en = np.stack([(c ** 2).sum(1) for c in coeffs], 1)
    tot = en.sum(1, keepdims=True) + EPS
    f = {}
    for j, (nm, c) in enumerate(zip(names, coeffs)):
        f[f"w_{nm}_logenergy"] = np.log(en[:, j] + 1.0)
        f[f"w_{nm}_relenergy"] = en[:, j] / tot[:, 0]
        f[f"w_{nm}_std"] = c.std(1)
        f[f"w_{nm}_meanabs"] = np.abs(c).mean(1)
    return pd.DataFrame(f)


def raw_features(X):
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(1, 17)])


GROUPS = {"time": time_features, "freq": freq_features, "nonlinear": nonlinear_features,
          "wavelet": wavelet_features}


def engineered(X):
    return pd.concat([fn(X) for fn in GROUPS.values()], axis=1)


def group_of(col):
    return {"t": "time", "f": "freq", "n": "nonlinear", "w": "wavelet", "X": "raw"}[col[0]]


def build_feature_sets(X):
    raw = raw_features(X)
    eng = engineered(X)
    return {"raw": raw, "eng": eng, "raw+eng": pd.concat([raw, eng], axis=1)}
