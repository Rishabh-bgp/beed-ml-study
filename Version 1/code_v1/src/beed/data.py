"""Data loading, provenance-derived group labels and leakage-aware splitters."""
from __future__ import annotations
import hashlib
import urllib.request
import numpy as np
import pandas as pd
from . import config as C

COLS = [f"X{i}" for i in range(1, 17)]


def load_beed(path=C.DATA):
    df = pd.read_csv(path)
    X = df[COLS].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    pos = np.zeros(len(y), dtype=int)          # position of each row within its class
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        pos[idx] = np.arange(len(idx))
    return df, X, y, pos


def ensure_bonn(path=C.BONN_CSV, url=C.BONN_URL, sha=C.BONN_SHA256):
    """Download the UCI 'Epileptic Seizure Recognition' (Bonn-derived) CSV and verify its hash."""
    if not path.exists():
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # pragma: no cover
            print("Could not download Bonn file:", e)
            return None
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != sha:
        print("WARNING: Bonn file hash mismatch", h)
    return path


def match_to_bonn(X, y, bonn_path=C.BONN_CSV):
    """Search every 16-sample window of every Bonn row for exact matches of BEED rows."""
    from numpy.lib.stride_tricks import sliding_window_view
    b = pd.read_csv(bonn_path)
    B = b[[f"X{i}" for i in range(1, 179)]].to_numpy(dtype=np.int64)
    W = sliding_window_view(B, 16, axis=1)
    lut = {}
    for r in range(W.shape[0]):
        for s in range(W.shape[1]):
            lut.setdefault(W[r, s].tobytes(), []).append((r, s))
    Xi = X.astype(np.int64)
    out = []
    for i in range(len(Xi)):
        hits = lut.get(Xi[i].tobytes())
        if hits:
            r, s = hits[0]
            name = str(b.iloc[r, 0])
            out.append(dict(index=i, beed_class=int(y[i]), n_hits=len(hits), bonn_row=r,
                            offset=s, bonn_label=int(b.y.iloc[r]), bonn_name=name,
                            bonn_segment=name.split(".", 1)[1] if "." in name else name))
    return pd.DataFrame(out), b


def make_groups(y, pos, bonn_match: pd.DataFrame | None = None, block=C.BLOCK):
    """Group labels for leakage-aware splitting.

    Classes 1-3 are continuous multichannel recordings (rows = time points): groups are contiguous
    blocks of `block` rows. Class-0 rows are independent 16-sample snippets drawn from 100 Bonn
    set-B segments: the Bonn segment tag is used when the provenance match is available."""
    groups = np.array([f"c{c}_b{p // block}" if c != 0 else f"c0_r{i}"
                       for i, (c, p) in enumerate(zip(y, pos))], dtype=object)
    if bonn_match is not None and len(bonn_match):
        m = bonn_match[bonn_match.beed_class == 0]
        groups[m["index"].to_numpy()] = ("c0_" + m.bonn_segment).to_numpy()
    return groups


def purge(train_idx, test_idx, y, pos, gap=C.GAP):
    """Drop training rows (classes 1-3) lying within `gap` rows of a test row of the same class."""
    if gap <= 0:
        return train_idx
    keep = np.ones(len(train_idx), dtype=bool)
    for c in np.unique(y):
        if c == 0:
            continue
        tp = np.sort(pos[test_idx][y[test_idx] == c])
        if len(tp) == 0:
            continue
        m = y[train_idx] == c
        p = pos[train_idx][m]
        j = np.searchsorted(tp, p)
        d1 = np.abs(p - tp[np.clip(j, 0, len(tp) - 1)])
        d0 = np.abs(p - tp[np.clip(j - 1, 0, len(tp) - 1)])
        keep[np.where(m)[0][np.minimum(d0, d1) <= gap]] = False
    return train_idx[keep]


def random_holdout(y, idx=None, test_size=0.2, seed=C.SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y)) if idx is None else idx
    tr, te = train_test_split(idx, test_size=test_size, stratify=y[idx], random_state=seed)
    return np.sort(tr), np.sort(te)


def blocked_holdout(y, pos, groups, idx=None, test_frac=0.2, seed=C.SEED, gap=C.GAP):
    """Leakage-safe hold-out: final 20% of each continuous recording (classes 1-3) and 20% of the
    Bonn segments (class 0) form the test set; a purge gap separates training and test in time."""
    idx = np.arange(len(y)) if idx is None else idx
    rng = np.random.RandomState(seed)
    test = np.zeros(len(y), dtype=bool)
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        if c == 0:
            g = np.unique(groups[ci])
            tg = rng.choice(g, size=int(round(test_frac * len(g))), replace=False)
            test[ci[np.isin(groups[ci], tg)]] = True
        else:
            test[ci[pos[ci] >= int(len(ci) * (1 - test_frac))]] = True
    te = np.where(test)[0]
    tr = purge(np.setdiff1d(idx, te), te, y, pos, gap)
    return np.sort(tr), np.sort(te)


def random_cv(y, idx, n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS, seed=C.SEED):
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in rskf.split(idx, y[idx]):
        yield idx[tr], idx[te]


def blocked_cv(y, pos, groups, idx, n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS, seed=C.SEED,
               gap=C.GAP):
    """Repeated class-stratified group k-fold over contiguous blocks / Bonn segments, then purge."""
    for r in range(n_repeats):
        rng = np.random.RandomState(seed + r)
        fold_of = {}
        for c in np.unique(y[idx]):
            g = np.unique(groups[idx][y[idx] == c])
            g = g[rng.permutation(len(g))]
            for k, gg in enumerate(g):
                fold_of[gg] = k % n_splits
        f = np.array([fold_of[g] for g in groups[idx]])
        for k in range(n_splits):
            te = idx[f == k]
            yield purge(idx[f != k], te, y, pos, gap), te


def splits(protocol, y, pos, groups, idx, n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS, seed=C.SEED):
    if protocol == "random":
        return list(random_cv(y, idx, n_splits, n_repeats, seed))
    return list(blocked_cv(y, pos, groups, idx, n_splits, n_repeats, seed))


def holdout(protocol, y, pos, groups, idx=None):
    return random_holdout(y, idx) if protocol == "random" else blocked_holdout(y, pos, groups, idx)


def inner_cv(protocol, y, pos, groups, idx, n_splits=3, seed=C.SEED):
    """Inner CV (positions relative to idx) for hyper-parameter tuning, respecting the protocol."""
    gen = splits(protocol, y, pos, groups, idx, n_splits, 1, seed)
    loc = np.full(len(y), -1)
    loc[idx] = np.arange(len(idx))
    return [(loc[tr], loc[te]) for tr, te in gen]
