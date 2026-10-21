"""Small deep models for the raw 16-value vectors (CPU friendly)."""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from . import config as C


class MLP(nn.Module):
    def __init__(self, n_in=16, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Compact 1D CNN over the 16-point axis (time for class 0, electrodes for classes 1-3)."""
    def __init__(self, n_cls=4, w=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, w, 3, padding=1), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, 2 * w, 3, padding=1), nn.BatchNorm1d(2 * w), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm1d(2 * w), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4 * w, n_cls))

    def forward(self, x):
        h = self.f(x.unsqueeze(1))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class BiLSTM(nn.Module):
    def __init__(self, n_cls=4, h=48):
        super().__init__()
        self.lstm = nn.LSTM(1, h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4 * h, n_cls))

    def forward(self, x):
        o, _ = self.lstm(x.unsqueeze(-1))
        return self.head(torch.cat([o.mean(1), o[:, -1]], 1))


class TinyTransformer(nn.Module):
    def __init__(self, n_cls=4, d=32, L=16):
        super().__init__()
        self.emb = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        h = self.enc(self.emb(x.unsqueeze(-1)) + self.pos)
        return self.head(h.mean(1))


DEEP = {"MLP": MLP, "1D-CNN": CNN1D, "BiLSTM": BiLSTM, "Transformer": TinyTransformer}


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DeepClassifier:
    """sklearn-like wrapper: per-feature standardisation, early stopping on a validation split."""
    def __init__(self, arch, n_cls=4, max_epochs=60, patience=8, lr=2e-3, bs=128, seed=C.SEED,
                 threads=2):
        self.arch, self.n_cls, self.max_epochs, self.patience = arch, n_cls, max_epochs, patience
        self.lr, self.bs, self.seed, self.threads = lr, bs, seed, threads

    def fit(self, X, y, val_idx=None):
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        rng = np.random.RandomState(self.seed)
        if val_idx is None:
            val_idx = rng.choice(len(y), int(0.15 * len(y)), replace=False)
        tr = np.setdiff1d(np.arange(len(y)), val_idx)
        Xt, yt = torch.tensor(Xs[tr]), torch.tensor(y[tr])
        Xv, yv = torch.tensor(Xs[val_idx]), y[val_idx]
        self.model = DEEP[self.arch](n_cls=self.n_cls)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        lossf = nn.CrossEntropyLoss()
        best, best_state, bad, self.history = -1, None, 0, []
        for ep in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(len(yt))
            for i in range(0, len(yt), self.bs):
                b = perm[i:i + self.bs]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(self.model(Xt[b]), yt[b])
                loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                pv = self.model(Xv).argmax(1).numpy()
            f1 = f1_score(yv, pv, average="macro")
            sched.step(-f1)
            self.history.append(f1)
            if f1 > best + 1e-4:
                best, bad = f1, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.epochs_ = ep + 1
        self.classes_ = np.arange(self.n_cls)
        return self

    def predict_proba(self, X):
        torch.set_num_threads(self.threads)
        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.tensor(((X - self.mu) / self.sd).astype(np.float32)))
        return torch.softmax(z, 1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def latency_ms(predict_fn, x_one, n=500):
    """Median single-sample CPU inference latency (ms)."""
    for _ in range(20):
        predict_fn(x_one)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); predict_fn(x_one); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))
