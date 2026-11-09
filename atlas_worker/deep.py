"""PyTorch training for sensor sequences and table-based autoencoders."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import DataError, coerce, split_indices
from .forecast import forecast_charts, series_from
from .metrics import forecast_metrics, from_labels
from .protocol import emit, log, stage
from .tabular import anomaly_labels, anomaly_report, make_preprocessor

MIN_BATCH = 2


# ---- networks. Sequence input is (batch, time, features). ------------------------
class RNNNet(nn.Module):
    def __init__(self, cell, n_in, hidden, layers, out, bidirectional=False):
        super().__init__()
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), out)

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1])


class CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(c_in, c_out, 3, dilation=dilation)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(F.pad(x, (self.pad, 0)))) + self.skip(x)


class TCN(nn.Module):
    """Also serves as WaveNet-style dilated causal stack through `depth`."""

    def __init__(self, n_in, hidden, out, depth=4):
        super().__init__()
        self.net = nn.Sequential(*[CausalBlock(n_in if i == 0 else hidden, hidden, 2 ** i) for i in range(depth)])
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))[:, :, -1])


class CNN1D(nn.Module):
    def __init__(self, n_in, hidden, out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_in, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).mean(2))


class DLinear(nn.Module):
    """Trend/remainder decomposition with one linear map each."""

    def __init__(self, n_in, seq_len, out, kernel=25):
        super().__init__()
        self.k = min(kernel, seq_len if seq_len % 2 else seq_len - 1)
        self.k = max(self.k, 1)
        self.trend, self.rest = nn.Linear(seq_len * n_in, out), nn.Linear(seq_len * n_in, out)

    def forward(self, x):
        z = x.transpose(1, 2)
        half = self.k // 2
        trend = F.avg_pool1d(F.pad(z, (half, half), mode="replicate"), self.k, stride=1)
        return self.trend(trend.flatten(1)) + self.rest((z - trend).flatten(1))


class LSTMAE(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.enc = nn.LSTM(n_in, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_in)

    def forward(self, x):
        h = self.enc(x)[0][:, -1].unsqueeze(1).expand(-1, x.size(1), -1)
        return self.out(self.dec(h)[0])


class TabAE(nn.Module):
    def __init__(self, n_in, hidden, latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class TabVAE(nn.Module):
    def __init__(self, n_in, hidden, latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_in))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        self.kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return self.dec(z)


class SVDDNet(nn.Module):
    """Deep SVDD: bias-free encoder trained to collapse normal data onto a fixed centre."""

    def __init__(self, n_in, hidden, latent):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden, bias=False), nn.ReLU(), nn.Linear(hidden, latent, bias=False))
        self.register_buffer("center", torch.zeros(latent))

    def forward(self, x):
        return self.net(x)


def make_net(spec):
    n, name = spec["n_in"], spec["net"]
    hidden, layers, out, seq = spec.get("hidden", 64), spec.get("layers", 1), spec.get("out", 1), spec.get("seq_len", 0)
    if name in ("rnn", "lstm", "gru"):
        return RNNNet(name, n, hidden, layers, out)
    if name == "bilstm":
        return RNNNet("lstm", n, hidden, layers, out, bidirectional=True)
    if name == "lstm_pred":
        return RNNNet("lstm", n, hidden, layers, n)
    if name == "tcn":
        return TCN(n, hidden, out, depth=4)
    if name == "wavenet":
        return TCN(n, hidden, out, depth=6)
    if name == "cnn1d":
        return CNN1D(n, hidden, out)
    if name == "dlinear":
        return DLinear(n, seq, out)
    if name == "lstm_ae":
        return LSTMAE(n, hidden)
    latent = spec.get("latent", max(2, n // 2))
    if name == "ae":
        return TabAE(n, hidden, latent)
    if name == "vae":
        return TabVAE(n, hidden, latent)
    if name == "svdd":
        return SVDDNet(n, hidden, latent)
    raise KeyError(name)


# ---- training loop ----------------------------------------------------------------
def loader(arrays, batch, shuffle=False):
    tensors = [torch.as_tensor(a) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch, shuffle=shuffle, drop_last=False)


def fit(net, make_loss, tr, va, epochs, lr, device, batch, patience=8):
    """Train with early stopping. On GPU out-of-memory, halve the batch; then fall back to CPU."""
    while True:
        try:
            return _fit(net, make_loss, tr, va, epochs, lr, device, batch, patience), device, batch
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or device != "cuda":
                raise
            torch.cuda.empty_cache()
            net.to("cpu")
            if batch > MIN_BATCH:
                batch = max(MIN_BATCH, batch // 2)
                log(f"GPU ran out of memory: retrying with batch size {batch}", "warn")
            else:
                device = "cpu"
                log("GPU ran out of memory at the smallest batch: continuing on CPU", "warn")


def _fit(net, make_loss, tr, va, epochs, lr, device, batch, patience):
    """`tr` and `va` are arrays held in memory, or callables taking a batch size and returning a DataLoader."""
    net.to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    train_loader = tr(batch) if callable(tr) else loader(tr, batch, shuffle=True)
    if callable(va):
        val_loader = va(batch)
    else:
        val_loader = loader(va, batch) if va is not None and len(va[0]) else None
    best, best_state, wait, history = float("inf"), None, 0, []
    for e in range(1, epochs + 1):
        net.train()
        total, n = 0.0, 0
        for b in train_loader:
            if len(b[0]) < 2:
                continue
            b = [x.to(device) for x in b]
            opt.zero_grad()
            loss = make_loss(net, b)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b[0])
            n += len(b[0])
        tl = total / max(n, 1)
        vl = tl
        if val_loader:
            net.eval()
            with torch.no_grad():
                total, n = 0.0, 0
                for b in val_loader:
                    if len(b[0]) < 2:
                        continue
                    b = [x.to(device) for x in b]
                    total += make_loss(net, b).item() * len(b[0])
                    n += len(b[0])
            vl = total / n if n else tl
        history.append((tl, vl))
        emit("epoch", epoch=e, total=epochs, train_loss=tl, val_loss=vl)
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                log(f"early stop at epoch {e}: no validation improvement for {patience} epochs")
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return history


def loss_chart(history):
    return dict(type="line", title="Training loss", xlabel="epoch",
                series=[dict(name="train", y=[h[0] for h in history]), dict(name="validation", y=[h[1] for h in history])])


@torch.no_grad()
def batched(net, X, device, batch=512):
    net.eval().to(device)
    return torch.cat([net(b[0].to(device)).cpu() for b in loader([X], batch)]).numpy()


def windows(X, seq_len, y=None):
    """Sliding windows over one split only, so no window spans two splits."""
    n = len(X) - seq_len + 1
    if n <= 0:
        raise DataError(f"Sequence length {seq_len} is longer than a data split ({len(X)} rows). Use a shorter sequence.")
    W = np.stack([X[i:i + seq_len] for i in range(n)]).astype(np.float32)
    return W if y is None else (W, y[seq_len - 1:])


def _save(job, net, spec, meta):
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(spec=spec, task=job["task"], **meta), indent=1, default=float))


def _setup(job, params):
    device = job.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA is not usable in this runtime: falling back to CPU", "warn")
        device = "cpu"
    torch.manual_seed(42)
    torch.set_num_threads(max(1, int(job.get("threads", 1))))
    return device, int(job.get("batch_size") or 64), params["epochs"], params["lr"]


# ---- forecasting ----------------------------------------------------------------
def forecast_recursive(net, tail, steps, device):
    h = list(tail)
    out = []
    net.eval().to(device)
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(h[-len(tail):], dtype=np.float32)[None, :, None], device=device)
            v = float(net(x).ravel()[0].cpu())
            out.append(v)
            h.append(v)
    return np.array(out)


def run_seq_forecast(df, features, job, algo, impl, params, ctx):
    target = job["dataset"]["target"]
    device, batch, epochs, lr = _setup(job, params)
    seq = params["seq_len"]
    stage("prepare")
    y = series_from(df, target)
    tr, va, te, _ = split_indices(len(y), None, job)
    mean, std = float(y[tr].mean()), float(y[tr].std() + 1e-9)
    s = (y - mean) / std
    log(f"chronological split: train {len(tr)}, validation {len(va)}, test {len(te)}; scaler fitted on train only")

    def pairs(part):
        W = windows(part[:-1], seq)
        return W[:, :, None], part[seq:].astype(np.float32)[:, None]

    stage("build")
    spec = dict(net=impl["net"], n_in=1, hidden=params["hidden"], layers=params.get("layers", 1), out=1, seq_len=seq)
    net = make_net(spec)
    log(f"{impl['net']} with {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    Xtr, ytr = pairs(s[tr])
    Xva, yva = pairs(s[va]) if len(va) > seq else (np.empty((0, seq, 1), np.float32), np.empty((0, 1), np.float32))
    mse = lambda m, b: F.mse_loss(m(b[0]), b[1])
    history, device, batch = fit(net, mse, (Xtr, ytr), (Xva, yva), epochs, lr, device, batch)
    stage("evaluate")
    y_hist = y[np.r_[tr, va]]
    pv = forecast_recursive(net, s[tr][-seq:], len(va), device) * std + mean
    pt = forecast_recursive(net, s[np.r_[tr, va]][-seq:], len(te), device) * std + mean
    val = forecast_metrics(pv, y[va], y[tr])
    test = forecast_metrics(pt, y[te], y_hist)
    horizon = int(job.get("horizon", 12))
    tail = s[-seq:]
    future = forecast_recursive(net, tail, horizon, device) * std + mean
    stage("save")
    pd.DataFrame({"actual": y[te], "forecast": pt}).to_csv(Path(job["run_dir"]) / "predictions.csv", index=False)
    pd.DataFrame({"step": range(1, horizon + 1), "forecast": future}).to_csv(Path(job["run_dir"]) / "future.csv", index=False)
    _save(job, net, spec, dict(kind="seq_forecast", target=target, mean=mean, std=std, seq_len=seq, tail=tail.tolist(), horizon=horizon))
    charts = forecast_charts(y_hist, y[te], pt, future) + [loss_chart(history)]
    return dict(primary=("mae", test["mae"], False), val=val, test=test, charts=charts, sample=None, params=params)


# ---- sequence classification --------------------------------------------------------
def _numeric(features):
    num = [f for f in features if f["type"] == "num"]
    if not num:
        raise DataError("Sequence models need numeric sensor columns; none were found.")
    if len(num) < len(features):
        log(f"sequence models use numeric columns only: ignoring {len(features) - len(num)} categorical", "warn")
    return num


def _standardise(df, num, tr):
    X = coerce(df, num).to_numpy(float)
    med = np.nanmedian(X[tr], 0)
    med = np.where(np.isnan(med), 0, med)
    X = _fill(X, med)
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-9
    return ((X - mean) / std).astype(np.float32), med, mean, std


def _fill(X, med):
    mask = np.isnan(X)
    X = X.copy()
    X[mask] = np.take(med, np.nonzero(mask)[1])
    return X


def run_seq_classification(df, features, job, algo, impl, params, ctx):
    target = job["dataset"]["target"]
    device, batch, epochs, lr = _setup(job, params)
    seq = params["seq_len"]
    stage("prepare")
    num = _numeric(features)
    classes = sorted(df[target].astype(str).unique().tolist())
    if len(classes) < 2:
        raise DataError("The target has only one class.")
    lookup = {c: i for i, c in enumerate(classes)}
    y = df[target].astype(str).map(lookup).to_numpy()
    tr, va, te, _ = split_indices(len(df), y, job)
    X, med, mean, std = _standardise(df, num, tr)
    Xtr, ytr = windows(X[tr], seq, y[tr])
    Xva, yva = windows(X[va], seq, y[va])
    Xte, yte = windows(X[te], seq, y[te])
    log(f"windows of {seq}: train {len(Xtr)}, validation {len(Xva)}, test {len(Xte)}; no window crosses a split")
    weights = len(ytr) / (len(classes) * np.maximum(np.bincount(ytr, minlength=len(classes)), 1))
    stage("build")
    spec = dict(net=impl["net"], n_in=len(num), hidden=params["hidden"], layers=params.get("layers", 1), out=len(classes), seq_len=seq)
    net = make_net(spec)
    log(f"{impl['net']} with {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    w = torch.tensor(weights, dtype=torch.float32)
    ce = lambda m, b: F.cross_entropy(m(b[0]), b[1], weight=w.to(b[0].device))
    history, device, batch = fit(net, ce, (Xtr, ytr), (Xva, yva), epochs, lr, device, batch)
    stage("evaluate")
    charts = []
    logits_v, logits_t = batched(net, Xva, device), batched(net, Xte, device)
    val = from_labels(logits_v.argmax(1), yva, classes, _softmax(logits_v))
    test = from_labels(logits_t.argmax(1), yte, classes, _softmax(logits_t), charts)
    charts.append(loss_chart(history))
    stage("save")
    out = df.iloc[te].iloc[seq - 1:].copy()
    out["prediction"] = [classes[i] for i in logits_t.argmax(1)]
    out.to_csv(Path(job["run_dir"]) / "predictions.csv", index=False)
    _save(job, net, spec, dict(kind="seq_class", features=num, classes=classes, target=target, seq_len=seq,
                               median=med.tolist(), mean=mean.tolist(), std=std.tolist()))
    return dict(primary=("f1_macro", test["f1_macro"], True), val=val, test=test, charts=charts, sample=Xte[:16], params=params)


def _softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ---- deep anomaly -------------------------------------------------------------------
def run_seq_anomaly(df, features, job, algo, impl, params, ctx):
    target = job["dataset"].get("target")
    device, batch, epochs, lr = _setup(job, params)
    seq = params["seq_len"]
    contamination = float(job.get("contamination", 0.05))
    stage("prepare")
    num = _numeric(features)
    y = anomaly_labels(df, job, target)
    job = dict(job, task="forecasting")      # time-ordered split for sequence data
    tr, va, te, _ = split_indices(len(df), None, job)
    X, med, mean, std = _standardise(df, num, tr)
    def part(idx):
        if y is None:
            return windows(X[idx], seq), None
        return windows(X[idx], seq, y[idx])
    Wtr, ltr = part(tr)
    Wva, _ = part(va)
    Wte, lte = part(te)
    if ltr is not None:
        Wtr = Wtr[ltr == 0]
        log(f"training on {len(Wtr)} windows labelled normal")
    stage("build")
    net_name = impl["net"]
    spec = dict(net=net_name, n_in=len(num), hidden=params["hidden"], layers=params.get("layers", 1), seq_len=seq)
    net = make_net(spec)
    log(f"{net_name} with {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    if net_name == "lstm_ae":
        loss = lambda m, b: F.mse_loss(m(b[0]), b[0])
    else:
        loss = lambda m, b: F.mse_loss(m(b[0][:, :-1]), b[0][:, -1])
    history, device, batch = fit(net, loss, (Wtr,), (Wva,), epochs, lr, device, batch)
    stage("evaluate")
    def err(W):
        if net_name == "lstm_ae":
            return ((batched(net, W, device) - W) ** 2).mean((1, 2))
        return ((batched(net, W[:, :-1], device) - W[:, -1]) ** 2).mean(1)
    df_te = df.iloc[te].iloc[seq - 1:]
    m, primary, charts, thr = anomaly_report(err(Wte), err(Wtr), lte, contamination, df_te, job)
    charts.append(loss_chart(history))
    stage("save")
    _save(job, net, spec, dict(kind="seq_anomaly", features=num, seq_len=seq, threshold=thr,
                               median=med.tolist(), mean=mean.tolist(), std=std.tolist()))
    return dict(primary=primary, val=None, test=m, charts=charts, sample=Wte[:16], params=params)


def _ae_loss(name):
    if name == "vae":
        return lambda m, b: F.mse_loss(m(b[0]), b[0]) + 1e-3 * m.kl
    if name == "svdd":
        return lambda m, b: ((m(b[0]) - m.center) ** 2).sum(1).mean()
    return lambda m, b: F.mse_loss(m(b[0]), b[0])


def tab_error(net, name, X, device):
    out = batched(net, X, device)
    if name == "svdd":
        return ((out - net.center.cpu().numpy()) ** 2).sum(1)
    return ((out - X) ** 2).mean(1)


def run_tab_ae(df, features, job, algo, impl, params, ctx):
    import joblib

    target = job["dataset"].get("target")
    device, batch, epochs, lr = _setup(job, params)
    contamination = float(job.get("contamination", 0.05))
    stage("prepare")
    y = anomaly_labels(df, job, target) if job["task"] == "anomaly" else None
    tr, va, te, _ = split_indices(len(df), None, job)
    if y is not None:
        tr = tr[y[tr] == 0]
    X = coerce(df, features)
    pre = make_preprocessor(features, "standard")
    Xtr = pre.fit_transform(X.iloc[tr]).astype(np.float32)
    Xva = pre.transform(X.iloc[va]).astype(np.float32)
    Xte = pre.transform(X.iloc[te]).astype(np.float32)
    stage("build")
    name = impl["net"]
    latent = params.get("n_components") or max(2, Xtr.shape[1] // 2)
    spec = dict(net=name, n_in=Xtr.shape[1], hidden=params["hidden"], latent=latent)
    net = make_net(spec)
    if name == "svdd":
        with torch.no_grad():
            net.center.copy_(net(torch.as_tensor(Xtr)).mean(0))
    log(f"{name} with {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    history, device, batch = fit(net, _ae_loss(name), (Xtr,), (Xva,), epochs, lr, device, batch)
    stage("evaluate")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, d / "pre.joblib")
    if job["task"] == "dimreduction":
        from sklearn.manifold import trustworthiness
        from .tabular import _projection_chart

        Xall = pre.transform(X).astype(np.float32)
        net.eval().to(device)
        with torch.no_grad():
            Z = torch.cat([net.enc(b[0].to(device)).cpu() for b in loader([Xall], 512)]).numpy()
        n = min(2000, len(Xall))
        sample = np.random.RandomState(42).choice(len(Xall), n, replace=False)
        m = dict(reconstruction_mse=float(tab_error(net, name, Xall, device).mean()),
                 trustworthiness=float(trustworthiness(Xall[sample], Z[sample], n_neighbors=min(5, max(1, n // 2 - 1)))),
                 output_dimensions=int(Z.shape[1]))
        stage("save")
        pd.concat([df.reset_index(drop=True), pd.DataFrame(Z, columns=[f"dim_{i + 1}" for i in range(Z.shape[1])])], axis=1).to_csv(
            Path(job["run_dir"]) / "predictions.csv", index=False)
        _save(job, net, spec, dict(kind="tab_ae", features=features, reducer=True))
        return dict(primary=("trustworthiness", m["trustworthiness"], True), val=None, test=m,
                    charts=_projection_chart(Z, None, "Latent space") + [loss_chart(history)], sample=X.iloc[te].head(16), params=params)
    m, primary, charts, thr = anomaly_report(tab_error(net, name, Xte, device), tab_error(net, name, Xtr, device),
                                             y[te] if y is not None else None, contamination, df.iloc[te], job)
    charts.append(loss_chart(history))
    stage("save")
    _save(job, net, spec, dict(kind="tab_ae", features=features, threshold=thr, reducer=False))
    return dict(primary=primary, val=None, test=m, charts=charts, sample=X.iloc[te].head(16), params=params)
