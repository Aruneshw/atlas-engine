"""Audio and video classification from class-subfolder datasets."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import fileinfo
from .data import DataError
from .deep import _setup, fit, loss_chart
from .metrics import from_labels
from .protocol import log, stage


def _gather(root, extensions):
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and any(f.suffix.lower() in extensions for f in d.rglob("*")))
    if len(classes) < 2:
        raise DataError("This needs one subfolder per class, with at least two classes.")
    items = [(str(f), i) for i, c in enumerate(classes) for f in sorted((root / c).rglob("*")) if f.suffix.lower() in extensions]
    y = np.array([b for _, b in items])
    counts = np.bincount(y)
    if counts.min() < 5:
        raise DataError(f"Class '{classes[int(counts.argmin())]}' has only {counts.min()} files: at least 5 are needed to split.")
    log(f"{len(items)} files, {len(classes)} classes: " + ", ".join(f"{c}={n}" for c, n in zip(classes[:8], counts[:8])))
    return items, y, classes


def _split(y, job):
    from sklearn.model_selection import train_test_split

    s = job.get("split") or {}
    val, test = s.get("val", 0.15), s.get("test", 0.15)
    tr, rest = train_test_split(np.arange(len(y)), test_size=val + test, random_state=42, stratify=y)
    va, te = train_test_split(rest, test_size=test / (val + test), random_state=42, stratify=y[rest])
    log(f"stratified split: train {len(tr)}, validation {len(va)}, test {len(te)}")
    return tr, va, te


def _train(job, params, net, make, items, y, classes, kind, extra, patience=6):
    """Shared training, evaluation and saving. `make(items)` builds a Dataset."""
    device, batch, epochs, lr = _setup(job, params)
    tr, va, te = _split(y, job)
    pick = lambda idx: make([items[i] for i in idx])
    log(f"{sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    loss = lambda m, b: F.cross_entropy(m(b[0]), b[1])
    dl = lambda ds, shuffle=False: (lambda k: DataLoader(ds, batch_size=k, shuffle=shuffle))
    history, device, batch = fit(net, loss, dl(pick(tr), True), dl(pick(va)), epochs, lr, device, batch, patience=patience)
    stage("evaluate")

    @torch.no_grad()
    def infer(ds):
        net.eval().to(device)
        out, ys = [], []
        for x, yb in DataLoader(ds, batch_size=batch):
            out.append(net(x.to(device)).cpu())
            ys.append(yb)
        return torch.softmax(torch.cat(out), 1).numpy(), torch.cat(ys).numpy()

    charts = []
    pv, yv = infer(pick(va))
    pt, yt = infer(pick(te))
    val = from_labels(pv.argmax(1), yv, classes, pv)
    test = from_labels(pt.argmax(1), yt, classes, pt, charts)
    charts.append(loss_chart(history))
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind=kind, task=job["task"], classes=classes, **extra), indent=1))
    sample = torch.stack([pick(te)[i][0] for i in range(min(4, len(te)))]).numpy()
    return dict(primary=("f1_macro", test["f1_macro"], True), val=val, test=test, charts=charts, sample=sample, params=params)


def _classify_rows(files, proba, classes):
    out = pd.DataFrame({"file": [str(f) for f in files], "prediction": [classes[i] for i in proba.argmax(1)]})
    for i, c in enumerate(classes):
        out[f"proba_{c}"] = proba[:, i]
    return out


# ---- audio --------------------------------------------------------------------------
def mel_filters(sr, n_fft, n_mels):
    hz2mel = lambda f: 2595 * math.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = torch.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = torch.tensor([mel2hz(float(m)) for m in pts]) * n_fft / sr
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    freqs = torch.arange(n_fft // 2 + 1).float()
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        fb[i] = torch.clamp(torch.minimum((freqs - lo) / (mid - lo + 1e-6), (hi - freqs) / (hi - mid + 1e-6)), min=0)
    return fb


def load_wave(path, sr, seconds):
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        wav, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise DataError(f"Could not read audio file {Path(path).name}: {e}") from e
    wav = wav.mean(1)
    if rate != sr:
        g = math.gcd(sr, rate)
        wav = resample_poly(wav, sr // g, rate // g).astype(np.float32)
    n = int(sr * seconds)
    return np.pad(wav, (0, max(0, n - len(wav))))[:n]


class AudioFeatures:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fb = mel_filters(cfg["sr"], cfg["n_fft"], cfg["n_mels"])
        self.window = torch.hann_window(cfg["n_fft"])

    def __call__(self, wav):
        c = self.cfg
        spec = torch.stft(torch.as_tensor(wav), c["n_fft"], hop_length=c["hop"], window=self.window, return_complex=True).abs() ** 2
        mel = torch.log(self.fb @ spec + 1e-6)
        if c["feature"] == "mfcc":
            n = mel.shape[0]
            k = torch.arange(n).float()
            dct = torch.cos(math.pi / n * (k + 0.5)[None] * k[:, None])
            mel = (dct @ mel)[:c["n_mfcc"]]
        return ((mel - mel.mean()) / (mel.std() + 1e-6))[None]


class AudioData(Dataset):
    def __init__(self, items, cfg):
        self.items, self.cfg, self.feat = items, cfg, AudioFeatures(cfg)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.feat(load_wave(path, self.cfg["sr"], self.cfg["seconds"])), label


class AudioCNN(nn.Module):
    def __init__(self, n, width=32):
        super().__init__()
        layers, c = [], 1
        for k in (width, width * 2, width * 4):
            layers += [nn.Conv2d(c, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(), nn.MaxPool2d(2)]
            c = k
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(c, n)

    def forward(self, x):
        return self.head(self.net(x).mean((2, 3)))


class TDNN(nn.Module):
    """x-vector style: dilated 1-D convolutions, then mean and std pooling over time."""

    def __init__(self, n_in, n, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_in, width, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(width),
                                 nn.Conv1d(width, width, 3, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm1d(width),
                                 nn.Conv1d(width, width, 3, dilation=3, padding=3), nn.ReLU(), nn.BatchNorm1d(width))
        self.head = nn.Linear(2 * width, n)

    def forward(self, x):
        h = self.net(x.squeeze(1))
        return self.head(torch.cat([h.mean(2), h.std(2)], 1))


def make_audio(net_name, cfg, n):
    in_bins = cfg["n_mfcc"] if cfg["feature"] == "mfcc" else cfg["n_mels"]
    return TDNN(in_bins, n) if net_name == "tdnn" else AudioCNN(n)


def run_audio(job, algo, impl, params, ctx):
    stage("validate")
    items, y, classes = _gather(job["dataset"]["path"], fileinfo.AUDIO_EXT)
    stage("prepare")
    cfg = dict(sr=params["sample_rate"], seconds=params["seconds"], n_fft=400 * params["sample_rate"] // 16000, hop=160 * params["sample_rate"] // 16000,
               n_mels=params["n_mels"], n_mfcc=20, feature=impl["feature"], net=impl["net"])
    stage("build")
    net = make_audio(impl["net"], cfg, len(classes))
    return _train(job, params, net, lambda it: AudioData(it, cfg), items, y, classes, "audio", dict(audio=cfg))


# ---- video --------------------------------------------------------------------------
def read_clip(path, frames, size):
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise DataError(f"Could not read video file {Path(path).name}.")
    want = set(np.linspace(0, total - 1, frames).astype(int).tolist())
    out = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size)))
    cap.release()
    if not out:
        raise DataError(f"No frames could be decoded from {Path(path).name}.")
    while len(out) < frames:
        out.append(out[-1])
    arr = (np.stack(out[:frames]).astype(np.float32) / 255 - 0.45) / 0.225
    return torch.from_numpy(arr).permute(3, 0, 1, 2)     # channels, time, height, width


class VideoData(Dataset):
    def __init__(self, items, frames, size):
        self.items, self.frames, self.size = items, frames, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return read_clip(self.items[i][0], self.frames, self.size), self.items[i][1]


class C3D(nn.Module):
    def __init__(self, n, w=32):
        super().__init__()
        layers, c = [], 3
        for i, k in enumerate((w, w * 2, w * 4, w * 4)):
            pool = 2 if i < 2 else (1, 2, 2)      # keeps short clips from running out of time steps
            layers += [nn.Conv3d(c, k, 3, padding=1), nn.BatchNorm3d(k), nn.ReLU(), nn.MaxPool3d(pool)]
            c = k
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(c, n)

    def forward(self, x):
        return self.head(self.net(x).mean((2, 3, 4)))


class CNNLSTM(nn.Module):
    def __init__(self, n, hidden=128):
        super().__init__()
        import torchvision

        cnn = torchvision.models.resnet18(weights=None)
        cnn.fc = nn.Identity()
        self.cnn, self.rnn, self.head = cnn, nn.LSTM(512, hidden, batch_first=True), nn.Linear(hidden, n)

    def forward(self, x):
        b, c, t, h, w = x.shape
        z = self.cnn(x.transpose(1, 2).reshape(b * t, c, h, w)).reshape(b, t, -1)
        return self.head(self.rnn(z)[0][:, -1])


def make_video(arch, n):
    if arch == "c3d":
        return C3D(n)
    if arch == "cnn_lstm":
        return CNNLSTM(n)
    import torchvision

    net = getattr(torchvision.models.video, arch)(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n)
    return net


def run_video(job, algo, impl, params, ctx):
    stage("validate")
    items, y, classes = _gather(job["dataset"]["path"], fileinfo.VIDEO_EXT)
    stage("prepare")
    arch, frames, size = params.get("arch", impl["arch"]), params["frames"], params["image_size"]
    stage("build")
    net = make_video(arch, len(classes))
    return _train(job, params, net, lambda it: VideoData(it, frames, size), items, y, classes, "video", dict(arch=arch, frames=frames, image_size=size))


# ---- prediction --------------------------------------------------------------------
def _net(p):
    def load():
        m = p.meta
        n = len(m["classes"])
        net = make_audio(m["audio"]["net"], m["audio"], n) if p.kind == "audio" else make_video(m["arch"], n)
        net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
        return net.eval()

    return p.cached(load)


def predict(p, images=None, **_):
    m = p.meta
    net = _net(p)
    with torch.no_grad():
        if p.kind == "audio":
            feat = AudioFeatures(m["audio"])
            xs = torch.stack([feat(load_wave(f, m["audio"]["sr"], m["audio"]["seconds"])) for f in images])
        else:
            xs = torch.stack([read_clip(f, m["frames"], m["image_size"]) for f in images])
        proba = torch.softmax(net(xs), 1).numpy()
    return _classify_rows(images, proba, m["classes"])


def onnx_parts(p, sample):
    return _net(p), (torch.as_tensor(sample).float(),), ["input"]
