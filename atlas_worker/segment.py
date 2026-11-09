"""Semantic segmentation from images/ and masks/ folders."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, to_tensor

from . import fileinfo
from .data import DataError
from .deep import _setup, fit, loss_chart
from .protocol import log, stage

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _mask_values(path):
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0].astype(np.int64) * 65536 + m[..., 1].astype(np.int64) * 256 + m[..., 2]
    return m


def pairs(job):
    root = Path(job["dataset"]["path"])
    info = fileinfo.detect_folder(root)
    if info["kind"] != "segmentation":
        raise DataError("This folder needs matching images/ and masks/ subfolders.")
    masks = {f.stem: f for f in (root / info["masks"]).rglob("*") if f.suffix.lower() in fileinfo.IMAGE_EXT}
    out = [(f, masks[f.stem]) for f in sorted((root / info["images"]).rglob("*")) if f.suffix.lower() in fileinfo.IMAGE_EXT and f.stem in masks]
    if len(out) < 10:
        raise DataError(f"Only {len(out)} image and mask pairs matched by file name: at least 10 are needed.")
    return out


class SegData(Dataset):
    def __init__(self, items, lookup, size):
        self.items, self.lookup, self.size = items, lookup, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        im_path, mask_path = self.items[i]
        im = Image.open(im_path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        raw = _mask_values(mask_path)
        small = np.array(Image.fromarray(raw.astype(np.int32), mode="I").resize((self.size, self.size), Image.NEAREST))
        y = np.vectorize(lambda v: self.lookup.get(int(v), 0))(small).astype(np.int64)
        return normalize(to_tensor(im), MEAN, STD), torch.from_numpy(y)


class TVWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]


def make_net(arch, n):
    if arch.startswith("tv:"):
        import torchvision

        return TVWrap(getattr(torchvision.models.segmentation, arch[3:])(weights=None, weights_backbone=None, num_classes=n, aux_loss=False))
    import segmentation_models_pytorch as smp

    name, encoder = arch.split(":")
    return getattr(smp, name)(encoder_name=encoder, encoder_weights=None, classes=n, activation=None)


def _ious(pred, y, n):
    conf = np.bincount(n * y.ravel() + pred.ravel(), minlength=n * n).reshape(n, n)
    tp = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - tp
    present = conf.sum(1) > 0
    iou = tp / np.maximum(union, 1)
    return float(iou[present].mean()), float(tp.sum() / max(conf.sum(), 1)), iou


def run(job, algo, impl, params, ctx):
    device, batch, epochs, lr = _setup(job, params)
    arch = params.get("arch", impl["arch"])
    size = params["image_size"] // 32 * 32
    stage("validate")
    items = pairs(job)
    values = set()
    for _, m in items[:200]:
        values |= set(np.unique(_mask_values(m)).tolist())
    if len(values) > 256:
        raise DataError(f"Masks hold {len(values)} distinct values: they must store one value per class.")
    if len(values) < 2:
        raise DataError("Every mask has only one value, so there is nothing to segment.")
    ordered = sorted(values)
    lookup = {int(v): i for i, v in enumerate(ordered)}
    n = len(ordered)
    log(f"{len(items)} pairs, {n} classes (mask values {ordered[:8]}{'...' if n > 8 else ''})")
    stage("prepare")
    order = np.random.RandomState(42).permutation(len(items))
    s = job.get("split") or {}
    a, b = int(len(items) * s.get("val", 0.15)) or 1, int(len(items) * s.get("test", 0.15)) or 1
    va, te, tr = order[:a], order[a:a + b], order[a + b:]
    part = lambda idx: SegData([items[i] for i in idx], lookup, size)
    log(f"train {len(tr)}, validation {len(va)}, test {len(te)}; image size {size}px (a multiple of 32)")
    stage("build")
    net = make_net(arch, n)
    log(f"{arch} with {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    stage("train")
    loss = lambda m, b: F.cross_entropy(m(b[0]), b[1])
    dl = lambda ds, shuffle=False: (lambda k: DataLoader(ds, batch_size=k, shuffle=shuffle, drop_last=shuffle and len(ds) > k))
    history, device, batch = fit(net, loss, dl(part(tr), True), dl(part(va)), epochs, lr, device, batch, patience=6)
    stage("evaluate")

    @torch.no_grad()
    def score(ds):
        net.eval().to(device)
        ps, ys = [], []
        for x, y in DataLoader(ds, batch_size=batch):
            ps.append(net(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
        return _ious(np.concatenate(ps), np.concatenate(ys), n)

    val_iou, val_acc, _ = score(part(va))
    test_iou, test_acc, per = score(part(te))
    m = dict(mean_iou=test_iou, pixel_accuracy=test_acc)
    labels = [str(v) for v in ordered][:20]
    charts = [dict(type="bars", title="IoU per mask value (test)", labels=labels, values=per[:20].tolist()), loss_chart(history)]
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="segment", task=job["task"], arch=arch, values=ordered, image_size=size), indent=1))
    sample = torch.stack([part(te)[i][0] for i in range(min(2, len(te)))]).numpy()
    return dict(primary=("mean_iou", test_iou, True), val=dict(mean_iou=val_iou, pixel_accuracy=val_acc), test=m, charts=charts, sample=sample, params=params)


def _load(p):
    net = make_net(p.meta["arch"], len(p.meta["values"]))
    net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
    return net.eval()


def predict(p, images=None, out_dir=None, **_):
    net = p.cached(lambda: _load(p))
    m = p.meta
    values = np.array(m["values"])
    rows = []
    for path in images:
        im = Image.open(path).convert("RGB")
        x = normalize(to_tensor(im.resize((m["image_size"], m["image_size"]), Image.BILINEAR)), MEAN, STD)[None]
        with torch.no_grad():
            y = net(x).argmax(1)[0].numpy()
        mask = Image.fromarray(values[y].astype(np.uint8 if values.max() < 256 else np.int32)).resize(im.size, Image.NEAREST)
        row = dict(image=str(path))
        if out_dir:
            f = Path(out_dir) / f"{Path(path).stem}_mask.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            mask.save(f)
            row["mask"] = str(f)
        else:
            buf = io.BytesIO()
            mask.save(buf, "PNG")
            row["mask_b64"] = base64.b64encode(buf.getvalue()).decode()
        for v, frac in zip(*np.unique(y, return_counts=True)):
            row[f"value_{values[v]}"] = float(frac) / y.size
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def onnx_parts(p, sample):
    return _load(p), (torch.as_tensor(sample).float(),), ["input"]
