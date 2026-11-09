"""Image classification from a folder of class subfolders, using torchvision backbones."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

from .data import DataError
from .deep import _setup, fit, loss_chart
from .metrics import from_labels
from .protocol import log, stage

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class LeNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                               nn.AdaptiveAvgPool2d(5))
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, n))

    def forward(self, x):
        return self.c(self.f(x))


def _replace_head(net, n):
    """Point the backbone's final classifier at `n` classes."""
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        net.fc = nn.Linear(net.fc.in_features, n)
    elif hasattr(net, "head") and isinstance(net.head, nn.Linear):     # swin
        net.head = nn.Linear(net.head.in_features, n)
    elif hasattr(net, "heads"):                                        # vit
        net.heads.head = nn.Linear(net.heads.head.in_features, n)
    elif hasattr(net, "classifier"):
        c = net.classifier
        if isinstance(c, nn.Linear):
            net.classifier = nn.Linear(c.in_features, n)
        elif isinstance(c, nn.Sequential) and any(isinstance(m, nn.Linear) for m in c):
            i = max(j for j, m in enumerate(c) if isinstance(m, nn.Linear))
            c[i] = nn.Linear(c[i].in_features, n)
        elif isinstance(c, nn.Sequential):                             # squeezenet
            c[1] = nn.Conv2d(c[1].in_channels, n, 1)
    else:
        raise RuntimeError("Unknown classifier layout")
    return net


def make_arch(arch, n, pretrained=False):
    if arch == "lenet":
        return LeNet(n)
    if arch.startswith("timm:"):
        import timm

        return timm.create_model(arch[5:], pretrained=pretrained, num_classes=n)
    kw = {"weights": "DEFAULT" if pretrained else None}
    if arch == "googlenet" and not pretrained:
        kw.update(aux_logits=False, init_weights=False)
    net = getattr(torchvision.models, arch)(**kw)
    return _replace_head(net, n)


def eval_tf(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def run_image(job, algo, impl, params, ctx):
    from sklearn.model_selection import train_test_split

    device, batch, epochs, lr = _setup(job, params)
    arch = params.get("arch", impl["arch"])
    size = impl.get("fixed_size") or params["image_size"]
    pretrained = impl.get("transfer") or params.get("pretrained") == "yes"
    stage("validate")
    root = Path(job["dataset"]["path"])
    try:
        base = torchvision.datasets.ImageFolder(root)
    except (FileNotFoundError, OSError) as e:
        raise DataError(f"Could not read image folder: {e}") from e
    y = np.array(base.targets)
    classes = base.classes
    if len(classes) < 2:
        raise DataError("Image classification needs at least 2 class subfolders.")
    counts = np.bincount(y)
    log(f"{len(y)} images, {len(classes)} classes: " + ", ".join(f"{c}={n}" for c, n in zip(classes[:8], counts[:8])))
    if counts.min() < 5:
        raise DataError(f"Class '{classes[int(counts.argmin())]}' has only {counts.min()} images: at least 5 are needed to split.")
    stage("prepare")
    s = job.get("split") or {}
    val, test = s.get("val", 0.15), s.get("test", 0.15)
    tr, rest = train_test_split(np.arange(len(y)), test_size=val + test, random_state=42, stratify=y)
    va, te = train_test_split(rest, test_size=test / (val + test), random_state=42, stratify=y[rest])
    log(f"stratified split: train {len(tr)}, validation {len(va)}, test {len(te)}")
    train_tf = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tfe = eval_tf(size)
    stage("build")
    net = make_arch(arch, len(classes), pretrained)
    if impl.get("transfer"):
        head = {id(p) for m in (getattr(net, "fc", None), getattr(net, "classifier", None)) if m is not None for p in m.parameters()}
        for p in net.parameters():
            if id(p) not in head:
                p.requires_grad = False
        log("backbone frozen: training the new classification head only")
    log(f"{arch} with {sum(p.numel() for p in net.parameters()):,} parameters, image {size}px, batch {batch}, device {device}")
    stage("train")
    ce = lambda m, b: F.cross_entropy(_logits(m(b[0])), b[1])
    tr_data, va_data, te_data = Split(base, tr, train_tf), Split(base, va, tfe), Split(base, te, tfe)
    history, device, batch = fit(net, ce, lambda n: stream(tr_data, n, True), lambda n: stream(va_data, n), epochs, lr, device, batch, patience=6)
    stage("evaluate")
    charts = []
    logits_v, yva = _predict(net, va_data, device, batch)
    logits_t, yte = _predict(net, te_data, device, batch)
    val_m = from_labels(logits_v.argmax(1), yva, classes, None)
    test_m = from_labels(logits_t.argmax(1), yte, classes, None, charts)
    Xte = torch.stack([te_data[i][0] for i in range(min(8, len(te_data)))])
    charts.append(loss_chart(history))
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="image", task=job["task"], arch=arch, classes=classes, image_size=size), indent=1))
    return dict(primary=("f1_macro", test_m["f1_macro"], True), val=val_m, test=test_m, charts=charts,
                sample=Xte[:8].numpy(), params=params)


def _logits(out):
    return out.logits if hasattr(out, "logits") else out


class Split(torch.utils.data.Dataset):
    """Decodes images one at a time, so a dataset never has to fit in RAM."""

    def __init__(self, base, idx, tf):
        self.base, self.idx, self.tf = base, idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        path, label = self.base.samples[self.idx[i]]
        return self.tf(self.base.loader(path)), label


def stream(dataset, batch, shuffle=False):
    workers = 0 if os.name == "nt" or len(dataset) < 200 else 2
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers)


@torch.no_grad()
def _predict(net, dataset, device, batch):
    net.eval().to(device)
    logits, labels = [], []
    for x, y in stream(dataset, batch):
        logits.append(_logits(net(x.to(device))).cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()
