"""Object detection from a YOLO-format or COCO-format folder."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import fileinfo
from .data import DataError
from .protocol import emit, log, stage


# ---- dataset: everything becomes [{"image", "split", "objects": [(cls, cx, cy, w, h)]}] normalised ----
def _split_of(path: Path):
    parts = {p.lower() for p in path.parts[-4:]}
    if parts & {"val", "valid", "validation"}:
        return "val"
    if "test" in parts:
        return "test"
    return "train" if "train" in parts else None


def _load_yolo(root):
    samples = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in fileinfo.IMAGE_EXT or not f.is_file():
            continue
        objects = []
        lab = fileinfo.label_path(f)
        if lab.exists():
            for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                bits = line.split()
                if len(bits) == 5:
                    objects.append((int(float(bits[0])), *map(float, bits[1:])))
        samples.append(dict(image=str(f), split=_split_of(f), objects=objects))
    n = 1 + max((o[0] for s in samples for o in s["objects"]), default=-1)
    names = fileinfo.yolo_names(root)
    return samples, names if names and len(names) >= n else [f"class_{i}" for i in range(n)]


def _load_coco(root, files):
    samples, cats = [], {}
    for f in files:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        for c in sorted(doc.get("categories", []), key=lambda c: c["id"]):
            cats.setdefault(c["id"], c["name"])
    index = {c: i for i, c in enumerate(cats)}
    for f in files:
        f = Path(f)
        doc = json.loads(f.read_text(encoding="utf-8"))
        anns = {}
        for a in doc["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in doc["images"]:
            name = Path(im["file_name"])
            candidates = [f.parent / name, root / name, root / "images" / name, root / "images" / f.stem / name, f.parent.parent / "images" / name, root / f.stem.replace("instances_", "") / name]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                continue
            w, h = im["width"], im["height"]
            objs = [(index[a["category_id"]], (a["bbox"][0] + a["bbox"][2] / 2) / w, (a["bbox"][1] + a["bbox"][3] / 2) / h, a["bbox"][2] / w, a["bbox"][3] / h)
                    for a in anns.get(im["id"], []) if a.get("bbox") and a["bbox"][2] > 0 and a["bbox"][3] > 0]
            samples.append(dict(image=str(path), split=_split_of(f) or _split_of(path), objects=objs))
    return samples, list(cats.values())


def load_samples(job):
    root = Path(job["dataset"]["path"])
    info = fileinfo.detect_folder(root)
    if info["kind"] != "detection":
        raise DataError("This folder is not a detection dataset. It needs YOLO labels (images/ and labels/) or a COCO annotation file.")
    samples, names = _load_coco(root, info["annotations"]) if info["format"] == "coco" else _load_yolo(root)
    if len(samples) < 8:
        raise DataError(f"Only {len(samples)} labelled images were found: at least 8 are needed.")
    if not names or not any(s["objects"] for s in samples):
        raise DataError("No bounding boxes were found in this dataset.")
    rs = np.random.RandomState(42)
    s = job.get("split") or {}
    val, test = s.get("val", 0.15), s.get("test", 0.15)
    have_val = any(x["split"] == "val" for x in samples)
    for x in samples:
        if x["split"] is None or (x["split"] == "train" and not have_val):
            r = rs.random_sample()
            x["split"] = "val" if r < val else "test" if r < val + test else "train"
    if not any(x["split"] == "test" for x in samples):         # keep a set that early stopping never touches
        pool = [x for x in samples if x["split"] == "val"]
        for x in pool[: len(pool) // 2]:
            x["split"] = "test"
    counts = {k: sum(x["split"] == k for x in samples) for k in ("train", "val", "test")}
    if min(counts.values()) == 0:
        raise DataError(f"Could not form all three splits from {len(samples)} images: {counts}. Add more labelled images.")
    log(f"{len(samples)} images, {sum(len(x['objects']) for x in samples)} boxes, {len(names)} classes; train {counts['train']}, validation {counts['val']}, test {counts['test']}")
    return samples, names


# ---- metric ------------------------------------------------------------------------
def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def average_precision(preds, truths, n_classes, thr=0.5):
    """Mean AP over classes that appear. preds: [(boxes, scores, labels)], truths: [(boxes, labels)]."""
    aps = {}
    for c in range(n_classes):
        total = sum(int((t[1] == c).sum()) for t in truths)
        if not total:
            continue
        rows = [(s, i, b) for i, (bx, sc, lb) in enumerate(preds) for b, s, l in zip(bx, sc, lb) if l == c]
        rows.sort(key=lambda r: -r[0])
        used = [set() for _ in truths]
        tp = np.zeros(len(rows))
        for k, (_, i, box) in enumerate(rows):
            gt = truths[i][0][truths[i][1] == c]
            if len(gt):
                ious = _iou(box, gt)
                j = int(ious.argmax())
                if ious[j] >= thr and j not in used[i]:
                    used[i].add(j)
                    tp[k] = 1
        cum = np.cumsum(tp)
        recall = cum / total
        precision = cum / np.maximum(np.arange(1, len(rows) + 1), 1)
        aps[c] = float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))
    return float(np.mean(list(aps.values()))) if aps else 0.0, aps


# ---- torchvision detectors -----------------------------------------------------------
class TVData:
    def __init__(self, items, size):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        s = self.items[i]
        im = Image.open(s["image"]).convert("RGB")
        w, h = im.size
        boxes = [[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h] for _, x, y, bw, bh in s["objects"]]
        target = dict(boxes=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                      labels=torch.as_tensor([o[0] + 1 for o in s["objects"]], dtype=torch.int64))
        return to_tensor(im), target


def make_tv(arch, n, size, pretrained):
    from torchvision.models import detection as D

    kw = dict(weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=n + 1)
    if not arch.startswith("ssd"):
        kw.update(min_size=size, max_size=int(size * 1.34))
    if not pretrained and not arch.startswith("ssd"):
        kw["trainable_backbone_layers"] = None
    if not pretrained:
        kw.pop("trainable_backbone_layers", None)
    return getattr(D, arch)(**kw)


def _collate(batch):
    return tuple(zip(*batch))


def _tv_predict(net, loader_, device, score=0.05):
    import torch

    net.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, targets in loader_:
            for out, t in zip(net([i.to(device) for i in imgs]), targets):
                preds.append((out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()))
                truths.append((t["boxes"].numpy(), t["labels"].numpy()))
    return preds, truths


def run_torchvision(job, algo, impl, params, ctx):
    import torch
    from torch.utils.data import DataLoader

    from .deep import _setup, loss_chart

    device, batch, epochs, lr = _setup(job, params)
    arch = params.get("arch", impl["arch"])
    size = params["image_size"]
    stage("validate")
    samples, names = load_samples(job)
    stage("prepare")
    part = lambda k: [s for s in samples if s["split"] == k]
    tr, va, te = part("train"), part("val"), part("test")
    dl = lambda items, shuffle=False: DataLoader(TVData(items, size), batch_size=batch, shuffle=shuffle, collate_fn=_collate)
    stage("build")
    net = make_tv(arch, len(names), size, params.get("pretrained") == "yes").to(device)
    log(f"{arch} with {sum(p.numel() for p in net.parameters()):,} parameters, batch {batch}, device {device}")
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    stage("train")
    history, best, best_state = [], -1.0, None
    va_dl = dl(va)
    for e in range(1, epochs + 1):
        net.train()
        total, n = 0.0, 0
        for imgs, targets in dl(tr, True):
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(net(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            total += loss.item()
            n += 1
        val_map, _ = average_precision(*_tv_predict(net, va_dl, device), len(names) + 1)
        history.append((total / max(n, 1), 1 - val_map))
        emit("epoch", epoch=e, total=epochs, train_loss=total / max(n, 1), val_loss=1 - val_map)
        if val_map >= best:
            best = val_map
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    stage("evaluate")
    preds, truths = _tv_predict(net, dl(te), device)
    test_map, per = average_precision(preds, truths, len(names) + 1)
    m = dict(map50=test_map, val_map50=best, test_images=len(te))
    charts = [dict(type="bars", title="AP@0.5 per class (test)", labels=[names[c - 1] for c in per], values=list(per.values())), loss_chart(history)]
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), d / "net.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="detect_tv", task=job["task"], arch=arch, names=names, image_size=size), indent=1))
    return dict(primary=("map50", test_map, True), val=dict(map50=best), test=m, charts=charts, sample=[s["image"] for s in te[:2]], params=params)


# ---- Ultralytics ---------------------------------------------------------------------
def _link(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:                       # Windows without developer mode
        shutil.copy2(src, dst)


def build_yolo_dir(samples, names, work):
    """One uniform layout from any source format. Images are linked, never duplicated."""
    work = Path(work)
    for s in samples:
        key = {"train": "train", "val": "val", "test": "test"}[s["split"]]
        im_dir, lab_dir = work / "images" / key, work / "labels" / key
        im_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{abs(hash(s['image'])):x}_{Path(s['image']).stem}"
        _link(s["image"], im_dir / f"{stem}{Path(s['image']).suffix}")
        (lab_dir / f"{stem}.txt").write_text("".join(" ".join(map(str, o)) + "\n" for o in s["objects"]))
    names_yaml = "\n".join(f"  {i}: {json.dumps(n)}" for i, n in enumerate(names))
    yaml = work / "data.yaml"
    yaml.write_text(f"path: {json.dumps(str(work))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names_yaml}\n")
    return yaml


def _ultra_class(arch):
    import ultralytics

    return ultralytics.RTDETR if arch.startswith("rtdetr") else ultralytics.YOLO


def run_ultralytics(job, algo, impl, params, ctx):
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    run = Path(job["run_dir"])
    os.environ.setdefault("YOLO_CONFIG_DIR", str(run.parent.parent / "cache" / "ultralytics"))
    import ultralytics  # noqa: F401  (after the environment above is set)
    import torch

    arch, size, epochs = params.get("arch", impl["arch"]), params["image_size"], params["epochs"]
    device = job.get("device", "cpu")
    device = 0 if device == "cuda" and torch.cuda.is_available() else "cpu"
    stage("validate")
    samples, names = load_samples(job)
    stage("prepare")
    yaml = build_yolo_dir(samples, names, run / "yolo_data")
    stage("build")
    pretrained = params.get("pretrained") == "yes"
    source = f"{arch}.pt" if pretrained else f"{arch}.yaml"
    model = _ultra_class(arch)(source)
    log(f"{source} (Ultralytics, AGPL-3.0), image {size}px, device {device}")

    def on_epoch(trainer):
        parts = trainer.tloss.values() if isinstance(trainer.tloss, dict) else [trainer.tloss]
        loss = sum(float(v.sum()) if hasattr(v, "sum") else float(v) for v in parts)
        fit = float((trainer.metrics or {}).get("metrics/mAP50(B)", 0.0))
        emit("epoch", epoch=trainer.epoch + 1, total=trainer.epochs, train_loss=loss, val_loss=1 - fit)

    model.add_callback("on_fit_epoch_end", on_epoch)
    stage("train")
    model.train(data=str(yaml), epochs=epochs, imgsz=size, batch=int(job.get("batch_size") or 8), device=device, workers=0 if os.name == "nt" else 2,
                project=str(run / "ultra"), name="train", exist_ok=True, plots=False, verbose=False, amp=device != "cpu", patience=max(10, epochs // 3),
                seed=42, lr0=params["lr"])
    best = run / "ultra" / "train" / "weights" / "best.pt"
    if not best.exists():
        best = run / "ultra" / "train" / "weights" / "last.pt"
    stage("evaluate")
    final = _ultra_class(arch)(str(best))
    res = final.val(data=str(yaml), split="test", imgsz=size, batch=int(job.get("batch_size") or 8), device=device, workers=0, plots=False, verbose=False,
                    project=str(run / "ultra"), name="test", exist_ok=True)
    m = dict(map50=float(res.box.map50), map50_95=float(res.box.map), precision=float(res.box.mp), recall=float(res.box.mr))
    per = {names[int(c)]: float(a) for c, a in zip(res.box.ap_class_index, res.box.ap50)}
    charts = [dict(type="bars", title="AP@0.5 per class (test)", labels=list(per), values=list(per.values()))] if per else []
    stage("save")
    d = run / "model"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, d / "best.pt")
    (d / "meta.json").write_text(json.dumps(dict(kind="detect_ultra", task=job["task"], arch=arch, names=names, image_size=size), indent=1))
    shutil.rmtree(run / "yolo_data", ignore_errors=True)
    shutil.rmtree(run / "ultra", ignore_errors=True)
    return dict(primary=("map50", m["map50"], True), val=None, test=m, charts=charts, sample=[s["image"] for s in samples if s["split"] == "test"][:2], params=params)


# ---- prediction and export -------------------------------------------------------------
def _rows(path, boxes, scores, labels, names):
    return [dict(image=str(path), label=names[int(l)], score=float(s), x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])) for b, s, l in zip(boxes, scores, labels)]


def predict(p, images=None, conf=0.25, **_):
    m = p.meta
    rows = []
    if p.kind == "detect_ultra":
        model = p.cached(lambda: _ultra_class(m["arch"])(str(p.dir / "best.pt")))
        for im in images:
            b = model.predict(str(im), imgsz=m["image_size"], conf=conf, verbose=False)[0].boxes
            rows += _rows(im, b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), m["names"])
    else:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        def load():
            net = make_tv(m["arch"], len(m["names"]), m["image_size"], False)
            net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
            return net.eval()

        net = p.cached(load)
        with torch.no_grad():
            for im in images:
                o = net([to_tensor(Image.open(im).convert("RGB"))])[0]
                keep = o["scores"] >= conf
                rows += _rows(im, o["boxes"][keep].numpy(), o["scores"][keep].numpy(), (o["labels"][keep] - 1).numpy(), m["names"])
    return pd.DataFrame(rows, columns=["image", "label", "score", "x1", "y1", "x2", "y2"])


def export_onnx(p, sample, out):
    """Export and compare final detections from the ONNX file and the original on a real image."""
    import onnxruntime as ort

    path = out / "model.onnx"
    m = p.meta
    if p.kind == "detect_ultra":
        model = _ultra_class(m["arch"])(str(p.dir / "best.pt"))
        made = Path(model.export(format="onnx", imgsz=m["image_size"], opset=17, simplify=False, verbose=False))
        shutil.move(str(made), path)
        im = sample[0]
        a = model.predict(im, imgsz=m["image_size"], conf=0.001, max_det=20, verbose=False)[0].boxes.xyxy.cpu().numpy()
        b = _ultra_class(m["arch"])(str(path), task="detect").predict(im, imgsz=m["image_size"], conf=0.001, max_det=20, verbose=False)[0].boxes.xyxy.cpu().numpy()
        if a.shape != b.shape:
            return False, dict(reason=f"box count differs: {a.shape[0]} vs {b.shape[0]}")
        diff = float(np.abs(a - b).max()) if a.size else 0.0
        return diff < 2.0, dict(max_difference=diff, rows=int(a.shape[0]), unit="pixels")
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    net = make_tv(m["arch"], len(m["names"]), m["image_size"], False)
    net.load_state_dict(torch.load(p.dir / "net.pt", map_location="cpu"))
    net.eval()
    x = to_tensor(Image.open(sample[0]).convert("RGB"))
    with torch.no_grad():
        ref = net([x])[0]
    torch.onnx.export(net, ([x],), str(path), input_names=["image"], output_names=["boxes", "labels", "scores"], opset_version=11, dynamo=False,
                      dynamic_axes={"image": {1: "h", 2: "w"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}})
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    if len(got[0]) != len(ref["boxes"]):
        return False, dict(reason="box count differs")
    diff = float(np.abs(got[0] - ref["boxes"].numpy()).max()) if len(got[0]) else 0.0
    return diff < 2.0, dict(max_difference=diff, rows=len(got[0]), unit="pixels")
