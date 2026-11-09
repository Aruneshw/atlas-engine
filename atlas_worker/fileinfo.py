"""File and folder detection. Standard library only: the GUI uses this before any runtime exists."""
from __future__ import annotations

import csv
import re
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
TEXT_DELIMITED = {".csv", ".tsv", ".tab", ".psv", ".txt", ".dat", ".log"}
TABLE_EXT = TEXT_DELIMITED | {".xlsx", ".xlsm", ".xltx", ".xls", ".xlsb", ".ods", ".json", ".jsonl", ".ndjson", ".parquet", ".pq", ".feather",
                              ".arrow", ".orc", ".h5", ".hdf5", ".xml", ".html", ".htm", ".dta", ".sas7bdat", ".xpt", ".sav", ".zsav",
                              ".sqlite", ".sqlite3", ".db", ".npy", ".npz", ".gz", ".bz2", ".xz", ".zip"}
REFUSED_EXT = {".pkl", ".pickle", ".joblib", ".pt", ".pth"}   # loading these can run arbitrary code
OPEN_FILTER = ("All files (*);;Data files (" + " ".join(f"*{e}" for e in sorted(TABLE_EXT)) + ")")
DELIMITERS = ",;\t|"


def read_head(path, size=65536) -> str:
    with open(path, "rb") as f:
        raw = f.read(size)
    for enc in ("utf-8-sig", "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def sniff_delimiter(path, default=",") -> str | None:
    """The column separator, or None when the text has no consistent one (plain prose)."""
    lines = [l for l in read_head(path).splitlines() if l.strip()][:40]
    if len(lines) > 1:
        lines = lines[:-1]                      # the last line of the sample may be cut off
    if not lines:
        return default
    best, score = None, 0
    for d in DELIMITERS:
        widths = [len(r) for r in csv.reader(lines, delimiter=d)]
        if widths and widths[0] > 1 and sum(w == widths[0] for w in widths) / len(widths) >= 0.9 and widths[0] > score:
            best, score = d, widths[0]
    if best:
        return best
    suffix = Path(path).suffix.lower()
    return None if suffix in (".txt", ".log", ".dat") else default


def _media_kind(names) -> str | None:
    n = dict(images=0, audio=0, video=0)
    for f in names:
        s = Path(f).suffix.lower()
        n["images"] += s in IMAGE_EXT
        n["audio"] += s in AUDIO_EXT
        n["video"] += s in VIDEO_EXT
    kind = max(n, key=n.get)
    return kind if n[kind] else None


def _files(d: Path, limit=20000):
    out = []
    for f in d.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            out.append(f)
            if len(out) >= limit:
                break
    return out


def coco_files(root: Path) -> list[Path]:
    found = []
    for f in list(root.glob("*.json")) + list(root.glob("annotations/*.json")) + list(root.glob("*/_annotations.coco.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                head = fh.read(2000000)
        except (OSError, UnicodeDecodeError):
            continue
        if '"annotations"' in head and '"images"' in head:
            found.append(f)
    return found


def yolo_names(root: Path) -> list[str] | None:
    """Class names from data.yaml, read without a YAML library."""
    for f in (root / "data.yaml", root / "dataset.yaml"):
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        m = re.search(r"^names:\s*\n((?:[ \t]+.+\n?)+)", text, re.M)
        if m:
            rows = [re.sub(r"^\s*(?:-|\d+\s*:)\s*", "", l).strip().strip("'\"") for l in m.group(1).splitlines() if l.strip()]
            return rows or None
    return None


def label_path(image: Path) -> Path:
    """YOLO convention: the label file sits in a sibling 'labels' tree."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def detect_folder(root) -> dict:
    """Work out what kind of dataset a folder holds."""
    root = Path(root)
    if (root / "nodes.csv").exists() and (root / "edges.csv").exists():
        return dict(kind="graph")
    if coco := coco_files(root):
        return dict(kind="detection", format="coco", annotations=[str(c) for c in coco])
    has_yaml = (root / "data.yaml").exists() or (root / "dataset.yaml").exists()
    images = [f for f in _files(root) if f.suffix.lower() in IMAGE_EXT]
    labelled = sum(label_path(f).exists() for f in images[:300])
    if images and (has_yaml or labelled >= 0.5 * min(len(images), 300)) and any(d.name == "labels" for d in root.rglob("labels")):
        return dict(kind="detection", format="yolo", names=yolo_names(root))
    for image_dir, mask_dir in (("images", "masks"), ("images", "annotations"), ("img", "mask"), ("imgs", "masks")):
        if (root / image_dir).is_dir() and (root / mask_dir).is_dir():
            return dict(kind="segmentation", images=image_dir, masks=mask_dir)
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    classes = {}
    for d in subdirs:
        files = _files(d, 5000)
        kind = _media_kind(files)
        if kind:
            classes.setdefault(kind, []).append((d.name, sum(f.suffix.lower() in {"images": IMAGE_EXT, "audio": AUDIO_EXT, "video": VIDEO_EXT}[kind] for f in files)))
    if classes:
        kind = max(classes, key=lambda k: sum(n for _, n in classes[k]))
        return dict(kind=kind, classes=[dict(name=a, files=b) for a, b in classes[kind]])
    flat = [f for f in root.iterdir() if f.is_file()]
    kind = _media_kind(flat)
    if kind:
        return dict(kind=kind, classes=[], flat=len([f for f in flat if f.suffix.lower() in {"images": IMAGE_EXT, "audio": AUDIO_EXT, "video": VIDEO_EXT}[kind]]))
    return dict(kind="unknown")
