"""Dataset profiling with the standard library, so a file can be checked before any runtime exists."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from atlas_worker.fileinfo import IMAGE_EXT, REFUSED_EXT, detect_folder, sniff_delimiter

SAMPLE_ROWS = 20000
NA = {"", "na", "n/a", "nan", "null", "none", "?"}
TARGET_HINTS = ("target", "label", "class", "failure", "fail", "fault", "y", "outcome", "churn", "price")
TIME_HINTS = ("time", "date", "timestamp", "datetime", "ds")
DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M")


class ProfileError(ValueError):
    pass


class NeedsConversion(Exception):
    """This file needs the runtime's readers (Excel, JSON, Parquet, odd encodings and so on)."""


TASKS_FOR_KIND = {
    "table": ["classification", "regression", "forecasting", "seq_classification", "anomaly", "clustering", "dimreduction", "text_classification", "text_generation", "recommendation"],
    "images": ["image_classification", "image_generation", "representation"],
    "detection": ["object_detection"], "segmentation": ["semantic_segmentation"], "audio": ["audio_classification"], "video": ["video_classification"],
    "graph": ["node_classification"], "environment": ["reinforcement_learning"],
}


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_date(s):
    for f in DATE_FORMATS:
        try:
            datetime.strptime(s, f)
            return True
        except ValueError:
            pass
    return False


def profile_table(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix.lower() in REFUSED_EXT:
        raise ProfileError(f"{p.suffix} files can run hidden code when opened, so they are refused. Export the data as CSV or Parquet.")
    if p.suffix.lower() not in (".csv", ".tsv", ".tab", ".psv", ".txt", ".dat", ".log"):
        raise NeedsConversion
    sep = sniff_delimiter(p)
    if sep is None:
        raise NeedsConversion                 # free-form text, one row per line
    try:
        with open(p, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=sep)
            header = next(reader, None)
            if not header or not any(h.strip() for h in header):
                raise ProfileError("Could not find a header row in this file. Is it a CSV?")
            if len(set(header)) != len(header):
                raise ProfileError("Two columns share the same name. Rename them first.")
            samples, rows, bad = [], 0, 0
            for row in reader:
                rows += 1
                if len(row) != len(header):
                    bad += 1
                elif len(samples) < SAMPLE_ROWS:
                    samples.append(row)
    except UnicodeDecodeError as e:
        raise NeedsConversion from e
    except csv.Error as e:
        raise ProfileError(f"This file is not valid CSV: {e}") from e
    if rows == 0:
        raise ProfileError("The file has a header but no rows.")
    columns = []
    for i, name in enumerate(header):
        values = [r[i].strip() for r in samples]
        present = [v for v in values if v.lower() not in NA]
        uniq = set(present)
        n = len(present) or 1
        probe = present[:300]
        if not present:
            kind = "empty"
        elif sum(map(_is_float, present)) >= 0.95 * n:
            kind = "numeric"
        elif probe and sum(map(_is_date, probe)) >= 0.95 * len(probe):
            kind = "datetime"
        elif len(uniq) <= 50:
            kind = "categorical"
        else:
            kind = "text"
        nums = [float(v) for v in present if _is_float(v)] if kind == "numeric" else []
        columns.append(dict(name=name, kind=kind, missing=1 - len(present) / max(len(values), 1), unique=len(uniq),
                            min=min(nums) if nums else None, max=max(nums) if nums else None,
                            example=present[0] if present else ""))
    return dict(kind="table", path=str(p), name=p.name, rows=rows, cols=len(header), columns=columns, malformed=bad,
                size_mb=p.stat().st_size / 1048576)


def profile_folder(path: str | Path) -> dict:
    """Recognise a dataset folder: class subfolders, YOLO, COCO, masks, or a graph."""
    p = Path(path)
    if not p.is_dir():
        raise ProfileError(f"Folder not found: {p}")
    info = detect_folder(p)
    kind = info["kind"]
    base = dict(kind=kind, path=str(p), name=p.name, rows=0, cols=0, columns=[], classes=[], malformed=0)
    if kind == "unknown":
        raise ProfileError("This folder is not a recognised dataset. Expected class subfolders of images, audio or video, images/ with labels/ or masks/, a COCO json, or nodes.csv with edges.csv.")
    if kind in ("images", "audio", "video"):
        classes = info.get("classes") or []
        base.update(classes=[dict(name=c["name"], images=c["files"]) for c in classes], rows=sum(c["files"] for c in classes) or info.get("flat", 0))
        if kind != "images" and len(classes) < 2:
            raise ProfileError(f"A {kind} dataset needs one subfolder per class, with at least two classes.")
    elif kind == "graph":
        nodes = profile_table(p / "nodes.csv")
        base.update(rows=nodes["rows"], cols=nodes["cols"], columns=nodes["columns"])
    elif kind == "detection":
        images = sum(1 for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        base.update(rows=images, format=info["format"], classes=[dict(name=n, images=0) for n in (info.get("names") or [])])
    elif kind == "segmentation":
        base.update(rows=sum(1 for f in (p / info["images"]).rglob("*") if f.suffix.lower() in IMAGE_EXT))
    return base


def environment_profile(name: str) -> dict:
    return dict(kind="environment", path=None, env=name, name=name, rows=0, cols=0, columns=[], classes=[], malformed=0)


def suggest(profile: dict) -> dict:
    """Best guess for target, time column and task. The user can change all of them."""
    kind = profile["kind"]
    if kind == "images":
        return dict(target=None, time=None, task="image_classification" if len(profile["classes"]) >= 2 else "image_generation")
    if kind == "graph":
        cols = profile["columns"]
        label = next((c["name"] for hint in TARGET_HINTS for c in cols if c["name"].lower() == hint), None)
        return dict(target=label or next((c["name"] for c in cols if c["kind"] == "categorical"), None), time=None, task="node_classification")
    if kind != "table":
        return dict(target=None, time=None, task=TASKS_FOR_KIND[kind][0])
    cols = profile["columns"]
    usable = [c for c in cols if c["kind"] in ("numeric", "categorical") and c["unique"] > 1]
    target = next((c for hint in TARGET_HINTS for c in usable if c["name"].lower() == hint), None)
    if target is None:
        target = next((c for c in usable for hint in TARGET_HINTS if hint in c["name"].lower() and len(hint) > 2), None)
    words = next((c for c in cols if c["kind"] == "text"), None)
    names = {c["name"].lower(): c["name"] for c in cols}
    if len(cols) == 1 and words:
        return dict(target=None, time=None, task="text_generation", text=words["name"])
    if {"user", "item"} <= set(names) or {"user_id", "item_id"} <= set(names):
        u, i = ("user", "item") if "user" in names else ("user_id", "item_id")
        rating = next((names[k] for k in ("rating", "score", "stars") if k in names), None)
        return dict(target=rating, time=None, task="recommendation", user=names[u], item=names[i])
    if words and any(c["kind"] == "categorical" and c["unique"] > 1 for c in cols):
        label = next((c for hint in TARGET_HINTS for c in cols if c["name"].lower() == hint and c["kind"] != "text"), None) or next(c for c in cols if c["kind"] == "categorical" and c["unique"] > 1)
        return dict(target=label["name"], time=None, task="text_classification", text=words["name"])
    if len(cols) == 1 and usable:
        return dict(target=usable[0]["name"], time=None, task="forecasting")     # one column can only be a series
    time = next((c for c in cols if c["kind"] == "datetime"), None)
    if time is None:
        time = next((c for c in cols if c["name"].lower() in TIME_HINTS and c["kind"] == "numeric"), None)
    if target is None:
        task = "forecasting" if time else "clustering"
        if time and usable:
            target = next((c for c in usable if c["kind"] == "numeric"), None)
    elif target["kind"] == "categorical" or target["unique"] <= 20:
        task = "classification"       # a time column still gives a time-ordered split here
    else:
        task = "forecasting" if time else "regression"
    return dict(target=target["name"] if target else None, time=time["name"] if time else None, task=task)


def task_requirements(task: str) -> tuple[bool, bool]:
    """(needs a target column, needs a time column)."""
    return (task in ("classification", "regression", "forecasting", "seq_classification", "text_classification", "recommendation", "node_classification"),
            task in ("forecasting", "seq_classification"))


def compatibility(algo: dict, task: str, profile: dict) -> tuple[str, list]:
    """state is ok, warn or blocked, with human reasons."""
    reasons = []
    if task not in algo["tasks"]:
        return "blocked", ["Does not support this task."]
    impl = algo["tasks"][task]
    env = profile.get("env")
    if env:
        adapter = impl["adapter"]
        grid, bandit, continuous = env in ("FrozenLake-v1", "CliffWalking-v1", "Taxi-v3"), env == "Bandit-10", env in ("Pendulum-v1", "MountainCarContinuous-v0")
        if adapter == "bandit" and not bandit:
            return "blocked", ["Bandit methods need the Bandit-10 environment."]
        if adapter == "rl_table" and not grid:
            return "blocked", ["Tabular methods need FrozenLake, CliffWalking or Taxi."]
        if adapter == "sb3" and bandit:
            return "blocked", ["Deep RL needs a control or grid environment."]
        if adapter == "sb3" and impl["actions"] == "discrete" and continuous:
            return "blocked", ["Needs discrete actions; this environment has continuous ones."]
        if adapter == "sb3" and impl["actions"] == "continuous" and not continuous:
            return "blocked", ["Needs continuous actions: try Pendulum."]
    rows = profile.get("rows", 0)
    if impl.get("target") == "nonnegative" and profile.get("target_min") is not None and profile["target_min"] < 0:
        return "blocked", ["Needs a non-negative target; this one has negative values."]
    limit = algo.get("max_rows")
    if limit and rows > limit:
        reasons.append(f"Above the practical limit of {limit:,} rows.")
    if "torch" in algo["groups"] and rows and rows < 200:
        reasons.append("Neural networks rarely learn well from under 200 rows.")
    if task in ("clustering", "dimreduction") and impl.get("inductive") is False:
        reasons.append("Labels this data only: cannot score new rows or be exported.")
    return ("warn" if reasons else "ok"), reasons
