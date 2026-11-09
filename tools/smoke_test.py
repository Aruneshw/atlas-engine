#!/usr/bin/env python3
"""Train, predict and export every registered algorithm on tiny data, each in its own forked process.

Writes atlas_engine/data/verified.json. The UI reads that file, so what it claims about an
algorithm is what actually passed here.

Usage: <env>/bin/python tools/smoke_test.py [--resume] [work_dir] [entry ...]
"""
import contextlib
import io
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")

import numpy as np
import pandas as pd
from PIL import Image

from atlas_worker import registry
from atlas_worker.data import load_table
from atlas_worker import __main__ as worker

RESUME = "--resume" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--resume"]
WORK = Path(ARGS[0] if ARGS else Path.home() / ".cache/atlas-smoke")
ONLY = {int(a) for a in ARGS[1:]}
OUT = ROOT / "atlas_engine/data/verified.json"
TIMEOUT = 240
HF_TINY = {369: "hf-internal-testing/tiny-random-bert", 370: "hf-internal-testing/tiny-random-roberta", 371: "hf-internal-testing/tiny-random-distilbert",
           372: "hf-internal-testing/tiny-random-ModernBertForSequenceClassification", 374: "hf-internal-testing/tiny-random-bart", 384: "hf-internal-testing/tiny-random-bert"}
LLM_TINY = "hf-internal-testing/tiny-random-gpt2"
TINY = dict(max_features=1000, max_length=16, block_size=16, rank=2, dim=8, neighbours=5, timesteps=300, steps=200, episodes=30, max_steps=50, sample_rate=8000,
            seconds=1.0, n_mels=32, frames=8, epochs=1, n_estimators=10, iterations=10, max_iter=20, hidden=8, seq_len=8, seasonal_periods=7, image_size=64,
            window=5, lags=3, n_components=2, n_clusters=2, perplexity=5.0, n_neighbors=3, min_samples=3, min_cluster_size=3, degree=2,
            n_knots=4, layers=2, p=1, d=1, q=1)


def datasets():
    d = WORK / "data"
    d.mkdir(parents=True, exist_ok=True)
    r = np.random.RandomState(0)
    n = 180
    X = r.normal(size=(n, 4))
    cat = r.choice(["a", "b", "c"], n)
    label = (X[:, 0] + X[:, 1] + (cat == "a") * 1.0 + r.normal(0, 0.5, n) > 0).astype(int)
    t = pd.DataFrame(X, columns=list("wxyz")).assign(kind=cat)
    t.loc[3, "w"] = np.nan
    t.loc[5, "kind"] = np.nan
    t.assign(target=np.where(label, "yes", "no")).to_csv(d / "clf.csv", index=False)
    t.assign(target=np.exp(0.3 * X[:, 0]) + 2 + r.normal(0, 0.1, n)).to_csv(d / "reg.csv", index=False)
    t.drop(columns="kind").dropna().assign().to_csv(d / "unsup.csv", index=False)
    t2 = t.drop(columns="kind").fillna(0)
    t2.assign(target=(np.abs(X).max(1) > 2.2).astype(int)).to_csv(d / "anom.csv", index=False)
    s = 10 + np.sin(np.arange(220) / 7 * 2 * np.pi) * 3 + np.arange(220) * 0.02 + r.normal(0, 0.2, 220)
    pd.DataFrame({"t": pd.date_range("2024-01-01", periods=220, freq="D").astype(str), "value": s}).to_csv(d / "series.csv", index=False)
    m = 400
    a = np.sin(np.arange(m) / 5) + r.normal(0, 0.2, m)
    b = np.cos(np.arange(m) / 9) + r.normal(0, 0.2, m)
    fail = ((np.roll(a, 2) > 0.8).astype(int))
    pd.DataFrame({"t": np.arange(m) * 2, "a": a, "b": b, "target": fail}).to_csv(d / "sensor.csv", index=False)
    img = d / "images"
    for c, colour in (("red", (200, 40, 40)), ("blue", (40, 40, 200))):
        (img / c).mkdir(parents=True, exist_ok=True)
        for i in range(12):
            arr = np.clip(np.array(colour) + r.normal(0, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(img / c / f"{i}.png")
    extra_datasets(d, r)
    return d


def extra_datasets(d, r):
    import cv2
    import soundfile as sf

    words = {"good": "great lovely wonderful happy fine excellent".split(), "bad": "awful terrible poor sad broken horrible".split()}
    rows = [(" ".join(r.choice(words[k], 8)), k) for k in words for _ in range(70)]
    pd.DataFrame(rows, columns=["review", "label"]).sample(frac=1, random_state=0).to_csv(d / "text.csv", index=False)
    (d / "prose.txt").write_text("\n".join(" ".join(r.choice(words["good"] + words["bad"], 12)) for _ in range(30)) + "\n")

    yolo = d / "yolo"
    shutil.rmtree(yolo, ignore_errors=True)
    (yolo / "images").mkdir(parents=True)
    (yolo / "labels").mkdir()
    (yolo / "data.yaml").write_text("names: [square, bar]\n")
    for i in range(30):
        im = np.full((128, 128, 3), 30, np.uint8)
        c, x, y = i % 2, r.randint(30, 98), r.randint(30, 98)
        w, h = (40, 40) if c == 0 else (60, 20)
        im[y - h // 2:y + h // 2, x - w // 2:x + w // 2] = (200, 220, 90)
        cv2.imwrite(str(yolo / "images" / f"{i}.jpg"), im)
        (yolo / "labels" / f"{i}.txt").write_text(f"{c} {x / 128} {y / 128} {w / 128} {h / 128}\n")

    seg = d / "seg"
    shutil.rmtree(seg, ignore_errors=True)
    (seg / "images").mkdir(parents=True)
    (seg / "masks").mkdir()
    for i in range(16):
        im = (r.random_sample((64, 64, 3)) * 60).astype(np.uint8)
        m = np.zeros((64, 64), np.uint8)
        x, y = r.randint(10, 40, 2)
        m[y:y + 20, x:x + 20] = 255
        im[m > 0] = (200, 120, 50)
        Image.fromarray(im).save(seg / "images" / f"{i}.png")
        Image.fromarray(m).save(seg / "masks" / f"{i}.png")

    audio = d / "audio"
    shutil.rmtree(audio, ignore_errors=True)
    for name, hz in (("low", 300), ("high", 1800)):
        (audio / name).mkdir(parents=True)
        for i in range(10):
            t = np.arange(8000) / 8000
            sf.write(audio / name / f"{i}.wav", np.sin(2 * np.pi * hz * t) * 0.4 + r.normal(0, 0.03, 8000), 8000)

    video = d / "video"
    shutil.rmtree(video, ignore_errors=True)
    for name, colour in (("left", 0), ("right", 1)):
        (video / name).mkdir(parents=True)
        for i in range(8):
            w = cv2.VideoWriter(str(video / name / f"{i}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
            for f in range(16):
                im = np.zeros((64, 64, 3), np.uint8)
                x = 4 + 3 * f if colour else 56 - 3 * f
                im[28:36, x:x + 8] = (240, 240, 240)
                w.write(im)
            w.release()

    graph = d / "graph"
    graph.mkdir(exist_ok=True)
    n = 80
    group = (np.arange(n) >= 40).astype(int)
    feats = r.normal(0, 1, (n, 3)) + group[:, None]
    pd.DataFrame(feats, columns=["f1", "f2", "f3"]).assign(id=np.arange(n), label=np.where(group, "b", "a")).to_csv(graph / "nodes.csv", index=False)
    pairs = [(a, b) for a in range(n) for b in r.choice(n, 4) if group[a] == group[b]]
    pd.DataFrame(pairs, columns=["source", "target"]).to_csv(graph / "edges.csv", index=False)

    rows = [(f"u{u}", f"i{i}", float(np.clip(3 + (u % 3 - 1) * (i % 3 - 1) + r.normal(0, 0.4), 1, 5))) for u in range(40) for i in r.choice(30, 12, replace=False)]
    pd.DataFrame(rows, columns=["user", "item", "rating"]).to_csv(d / "ratings.csv", index=False)


def job_for(entry, task, algo, d, run):
    file, target, time_col = {"classification": ("clf.csv", "target", None), "regression": ("reg.csv", "target", None),
                              "clustering": ("unsup.csv", None, None), "dimreduction": ("unsup.csv", None, None),
                              "anomaly": ("anom.csv", "target", None), "forecasting": ("series.csv", "value", "t"),
                              "seq_classification": ("sensor.csv", "target", "t"),
                              "image_classification": ("images", None, None), "text_classification": ("text.csv", "label", None), "text_generation": ("prose.txt", None, None),
                              "object_detection": ("yolo", None, None), "semantic_segmentation": ("seg", None, None), "audio_classification": ("audio", None, None),
                              "video_classification": ("video", None, None), "image_generation": ("images", None, None), "representation": ("images", None, None),
                              "node_classification": ("graph", "label", None), "recommendation": ("ratings.csv", "rating", None), "reinforcement_learning": (None, None, None)}[task]
    if algo["tasks"][task]["adapter"] == "seq" and task == "anomaly":
        file, time_col = "sensor.csv", "t"
    params = {p["name"]: TINY[p["name"]] for p in algo["params"] if p["name"] in TINY and p["type"] not in ("choice", "text") and p.get("min", 0) <= TINY[p["name"]] <= p.get("max", 1e9)}
    impl = algo["tasks"][task]
    if task == "object_detection":
        params["image_size"] = 128
    if impl["adapter"] == "hf_text":
        params["model"] = HF_TINY[entry]
    if impl["adapter"] == "llm":
        params["model"] = LLM_TINY
    if any(p["name"] == "image_size" and p["type"] == "choice" for p in algo["params"]):
        params["image_size"] = "32"
    if entry in (318, 320):
        params["image_size"] = 224       # these adaptive pools only export at sizes they divide evenly
    if entry == 548:
        params["image_size"] = 256
    dataset = dict(path=str(d / file) if file else None, target=target, time_column=time_col)
    if task in ("text_classification", "text_generation"):
        dataset["text_column"] = "review" if task == "text_classification" else "text"
    if task == "recommendation":
        dataset.update(user_column="user", item_column="item")
    if task == "reinforcement_learning":
        a = impl["adapter"]
        dataset["env"] = "Bandit-10" if a == "bandit" else "FrozenLake-v1" if a == "rl_table" else "Pendulum-v1" if impl.get("actions") == "continuous" else "CartPole-v1"
    return dict(run_id=run.name, run_dir=str(run), entry=entry, task=task, name=f"#{entry}",
                dataset=dataset, params=params,
                split=dict(val=0.2, test=0.2, seed=42), device="cpu", threads=2, batch_size=4, horizon=5, seq_len=8)


def check(entry, task, algo, d, run):
    """Runs inside the forked child. Returns the result dict."""
    job = job_for(entry, task, algo, d, run)
    run.mkdir(parents=True, exist_ok=True)
    (run.parent / f"{run.name}.json").write_text(json.dumps(job))
    res = {"train": False, "predict": None, "onnx": None, "api": None}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            worker.cmd_train(str(run.parent / f"{run.name}.json"))
        except SystemExit:
            pass
    events = [json.loads(l[2:]) for l in out.getvalue().splitlines() if l.startswith("@@")]
    err = next((e for e in events if e["t"] == "error"), None)
    if err or not any(e["t"] == "done" for e in events):
        traces = [e["msg"] for e in events if e["t"] == "log" and e.get("level") == "error"]
        res["error"] = (err or {}).get("msg", "no done event") + ("\n" + traces[-1] if traces else "")
        return res
    res["train"] = True
    summary = next(e for e in events if e["t"] == "done")["summary"]
    res["primary"] = summary["primary"]

    from atlas_worker.predictor import Predictor
    p = Predictor(run / "model")
    if p.can_predict:
        try:
            ds = Path(job["dataset"]["path"]) if job["dataset"]["path"] else None
            if p.needs in ("steps", "samples", "episodes"):
                got = p.predict(steps=3)
            elif p.needs == "prompt":
                got = p.predict(prompt="hello there", steps=5)
            elif p.needs == "graph":
                got = p.predict(path=ds)
            elif p.needs in ("images", "audio", "video"):
                from atlas_worker.fileinfo import AUDIO_EXT, IMAGE_EXT, VIDEO_EXT
                ext = {"images": IMAGE_EXT, "audio": AUDIO_EXT, "video": VIDEO_EXT}[p.needs]
                got = p.predict(images=[str(f) for f in sorted(ds.rglob("*")) if f.suffix.lower() in ext][:2], out_dir=str(run / "predict"))
            else:
                got = p.predict(df=load_table(ds).head(30))
            # A detector trained for one epoch may honestly find nothing, so an empty table is fine for it.
            res["predict"] = bool(p.kind.startswith("detect") or (len(got) > 0 and not got.isna().all(axis=None)))
        except Exception as e:
            res["predict"] = False
            res["predict_error"] = f"{type(e).__name__}: {e}"
    else:
        res["predict"] = "not_applicable"

    from atlas_worker.export import run_export
    with contextlib.redirect_stdout(io.StringIO()):
        rep = run_export(str(run), str(run / "export"), ["onnx", "api"])
    if rep.get("blocked"):
        res["onnx"] = res["api"] = "not_applicable"
    else:
        res["onnx"] = rep["onnx"]["status"]
        res["api"] = rep["api"]["status"]
        if rep["onnx"]["status"] != "ok":
            res["onnx_reason"] = (rep["onnx"].get("reason") or json.dumps(rep["onnx"].get("check")))[:200]
    return res


def run_one(entry, task, algo, d):
    run = WORK / "runs" / f"{entry}_{task}"
    shutil.rmtree(run, ignore_errors=True)
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            payload = check(entry, task, algo, d, run)
        except BaseException as e:
            import traceback
            payload = {"train": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}
        with os.fdopen(w_fd, "w") as f:
            json.dump(payload, f, default=str)
        os._exit(0)
    os.close(w_fd)
    t0 = time.time()
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.time() - t0 > TIMEOUT:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(r_fd)
            return {"train": False, "error": f"timed out after {TIMEOUT}s"}
        time.sleep(0.05)
    with os.fdopen(r_fd) as f:
        raw = f.read()
    if not raw:
        return {"train": False, "error": f"worker died, status {status}"}
    res = json.loads(raw)
    res["seconds"] = round(time.time() - t0, 1)
    shutil.rmtree(run, ignore_errors=True)
    return res


def main():
    d = datasets()
    results = json.loads(OUT.read_text()) if (ONLY or RESUME) and OUT.exists() else {}
    jobs = [(e, t, a) for e, a in sorted(registry.ALGOS.items()) for t in a["tasks"] if not ONLY or e in ONLY]
    if RESUME:
        jobs = [(e, t, a) for e, t, a in jobs if f"{e}:{t}" not in results]
    for i, (entry, task, algo) in enumerate(jobs, 1):
        r = run_one(entry, task, algo, d)
        results[f"{entry}:{task}"] = r
        flag = "ok  " if r.get("train") and r.get("predict") else "FAIL"
        print(f"[{i}/{len(jobs)}] {flag} #{entry} {task} {r.get('seconds', '')}s onnx={r.get('onnx')} api={r.get('api')} "
              f"{r.get('error', '')[:300]} {r.get('predict_error', '')[:200]}", flush=True)
        OUT.write_text(json.dumps(results, indent=1))
    good = sum(1 for r in results.values() if r.get("train") and r.get("predict"))
    onnx = sum(1 for r in results.values() if r.get("onnx") == "ok")
    print(f"DONE: {good}/{len(results)} train+predict ok, {onnx} verified ONNX")


if __name__ == "__main__":
    main()
