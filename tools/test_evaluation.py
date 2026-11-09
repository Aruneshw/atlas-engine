#!/usr/bin/env python3
"""Known-answer tests for Atlas's evaluation: each dataset is built so the right outcome is known in advance.

Usage: <runtime python> tools/test_evaluation.py <work_dir>
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from atlas_worker import audit
from atlas_worker.data import TimeFolds, split_indices, validate

WORK = Path(sys.argv[1])
WORK.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""), flush=True)


def train(name, csv, target, entry=3, task="regression", time=None, features=None, **extra):
    run = WORK / name
    job = dict(run_id=name, run_dir=str(run), entry=entry, name=name, task=task, dataset=dict(path=str(csv), target=target, time_column=time),
               params=None, split=dict(val=.15, test=.15, seed=42), device="cpu", threads=2, batch_size=64, **extra)
    if features:
        job["dataset"]["features"] = features
    path = WORK / f"{name}.json"
    path.write_text(json.dumps(job))
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "atlas_worker", "train", str(path)], env=ENV, capture_output=True, text=True)
    errors = [json.loads(l[2:]) for l in p.stdout.splitlines() if l.startswith('@@{"t": "error"')]
    metrics = json.loads((run / "metrics.json").read_text()) if p.returncode == 0 else None
    return p.returncode, metrics, errors


r = np.random.RandomState(0)

# 1. Many rows per timestamp: no timestamp may land on two sides.
n, per = 6000, 40
panel = pd.DataFrame({"t": np.repeat(pd.date_range("2024-01-01", periods=n // per, freq="D"), per), "a": r.normal(size=n), "b": r.normal(size=n)})
panel["y"] = panel.a * 2 + r.normal(0, .1, n)
job = dict(dataset=dict(target="y", time_column="t"), task="regression", split=dict(val=.15, test=.15))
df, feats = validate(panel, job)
tr, va, te, chrono = split_indices(len(df), df.y.to_numpy(), job)
t = job["_times"]
check("a timestamp is never in two splits", chrono and t[tr].max() < t[va].min() and t[va].max() < t[te].min(), f"{len(tr)}/{len(va)}/{len(te)} rows")
check("no row is lost by the split", len(tr) + len(va) + len(te) == len(df))
folds = list(TimeFolds(4, t[tr]).split(tr))
check("cross-validation folds keep time apart", len(folds) == 4 and all(t[tr][a].max() < t[tr][b].min() for a, b in folds))

# 2. Leak detection on data with planted leaks, and silence on data with none.
base = pd.DataFrame({"f": r.normal(size=3000), "g": r.normal(size=3000), "h": r.normal(size=3000)})
base["y"] = base.f + base.g ** 2 + r.normal(0, .5, 3000)
feats = [{"name": c, "type": "num"} for c in ("f", "g", "h")]
check("no false alarm on honest columns", audit.find_leaks(base[["f", "g", "h"]], base.y.to_numpy(), feats, "regression") == [])
leak = base.assign(bias=base.f - base.y)
out = audit.find_leaks(leak[["f", "g", "h", "bias"]], leak.y.to_numpy(), feats + [{"name": "bias", "type": "num"}], "regression")
check("finds a pair that rebuilds the target", any("'f'" in s and "'bias'" in s for s in out), str(out)[:100])
copy = base.assign(c=base.y * 1.01 + 3)
out = audit.find_leaks(copy[["f", "c"]], copy.y.to_numpy(), [{"name": "f", "type": "num"}, {"name": "c", "type": "num"}], "regression")
check("finds a disguised copy of the target", any("'c'" in s for s in out), str(out)[:100])
cat = pd.DataFrame({"k": ["a", "b"] * 500, "n": r.normal(size=1000)})
out = audit.find_leaks(cat, (cat.k == "a").astype(int).to_numpy(), [{"name": "k", "type": "cat"}, {"name": "n", "type": "num"}], "classification")
check("finds a category that decides the class", any("'k'" in s for s in out), str(out)[:100])

# 3. Split problems.
X = pd.DataFrame({"x": np.arange(100.0), "k": ["a"] * 80 + ["new"] * 20})
out = audit.split_problems(X, [{"name": "x", "type": "num"}, {"name": "k", "type": "cat"}], np.arange(80), np.arange(80, 100))
check("warns about a category training never saw", any("'new'" in s for s in out), str(out)[:100])
Y = pd.DataFrame({"x": np.r_[np.arange(80.0), np.arange(20.0)]})
out = audit.split_problems(Y, [{"name": "x", "type": "num"}], np.arange(80), np.arange(80, 100))
check("warns when test rows repeat training rows", any("also appear" in s for s in out), str(out)[:100])

# 4. Baseline and interval arithmetic against values worked out by hand.
y_tr, y_te = np.zeros(100), np.r_[np.ones(50), -np.ones(50)]
ex = audit.regression_extras(y_te.copy(), y_tr, y_te)
check("baseline MAE is the error of guessing the training mean", ex["baseline_mae"] == 1.0, str(ex["baseline_mae"]))
noisy = y_te + r.normal(0, .5, 100)
ex = audit.regression_extras(noisy, y_tr, y_te)
true = float(np.mean(np.abs(noisy - y_te)))
check("the range contains the measured error", ex["mae_low"] < true < ex["mae_high"], f"{ex['mae_low']:.3f} < {true:.3f} < {ex['mae_high']:.3f}")
ex = audit.classification_extras(np.ones(100, int), np.r_[np.ones(90, int), np.zeros(10, int)], np.r_[np.ones(70, int), np.zeros(30, int)])
check("majority-class baseline", abs(ex["baseline_accuracy"] - .7) < 1e-9, str(ex["baseline_accuracy"]))

# 5. Through the real worker.
single = WORK / "series.csv"
pd.DataFrame({"t": pd.date_range("2024-01-01", periods=400, freq="h"), "y": 10 * np.sin(np.arange(400) / 12) + r.normal(0, .5, 400)}).to_csv(single, index=False)
code, m, err = train("single_series", single, "y", entry=166, task="forecasting", time="t")
check("a real single series can be forecast", code == 0 and m["test"]["mae"] > 0, str(err)[:120])
many = WORK / "panel.csv"
panel.to_csv(many, index=False)
code, m, err = train("panel_forecast", many, "y", entry=166, task="forecasting", time="t")
check("forecasting mixed series is refused", code == 2 and err and "many series" in err[0]["msg"], (err[0]["msg"][:80] if err else str(code)))

clean = WORK / "clean.csv"
base.assign(t=pd.date_range("2024-01-01", periods=3000, freq="h")).to_csv(clean, index=False)
code, honest, _ = train("honest", clean, "y", entry=51, time="t", features=["f", "g", "h"])
check("honest run has no leak warning", code == 0 and not any("leak" in w.lower() for w in (honest["warnings"] or [])), str(honest["warnings"])[:120])
check("honest score is sensible, not perfect", .5 < honest["test"]["r2"] < .99, f"r2 {honest['test']['r2']:.3f}")
check("score sits inside its own range", honest["test"]["r2_low"] < honest["test"]["r2"] < honest["test"]["r2_high"], f"{honest['test']['r2_low']:.3f} to {honest['test']['r2_high']:.3f}")
check("beats the baseline", honest["test"]["mae"] < honest["test"]["baseline_mae"], f"{honest['test']['mae']:.3f} vs {honest['test']['baseline_mae']:.3f}")
dirty = WORK / "dirty.csv"
leak.assign(t=pd.date_range("2024-01-01", periods=3000, freq="h")).to_csv(dirty, index=False)
code, m, _ = train("leaky", dirty, "y", entry=3, time="t")
check("a leaky run names the columns and warns the score is too high", code == 0 and any("bias" in w for w in m["warnings"]) and any("unusually high" in w for w in m["warnings"]), str(m["warnings"])[:140])
code, m, _ = train("column_picker", dirty, "y", entry=3, time="t", features=["f", "g", "h"])
check("unticked columns are really left out", code == 0 and m["test"]["r2"] < .99 and not any("bias" in w for w in (m["warnings"] or [])), f"r2 {m['test']['r2']:.3f}")
code, m, _ = train("tuned_cv", clean, "y", entry=55, time="t", features=["f", "g", "h"], cv=3, tune=3)
check("tuning and time-ordered folds still run", code == 0, str(m and m.get("tuned")))
noise = WORK / "noise.csv"
pd.DataFrame({"a": r.normal(size=2000), "b": r.normal(size=2000), "y": r.normal(size=2000)}).to_csv(noise, index=False)
code, m, _ = train("noise", noise, "y", entry=3)
check("pure noise is reported as having no skill", code == 0 and abs(m["test"]["r2"]) < .05 and any("no better" in w for w in (m["warnings"] or [])) or (code == 0 and m["test"]["mae"] >= m["test"]["baseline_mae"] * .98), f"r2 {m['test']['r2']:.3f}")

# 6. An exported model must still serve predictions.
out = WORK / "export_honest"
p = subprocess.run([sys.executable, "-W", "ignore", "-m", "atlas_worker", "export", str(WORK / "honest"), str(out), "onnx,api"], env=ENV, capture_output=True, text=True)
report = json.loads((out / "export_report.json").read_text()) if (out / "export_report.json").exists() else {}
check("export succeeds and ONNX matches", p.returncode == 0 and report.get("onnx", {}).get("status") == "ok", str(report.get("onnx"))[:90])
check("the bundle includes the audit module", (out / "atlas_worker/audit.py").exists())
env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
server = subprocess.Popen([sys.executable, "serve.py"], cwd=out, env={**env, "PORT": "8767"}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
try:
    answer = None
    for _ in range(120):
        try:
            req = urllib.request.Request("http://127.0.0.1:8767/predict", data=json.dumps({"rows": [{"f": .1, "g": .2, "h": .3}]}).encode(), headers={"Content-Type": "application/json"})
            answer = json.loads(urllib.request.urlopen(req, timeout=5).read())
            break
        except OSError:
            if server.poll() is not None:
                break
            import time

            time.sleep(.5)
    check("the exported web API answers", bool(answer and "predictions" in answer), str(answer)[:90] if answer else (server.stderr.read()[-300:] if server.poll() is not None else "no answer"))
finally:
    server.kill()

print(f"\n{sum(results)}/{len(results)} passed", flush=True)
sys.exit(0 if all(results) else 1)
