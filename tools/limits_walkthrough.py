#!/usr/bin/env python3
"""Drive the real window offscreen and prove the hardware controls work, with real training.

Moves the limit sliders during a run, presses the GPU accelerator during a run, and checks what the kernel
and nvidia-smi really report. Runs and settings go to a private folder; runtimes are shared with the app.

Usage: python3 tools/limits_walkthrough.py <out_dir> [cpu|ram|gpu|torch ...]
"""
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
WANTED = set(sys.argv[2:]) or {"cpu", "ram", "gpu", "torch"}
HOME = Path.home() / ".cache/atlas-limit-ui"
for name in ("runs", "settings.json"):
    target = HOME / name
    shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
HOME.mkdir(parents=True, exist_ok=True)
if not (HOME / "envs").exists():
    (HOME / "envs").symlink_to(Path.home() / ".local/share/atlas-engine/envs")
os.environ["ATLAS_HOME"] = str(HOME)

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from atlas_engine import theme as T
from atlas_engine.app import MainWindow


def table(name, rows, features=12):
    path = HOME / name
    if path.exists():
        return path
    r = np.random.RandomState(0)
    X = r.normal(size=(rows, features))
    y = X[:, 0] * 3 - X[:, 1] ** 2 + np.sin(X[:, 2] * 2) + r.normal(0, .3, rows)
    head = ",".join(f"f{i}" for i in range(features)) + ",target"
    np.savetxt(path, np.column_stack([X, y]), delimiter=",", header=head, comments="", fmt="%.5f")
    return path


app = QApplication([])
app.setFont(T.font(10))
app.setStyleSheet(T.STYLE)
w = MainWindow()
w.resize(1320, 840)
w.show()
w.confirm = lambda text: (print("  dialog:", text, flush=True), True)[1]
seen = []
w.governor.updated.connect(lambda s: seen.append((time.time(), s)))


def spin(cond=lambda: False, timeout=0.3):
    t0 = time.time()
    while time.time() - t0 < timeout and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def shot(name):
    spin(timeout=0.3)
    w.grab().save(str(OUT / f"{name}.png"))


def setup(data, entry=56, params=None, device="cpu", task="regression"):
    w.load_dataset(str(data))
    assert spin(lambda: w.profile is not None and w.profile.get("name") == Path(data).name and not w.runner.running, 60)
    w.task_box.setCurrentIndex(w.task_box.findData(task))
    w.target_box.setCurrentIndex(w.target_box.findData("target"))
    spin(timeout=0.2)
    assert w.ds_next.isEnabled(), w.ds_msg.text()
    w.goto(1)
    w.algo_view.setCurrentIndex(w.algo_view.findData("task"))
    w.algo_search.setText("")
    spin(timeout=0.2)
    t = w.algo_table
    row = next(i for i in range(t.rowCount()) if t.item(i, 0).data(Qt.ItemDataRole.UserRole) == entry)
    t.selectRow(row)
    t.setCurrentCell(row, 0)
    spin(timeout=0.2)
    w.use_algo()
    for k, v in (params or {}).items():
        w.param_widgets[k].setValue(v)
    w.device_box.setCurrentIndex(w.device_box.findData(device))
    spin(timeout=0.2)


def finish(timeout=1800):
    assert spin(lambda: not w.runner.running and not w.cancel_btn.isEnabled(), timeout), "timed out:\n" + w.term.toPlainText()[-800:]
    spin(timeout=0.3)
    assert w.to_results.isEnabled(), "run failed: " + w.train_msg.text() + "\n" + w.term.toPlainText()[-1200:]
    import json

    return json.loads((Path(w.run_dir) / "metrics.json").read_text())


def scope_file(name):
    return (w.governor.scope.path / name).read_text().strip()


def cpu_average(seconds):
    t0 = time.time()
    n = len(seen)
    spin(timeout=seconds)
    vals = [s["cpu_pct"] for t, s in seen[n:] if "cpu_pct" in s]
    return statistics.mean(vals) if vals else 0.0, len(vals), time.time() - t0


def wait_training():
    """Until the fit itself is running, so readings are not of file loading."""
    assert spin(lambda: w.governor.scope is not None or not w.runner.running, 300), "the job never reached its scope"
    assert spin(lambda: "fitting" in w.term.toPlainText() or not w.runner.running, 300)


def reset():
    for key, top in (("cpu", 100), ("ram", w.hw_config.ram_top), ("gpu", 100), ("vram", w.hw_config.vram_top)):
        w.hw_config.rows[key].slider.setValue(top)


assert spin(lambda: w.hw is not None, 20)
print("hardware:", w.hw_label.text(), "| limits available:", w.hw.limits_ok, w.hw.limits_why, flush=True)
assert w.hw.limits_ok, w.hw.limits_why
SMALL = table("small.csv", 60000)

if "cpu" in WANTED:
    print("\n=== CPU limit, changed during the run ===", flush=True)
    setup(SMALL, params=dict(n_estimators=2500, learning_rate=.03, max_depth=7))
    assert not w.hw_config.accel.isHidden() and w.hw_config.accel.isEnabled()
    w.hw_config.rows["cpu"].slider.setValue(25)
    w.hw_config.rows["ram"].slider.setValue(8)
    shot("1_configure_limits")
    assert w.hw_live.rows["cpu"].value() == 25 and w.limits.ram_gb == 4.0
    t0 = time.time()
    w.start_training()
    wait_training()
    expect = f"{int(.25 * os.cpu_count() * 100000)} 100000"
    assert scope_file("cpu.max") == expect, scope_file("cpu.max")
    assert scope_file("memory.high") == str(4 << 30) or int(scope_file("memory.high")) <= 4 << 30, scope_file("memory.high")
    avg, n, _ = cpu_average(8)
    print(f"limit 25%: measured {avg:.1f}% over {n} samples", flush=True)
    shot("2_training_at_25")
    w.hw_live.rows["cpu"].slider.setValue(10)
    spin(timeout=1.5)
    assert scope_file("cpu.max") == f"{int(.10 * os.cpu_count() * 100000)} 100000", scope_file("cpu.max")
    avg, n, _ = cpu_average(8)
    print(f"moved to 10% mid-run: measured {avg:.1f}% over {n} samples", flush=True)
    shot("3_training_at_10")
    assert w.hw_config.rows["cpu"].value() == 10, "the Configure page should follow"
    w.hw_live.rows["cpu"].slider.setValue(100)
    spin(timeout=1.5)
    assert scope_file("cpu.max").startswith("max"), scope_file("cpu.max")
    avg, n, _ = cpu_average(5)
    print(f"limit removed mid-run: measured {avg:.1f}% over {n} samples", flush=True)
    m = finish()
    print(f"completed in {time.time() - t0:.0f}s, device {m['device']}, test r2 {m['test']['r2']:.4f}", flush=True)
    reset()

if "ram" in WANTED:
    print("\n=== RAM limit far below what the job needs: it must finish, not die ===", flush=True)
    big = table("big.csv", 700000)
    setup(big, params=dict(n_estimators=120, learning_rate=.1, max_depth=6))
    t0 = time.time()
    w.start_training()
    m = finish()
    free_run, free_peak = time.time() - t0, m["peak_rss_mb"]
    print(f"no limit: {free_run:.0f}s, peak {free_peak} MB", flush=True)
    limit_steps = max(1, int((free_peak * .55) / 512))
    setup(big, params=dict(n_estimators=120, learning_rate=.1, max_depth=6))
    w.hw_config.rows["ram"].slider.setValue(limit_steps)
    print(f"limit set to {w.limits.ram_gb} GB", flush=True)
    t0 = time.time()
    w.start_training()
    wait_training()
    shot("4_ram_limited")
    m = finish(3000)
    high = sum(1 for _, s in seen if s.get("ram_slowed"))
    print(f"with limit: {time.time() - t0:.0f}s (was {free_run:.0f}s), peak {m['peak_rss_mb']} MB, slowed in {high} samples, test r2 {m['test']['r2']:.4f}", flush=True)
    reset()

if "ramx" in WANTED:
    print("\n=== RAM limit a quarter of what the job needs, on a machine with no swap ===", flush=True)
    huge = table("huge.csv", 2600000)
    setup(huge, params=dict(n_estimators=30, learning_rate=.1, max_depth=6))
    w.hw_config.rows["ram"].slider.setValue(1)
    print(f"limit set to {w.limits.ram_gb} GB", flush=True)
    t0 = time.time()
    n = len(seen)
    w.start_training()
    m = finish(3000)
    raised = max((s.get("ram_raised_to", 0) for _, s in seen[n:]), default=0)
    print(f"finished in {time.time() - t0:.0f}s, peak {m['peak_rss_mb']} MB, limit rose to {raised / 2**30:.2f} GB, test r2 {m['test']['r2']:.4f}", flush=True)
    reset()

if "gpu" in WANTED:
    print("\n=== GPU accelerator pressed during a CPU run (XGBoost) ===", flush=True)
    setup(SMALL, params=dict(n_estimators=3000, learning_rate=.03, max_depth=9))
    assert not w.hw_live.accel.isHidden()
    w.start_training()
    wait_training()
    assert w.planned_device == "cpu" and w.hw_live.accel.isEnabled()
    shot("5_cpu_run_accelerator_off")
    print("pressing GPU accelerator", flush=True)
    t0 = time.time()
    w.hw_live.accel.click()
    assert spin(lambda: w.planned_device == "cuda" and w.runner.running, 60), w.term.toPlainText()[-400:]
    assert spin(lambda: "fitting" in w.term.toPlainText() and w.governor.scope is not None or not w.runner.running, 1500), w.term.toPlainText()[-800:]
    print(f"on GPU after {time.time() - t0:.0f}s (includes any first-time download)", flush=True)
    assert w.runner.running, w.term.toPlainText()[-1000:]
    assert w.hw_live.accel.isChecked()
    spin(timeout=4)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("processes on the GPU:", apps or "(none)", flush=True)
    assert apps, "the app says GPU, but no process is on the GPU"
    shot("6_gpu_running")
    util = [s["gpu"]["util"] for _, s in seen[-12:] if s.get("gpu")]
    print("GPU use, no limit:", util, flush=True)
    w.hw_live.rows["gpu"].slider.setValue(30)
    n = len(seen)
    spin(timeout=20)
    util = [s["gpu"]["util"] for _, s in seen[n + 16:] if s.get("gpu")]
    mean = statistics.mean(util) if util else 0
    others = len(subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split())
    print(f"GPU limit 30%: {util} mean {mean:.0f}, duty {w.governor.duty:.2f}, processes on the GPU {others}", flush=True)
    assert others > 1 or mean <= 40, "the GPU limit did not hold with nothing else on the GPU"
    shot("7_gpu_limited")
    w.hw_live.rows["gpu"].slider.setValue(100)
    m = finish()
    print(f"completed on {m['device']}, test r2 {m['test']['r2']:.4f}, {m['train_seconds']}s", flush=True)
    assert m["device"] == "cuda"
    reset()

if "torch" in WANTED:
    print("\n=== A PyTorch model on the GPU, with a GPU memory limit ===", flush=True)
    series = HOME / "series.csv"
    if not series.exists():
        r = np.random.RandomState(1)
        t_ = np.arange(6000)
        np.savetxt(series, np.column_stack([t_, 10 * np.sin(t_ / 24) + r.normal(0, 1, 6000)]), delimiter=",", header="step,target", comments="", fmt="%.4f")
    w.load_dataset(str(series))
    assert spin(lambda: w.profile is not None and w.profile.get("name") == "series.csv" and not w.runner.running, 30)
    w.task_box.setCurrentIndex(w.task_box.findData("forecasting"))
    w.target_box.setCurrentIndex(w.target_box.findData("target"))
    w.time_box.setCurrentIndex(w.time_box.findData("step"))
    spin(timeout=.2)
    assert w.ds_next.isEnabled(), w.ds_msg.text()
    w.goto(1)
    w.algo_view.setCurrentIndex(w.algo_view.findData("task"))
    spin(timeout=.2)
    t = w.algo_table
    row = next(i for i in range(t.rowCount()) if t.item(i, 0).data(Qt.ItemDataRole.UserRole) == 344)
    t.selectRow(row)
    t.setCurrentCell(row, 0)
    spin(timeout=.2)
    w.use_algo()
    w.param_widgets["epochs"].setValue(40)
    w.device_box.setCurrentIndex(w.device_box.findData("gpu"))
    spin(timeout=.2)
    assert w.hw_config.accel.isChecked()
    assert not w.hw_config.rows["vram"].isHidden()
    w.hw_config.rows["vram"].slider.setValue(4)
    shot("8_torch_configure")
    t0 = time.time()
    w.start_training()
    m = finish()
    print(f"completed on {m['device']} in {time.time() - t0:.0f}s, {m['test']}", flush=True)
    assert m["device"] == "cuda"
    text = w.term.toPlainText()
    print("\n".join(l for l in text.splitlines() if "torch" in l or "cuda" in l or "limited" in l or "GPU" in l)[:900], flush=True)
    shot("9_torch_done")
    reset()

if "pause" in WANTED:
    print("\n=== Machine nearly out of memory: the job must pause, then carry on ===", flush=True)
    import atlas_engine.limits as L

    real = L._meminfo

    def starving():
        total, _, swap = real()
        return total, int(L.memory_floor(total) * .3), swap

    setup(table("big.csv", 700000), params=dict(n_estimators=400, learning_rate=.1, max_depth=6))
    w.start_training()
    wait_training()
    assert spin(lambda: seen and seen[-1][1].get("ram_bytes", 0) > 300 << 20, 60), "the job stayed too small to pause"
    L._meminfo = starving
    assert spin(lambda: w.governor.paused, 5), "did not pause"
    spin(timeout=1)
    assert scope_file("cgroup.freeze") == "1"
    avg, n, _ = cpu_average(3)
    print(f"paused: frozen, CPU {avg:.1f}%, status: {w.hw_live.status.text()}", flush=True)
    shot("10_paused_low_memory")
    L._meminfo = real
    assert spin(lambda: not w.governor.paused, 5), "did not resume"
    m = finish()
    print(f"resumed by itself and completed, test r2 {m['test']['r2']:.4f}", flush=True)
    print("--- Cancel while paused ---", flush=True)
    setup(table("big.csv", 700000), params=dict(n_estimators=400, learning_rate=.1, max_depth=6))
    w.start_training()
    wait_training()
    assert spin(lambda: seen and seen[-1][1].get("ram_bytes", 0) > 300 << 20, 60), "the job stayed too small to pause"
    L._meminfo = starving
    assert spin(lambda: w.governor.paused, 5), "did not pause"
    spin(timeout=1)
    w.cancel_btn.click()
    assert spin(lambda: not w.runner.running, 15), "Cancel did not stop a paused job"
    L._meminfo = real
    left = subprocess.run(["pgrep", "-f", "atlas_worker train"], capture_output=True, text=True).stdout.split()
    print(f"cancelled while paused, workers left: {len(left)}, message: {w.train_msg.text()}", flush=True)
    assert not left

if "oom" in WANTED:
    print("\n=== GPU out of memory: smaller batch, then CPU, never a dead run ===", flush=True)
    calls = []
    w._launch = lambda choice, note=None: calls.append((choice, note))
    w.extra_widgets["batch"] = w._spin(dict(type="int", min=0, max=4096, default=0))
    w.planned_device, w.last_batch, w.last_error = "cuda", 32, "OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"
    assert w._retry_after_gpu_memory() and calls[-1][0] == "gpu" and w.extra_widgets["batch"].value() == 16, calls
    print(" ", calls[-1], flush=True)
    w.last_batch, w.last_error = 1, "RuntimeError: CUDA out of memory"
    assert w._retry_after_gpu_memory() and calls[-1][0] == "cpu", calls
    print(" ", calls[-1], flush=True)
    w.last_error = "ValueError: something else"
    assert not w._retry_after_gpu_memory()
    print("  other errors are left alone", flush=True)

if "picker" in WANTED:
    print("\n=== Column picker and task guess in the real window ===", flush=True)
    import json as J

    r = np.random.RandomState(3)
    t_ = np.repeat(np.arange(300), 10)
    f = r.normal(size=3000)
    y = f * 2 + r.normal(0, 1, 3000)
    path = HOME / "repeats.csv"
    np.savetxt(path, np.column_stack([t_, f, f - y, y]), delimiter=",", header="when,a,bias,reading", comments="", fmt="%.5f")
    w.load_dataset(str(path))
    assert spin(lambda: w.profile is not None and w.profile.get("name") == path.name, 20)
    w.task_box.setCurrentIndex(w.task_box.findData("forecasting"))
    w.target_box.setCurrentIndex(w.target_box.findData("reading"))
    w.time_box.setCurrentIndex(w.time_box.findData("when"))
    spin(timeout=.2)
    assert not w.ds_next.isEnabled(), "forecasting mixed series must be refused"
    print("forecast refused in the window:", w.ds_msg.text(), flush=True)
    w.task_box.setCurrentIndex(w.task_box.findData("regression"))
    spin(timeout=.2)
    assert w.ds_next.isEnabled(), w.ds_msg.text()
    t = w.ds_table
    row = next(i for i in range(t.rowCount()) if t.item(i, 0).text() == "bias")
    t.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
    shot("11_column_picker")
    w.goto(1)
    w.algo_view.setCurrentIndex(w.algo_view.findData("task"))
    spin(timeout=.2)
    tb = w.algo_table
    r_ = next(i for i in range(tb.rowCount()) if tb.item(i, 0).data(Qt.ItemDataRole.UserRole) == 3)
    tb.selectRow(r_)
    tb.setCurrentCell(r_, 0)
    spin(timeout=.2)
    w.use_algo()
    w.device_box.setCurrentIndex(w.device_box.findData("cpu"))
    w.start_training()
    m = finish(300)
    job = J.loads((Path(w.run_dir) / "job.json").read_text())
    print("features sent:", job["dataset"].get("features"), "| test r2", round(m["test"]["r2"], 3), "| warnings", m["warnings"], flush=True)
    assert "bias" not in job["dataset"]["features"] and m["test"]["r2"] < .95
    shot("12_results_with_ranges")

if "gpuexport" in WANTED:
    print("\n=== Export a model trained on the GPU ===", flush=True)
    setup(SMALL, params=dict(n_estimators=200, learning_rate=.1, max_depth=6), device="gpu")
    w.start_training()
    m = finish(600)
    print("device", m["device"], "libraries", {k: v for k, v in m["libraries"].items() if k == "xgboost"}, flush=True)
    w.goto(6)
    out = HOME / "export_gpu"
    shutil.rmtree(out, ignore_errors=True)
    w.exp_dir.setText(str(out))
    w.run_export()
    assert spin(lambda: w.export_btn.isEnabled() and not w.runner.running, 240), w.exp_term.toPlainText()
    req = (out / "requirements.txt").read_text()
    print("export:", w.export_msg.text()[:120], "| pin:", [l for l in req.splitlines() if l.startswith("xgboost")], flush=True)
    assert m["libraries"]["xgboost"] in req

print("\nWALKTHROUGH PASSED", flush=True)
app.quit()
