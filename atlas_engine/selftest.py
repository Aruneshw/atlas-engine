"""Built-in checks for a packaged build. `--selftest` is quick; `--selftest-train` trains and exports for real."""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from . import paths


def _report(**kw):
    """Windowed Windows builds have no console, so the result also goes to a file."""
    text = "SELFTEST " + json.dumps(kw)
    print(text, flush=True)
    paths.data_dir().mkdir(parents=True, exist_ok=True)
    (paths.data_dir() / "selftest_result.txt").write_text(text + "\n")


def quick(app, window):
    def finish():
        checks = dict(platform=app.platformName(), entries=len(window.entries), uv=paths.uv_path(),
                      worker=(paths.worker_root() / "atlas_worker" / "__main__.py").exists(),
                      gpu=window.hw.gpu.state if window.hw else None, ram_gb=round(window.hw.ram_total / 1024 ** 3) if window.hw else None)
        ok = checks["entries"] == 830 and bool(checks["uv"]) and checks["worker"] and checks["gpu"] is not None
        _report(ok=ok, **checks)
        app.exit(0 if ok else 1)

    QTimer.singleShot(4000, finish)


def full(app, w):
    """Generate data, then run real setup, training, prediction and export through the window."""
    work = paths.data_dir() / "selftest"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    data = work / "data.csv"
    rng = random.Random(0)
    lines = ["a,b,kind,label"]
    for _ in range(200):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        lines.append(f"{a:.4f},{b:.4f},{rng.choice('xyz')},{'on' if a + b > 0 else 'off'}")
    data.write_text("\n".join(lines) + "\n")
    state = dict(phase="wait_hw", t0=time.time())
    out = work / "export"

    def fail(why):
        _report(ok=False, why=why, terminal=w.term.toPlainText()[-1500:], export_log=w.exp_term.toPlainText()[-800:])
        app.exit(1)

    def tick():
        if time.time() - state["t0"] > 900:
            return fail(f"timeout in {state['phase']}")
        p = state["phase"]
        if p == "wait_hw" and w.hw is not None:
            w.load_dataset(str(data))
            w.task_box.setCurrentIndex(w.task_box.findData("classification"))
            w.target_box.setCurrentIndex(w.target_box.findData("label"))
            w.selected_algo = 26
            w.use_algo()
            w.hw_config.rows["cpu"].slider.setValue(50)     # prove a limited run works from a packaged build
            w.save_limits.stop()                            # and leave the user's saved limits alone
            w.start_training()
            state["phase"] = "training"
        elif p == "training" and w.runner.running and w.governor.scope is not None:
            state["limited"] = True
        elif p == "training" and not w.runner.running:
            if not w.to_results.isEnabled():
                return fail(w.train_msg.text())
            if w.hw.limits_ok and not state.get("limited"):
                return fail("limits are available here, but the job never ran under them")
            w.goto(6)
            w.exp_dir.setText(str(out))
            w.run_export()
            state["phase"] = "exporting"
        elif p == "exporting" and not w.runner.running:
            report = json.loads((out / "export_report.json").read_text()) if (out / "export_report.json").exists() else None
            m = json.loads((Path(w.run_dir) / "metrics.json").read_text())
            ok = bool(report) and report["onnx"]["status"] == "ok" and report["api"]["status"] == "ok"
            _report(ok=ok, algorithm=m["name"], metric=m["primary"], limits=dict(available=w.hw.limits_ok, applied=bool(state.get("limited"))), onnx=report and report["onnx"], files=sorted(os.listdir(out)))
            app.exit(0 if ok else 1)

    t = QTimer(app)
    t.timeout.connect(tick)
    t.start(200)
