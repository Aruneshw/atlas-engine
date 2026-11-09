#!/usr/bin/env python3
"""Drive the real main window offscreen through the whole workflow and save a screenshot of each page.

Uses real training processes. Usage: python3 tools/ui_walkthrough.py <out_dir>
"""
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
HOME = Path.home() / ".cache/atlas-ui"
shutil.rmtree(HOME / "runs", ignore_errors=True)
HOME.mkdir(parents=True, exist_ok=True)
(HOME / "envs").mkdir(exist_ok=True)
link = HOME / "envs/cpu"
if not link.exists():
    link.symlink_to(Path.home() / ".local/share/atlas-engine/envs/cpu")
os.environ["ATLAS_HOME"] = str(HOME)

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from atlas_engine import theme as T
from atlas_engine.app import MainWindow


def data():
    d = HOME / "data"
    d.mkdir(exist_ok=True)
    r = np.random.RandomState(1)
    n = 1500
    t = np.arange(n)
    temp = 60 + 8 * np.sin(t / 40) + r.normal(0, 1.5, n)
    vib = 2 + 0.4 * np.sin(t / 15) + r.normal(0, 0.2, n)
    pres = 30 - 0.02 * (temp - 60) ** 2 + r.normal(0, 0.5, n)
    spikes = (r.random_sample(n) < 0.08) * r.uniform(0.8, 1.6, n)
    vib = vib + spikes
    fail = (0.05 * (temp - 60) + 1.6 * spikes + r.normal(0, 0.25, n) > 1.0).astype(int)
    kinds = r.choice(["press", "lathe", "mill"], n)
    import csv
    with open(d / "machine_sensors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "temperature", "vibration", "pressure", "machine_type", "failure"])
        for i in range(n):
            ts = f"2026-{1 + i // 720:02d}-{1 + (i // 24) % 28:02d} {i % 24:02d}:00:00"
            w.writerow([ts, f"{temp[i]:.3f}", "" if i % 97 == 0 else f"{vib[i]:.3f}", f"{pres[i]:.3f}", kinds[i], fail[i]])
    with open(d / "new_readings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "vibration", "pressure", "machine_type"])
        for i in range(40):
            w.writerow([f"{temp[i] + 5:.3f}", f"{vib[i]:.3f}", f"{pres[i]:.3f}", kinds[i]])
    with open(d / "load.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["load"])
        for i in range(300):
            w.writerow([f"{100 + 20 * np.sin(i / 24 * 2 * np.pi) + 0.05 * i + r.normal(0, 2):.2f}"])
    return d


app = QApplication([])
app.setFont(T.font(10))
app.setStyleSheet(T.STYLE)
d = data()
w = MainWindow()
w.resize(1320, 840)
w.show()


def spin(cond=lambda: False, timeout=0.3):
    t0 = time.time()
    while time.time() - t0 < timeout and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def shot(name):
    spin(timeout=0.25)
    w.grab().save(str(OUT / f"{name}.png"))


def select(entry):
    w.goto(1)
    w.algo_view.setCurrentIndex(w.algo_view.findData("ready"))
    w.algo_search.setText("")
    spin(timeout=0.1)
    t = w.algo_table
    row = next(i for i in range(t.rowCount()) if t.item(i, 0).data(Qt.ItemDataRole.UserRole) == entry)
    t.selectRow(row)
    t.setCurrentCell(row, 0)
    spin(timeout=0.1)


def train(entry, tweak=None, timeout=420):
    select(entry)
    assert w.algo_use.isEnabled(), f"#{entry} not usable"
    w.use_algo()
    spin(lambda: w.hw is not None, 15)
    if tweak:
        tweak()
    w.start_training()
    assert spin(lambda: w.runner.running, 5) or w.to_results.isEnabled(), "did not start"
    done = spin(lambda: not w.runner.running and w.cancel_btn.isEnabled() is False, timeout)
    assert done, "training timed out"
    print(f"#{entry}: terminal end:\n" + "\n".join(w.term.toPlainText().splitlines()[-4:]))
    assert w.to_results.isEnabled(), f"#{entry} failed: {w.train_msg.text()}"


assert spin(lambda: w.hw is not None, 15)
print("hardware:", w.hw_label.text())
print("gpu state:", w.hw.gpu.state, "|", w.hw.gpu.detail)
shot("00_start")
w.load_dataset(str(d / "machine_sensors.csv"))
print("suggested:", w.task_box.currentData(), w.target_box.currentData(), w.time_box.currentData(), "|", w.ds_msg.text())
# Table classification on the sensor file, using the time column for an honest split.
w.task_box.setCurrentIndex(w.task_box.findData("classification"))
shot("01_dataset")
select(51)
shot("02_algorithm")
w.use_algo()
spin(lambda: w.hw is not None, 5)
shot("03_configure")
w.goto(1)
train(51, lambda: w.param_widgets["n_estimators"].setValue(60))
shot("04_train")
train(56, lambda: w.param_widgets["n_estimators"].setValue(60))
train(26)
w.goto(4)
shot("05_results")
assert w.runs_table.rowCount() == 3, w.runs_table.rowCount()

w.goto(5)
w.pred_input.setText(str(d / "new_readings.csv"))
w.run_predict()
assert spin(lambda: w.pred_table.rowCount() > 1 and not w.runner.running, 60), w.predict_msg.text() + w.pred_term.toPlainText()
print("predict:", w.predict_msg.text())
for i, v in enumerate(["71", "3.1", "29", "lathe"]):
    from PySide6.QtWidgets import QTableWidgetItem
    w.single.setItem(0, i, QTableWidgetItem(v))
w.run_single()
assert spin(lambda: w.pred_table.rowCount() == 1 and not w.runner.running, 60), w.pred_term.toPlainText()
print("single row columns:", [w.pred_table.horizontalHeaderItem(i).text() for i in range(w.pred_table.columnCount())][-3:])
shot("06_predict")

w.goto(6)
w.exp_dir.setText(str(HOME / "export_lr"))
shutil.rmtree(HOME / "export_lr", ignore_errors=True)
w.run_export()
assert spin(lambda: w.export_btn.isEnabled() and not w.runner.running, 90), w.exp_term.toPlainText()
print("export:", w.export_msg.text(), sorted(p.name for p in (HOME / "export_lr").iterdir()))
shot("07_export")

# Forecasting with a neural network.
w.load_dataset(str(d / "load.csv"))
w.task_box.setCurrentIndex(w.task_box.findData("forecasting"))
w.target_box.setCurrentIndex(w.target_box.findData("load"))
spin(timeout=0.1)
assert w.ds_next.isEnabled(), w.ds_msg.text()
train(344, lambda: (w.param_widgets["epochs"].setValue(8), w.param_widgets["seq_len"].setValue(24)))
w.goto(4)
shot("08_forecast")

# Parquet goes through the runtime, then trains like any table.
import subprocess
pq = d / "sensors.parquet"
subprocess.run([str(link / "bin/python"), "-c", f"import pandas as pd; pd.read_csv(r'{d}/machine_sensors.csv').to_parquet(r'{pq}')"], check=True)
w.load_dataset(str(pq))
assert spin(lambda: w.profile is not None and w.profile.get("name") == "sensors.parquet" and not w.runner.running, 60), w.term.toPlainText()[-400:]
print("parquet:", w.ds_name.text())
w.task_box.setCurrentIndex(w.task_box.findData("regression"))
w.target_box.setCurrentIndex(w.target_box.findData("pressure"))
train(17)

# Image classification from a folder.
from PySide6.QtGui import QColor, QImage
img = d / "images"
for cls, colour in (("ok", "#3a8f4f"), ("defect", "#a33f3f")):
    (img / cls).mkdir(parents=True, exist_ok=True)
    for i in range(14):
        q = QImage(48, 48, QImage.Format.Format_RGB32)
        q.fill(QColor(colour).lighter(95 + i))
        q.save(str(img / cls / f"{i}.png"))
w.load_dataset(str(img))
print("images:", w.ds_name.text(), w.task_box.currentData())
train(329, lambda: (w.param_widgets["epochs"].setValue(1), w.param_widgets["image_size"].setValue(64)))
w.goto(5)
w.pred_input.setText(str(img / "ok"))
w.run_predict()
assert spin(lambda: w.pred_table.rowCount() >= 2 and not w.runner.running, 90), w.predict_msg.text() + w.pred_term.toPlainText()[-400:]
print("image predict:", w.predict_msg.text())
w.goto(6)
shutil.rmtree(HOME / "export_img", ignore_errors=True)
w.exp_dir.setText(str(HOME / "export_img"))
w.run_export()
assert spin(lambda: w.export_btn.isEnabled() and not w.runner.running, 180), w.exp_term.toPlainText()
print("image export:", w.export_msg.text())
shot("09_image_export")

# File dialog: every connected drive must be in the sidebar.
from PySide6.QtWidgets import QFileDialog
import atlas_engine.app as A
seen = {}
def fake_exec(self):
    spin(timeout=0.2)
    seen["urls"] = [u.toLocalFile() for u in self.sidebarUrls()]
    self.grab().save(str(OUT / "10_file_dialog.png"))
    return 0
real_exec = QFileDialog.exec
QFileDialog.exec = fake_exec
A.choose_path(w, "Dataset file", "/media/aruneshwaran/ssd")
QFileDialog.exec = real_exec
print("dialog sidebar:", seen["urls"])
print("look in:", [w.look_in.itemText(i) for i in range(w.look_in.count())])
assert any("/media/aruneshwaran/ssd" == u.rstrip("/") for u in seen["urls"])

# A real Excel file, read from the external drive.
SSD = Path("/media/aruneshwaran/ssd/atlas-engine-test")
SSD.mkdir(exist_ok=True)
xlsx = SSD / "sensors_from_ssd.xlsx"
subprocess.run([str(link / "bin/python"), "-c", f"import pandas as pd; pd.read_csv(r'{d}/machine_sensors.csv').to_excel(r'{xlsx}', index=False)"], check=True)
w.load_dataset(str(xlsx))
assert spin(lambda: w.profile is not None and w.profile.get("name") == xlsx.name and not w.runner.running, 120), w.term.toPlainText()[-500:]
print("excel from ssd:", w.ds_name.text())
w.task_box.setCurrentIndex(w.task_box.findData("classification"))
w.target_box.setCurrentIndex(w.target_box.findData("failure"))
spin(timeout=0.1)
shot("11_excel_from_ssd")
train(35)
shutil.rmtree(SSD)

SMOKE = Path.home() / ".cache/atlas-smoke/data"
w.load_dataset(str(SMOKE / "text.csv"))
print("text:", w.task_box.currentData(), w.text_box.currentData(), w.target_box.currentData(), "|", w.ds_msg.text())
assert w.task_box.currentData() == "text_classification" and w.ds_next.isEnabled()
train(387)
w.load_dataset(str(SMOKE / "ratings.csv"))
print("ratings:", w.task_box.currentData(), w.user_box.currentData(), w.item_box.currentData(), w.target_box.currentData())
assert w.task_box.currentData() == "recommendation" and w.ds_next.isEnabled()
train(767, lambda: w.param_widgets["epochs"].setValue(3))
w.load_dataset(str(SMOKE / "yolo"))
print("detection:", w.ds_name.text(), w.task_box.currentData())
train(524, lambda: (w.param_widgets["epochs"].setValue(1), w.param_widgets["image_size"].setValue(128)), timeout=400)
w.goto(4)
shot("12_detection_results")
w.goto(5)
w.pred_input.setText(str(SMOKE / "yolo/images/0.jpg"))
w.run_predict()
assert spin(lambda: w.predict_btn.isEnabled() and not w.runner.running, 120), w.pred_term.toPlainText()[-400:]
print("yolo predict:", w.predict_msg.text() or w.pred_term.toPlainText()[-200:])
w.load_dataset(str(SMOKE / "seg"))
print("segmentation:", w.ds_name.text(), w.task_box.currentData())
train(543, lambda: (w.param_widgets["epochs"].setValue(1), w.param_widgets["image_size"].setValue(64)))
w.load_dataset(str(SMOKE / "audio"))
print("audio:", w.ds_name.text(), w.task_box.currentData())
train(659, lambda: w.param_widgets["epochs"].setValue(2))
w.load_dataset(str(SMOKE / "graph"))
print("graph:", w.ds_name.text(), w.task_box.currentData(), w.target_box.currentData())
train(753, lambda: w.param_widgets["epochs"].setValue(5))
w.set_profile(__import__("atlas_engine.profile", fromlist=["x"]).environment_profile("CartPole-v1"))
print("rl:", w.ds_name.text(), w.task_box.currentData())
select(711)
assert not w.algo_use.isEnabled(), "SAC must be blocked on a discrete environment"
train(707, lambda: w.param_widgets["timesteps"].setValue(1500))
w.goto(4)
shot("13_rl_results")
w.goto(5)
w.pred_steps.setValue(3)
w.run_predict()
assert spin(lambda: w.pred_table.rowCount() == 3 and not w.runner.running, 60), w.pred_term.toPlainText()[-300:]
w.goto(6)
shutil.rmtree(HOME / "export_rl", ignore_errors=True)
w.exp_dir.setText(str(HOME / "export_rl"))
w.run_export()
assert spin(lambda: w.export_btn.isEnabled() and not w.runner.running, 90), w.exp_term.toPlainText()
print("rl export:", w.export_msg.text())
w.load_dataset(str(SMOKE / "images"))
w.task_box.setCurrentIndex(w.task_box.findData("image_generation"))
spin(timeout=0.1)
train(615, lambda: w.param_widgets["epochs"].setValue(3))
w.goto(4)
shot("14_generated")

# Cancel really stops the worker and its children.
w.load_dataset(str(d / "load.csv"))
w.task_box.setCurrentIndex(w.task_box.findData("forecasting"))
spin(timeout=0.1)
select(344)
w.use_algo()
w.param_widgets["epochs"].setValue(2000)
w.param_widgets["hidden"].setValue(256)
w.start_training()
assert spin(lambda: "epoch 2/" in w.term.toPlainText(), 120), w.term.toPlainText()[-500:]
pid = w.runner.proc.processId()
w.runner.cancel()
assert spin(lambda: not w.runner.running, 10)
time.sleep(0.5)
alive = os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z"
print("cancelled:", w.train_msg.text(), "| process still alive:", alive)
assert not alive

# Bad data is explained rather than crashing.
bad = d / "bad.csv"
bad.write_text("a,b\n1,2\n3\n")
w.load_dataset(str(bad))
print("short file:", w.ds_name.text())
w.load_dataset(str(d / "nope.csv"))
print("missing file:", w.ds_msg.text())
print("WALKTHROUGH OK")
