"""Atlas Engine main window: dataset, algorithm, configure, train, results, predict, export."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QStandardPaths, QStorageInfo, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from atlas_worker import registry
from atlas_worker.fileinfo import OPEN_FILTER
from atlas_worker.registry import ENVIRONMENTS, NEEDS, TASKS

from . import paths, profile as prof, resources, runtime
from . import theme as T
from .limits import Governor, Limits
from .limits_ui import HardwareControls
from .runner import Runner, Terminal
from .widgets import Chart, fmt, label, panel

STEPS = ["DATASET", "ALGORITHM", "CONFIGURE", "TRAIN", "RESULTS", "PREDICT", "EXPORT"]
STAGES = ["hardware", "runtime", "validate", "prepare", "build", "train", "evaluate", "save"]
STATUS_LABEL = {"ready": "TRAINABLE", "component": "COMPONENT", "tool": "TOOL", "planned": "REFERENCE"}
STATUS_COLOR = {"ready": T.ACCENT, "component": T.CYAN, "tool": T.VIOLET, "planned": T.FAINT}
GB = 1024 ** 3


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)], env=paths.worker_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def external_drives():
    """Mounted drives other than the system disk, as (label, root path, bytes)."""
    system = bytes(QStorageInfo.root().device())
    out = []
    for v in QStorageInfo.mountedVolumes():
        if not v.isValid() or not v.isReady() or bytes(v.device()) == system:
            continue
        root = v.rootPath()
        if sys.platform != "win32":
            device = bytes(v.device()).decode(errors="replace")
            if not device.startswith("/dev/") or device.startswith("/dev/loop") or root.startswith(("/boot", "/snap", "/var/lib", "/tmp", "/proc", "/sys")):
                continue
        out.append((v.displayName() or Path(root).name or root, root, v.bytesTotal()))
    return sorted(out, key=lambda d: d[1])


def choose_path(parent, title, start=None, folder=False):
    """File or folder picker whose sidebar lists every connected drive."""
    dialog = QFileDialog(parent, title, str(start or Path.home()))
    if sys.platform != "win32":
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)   # the bundled app has no native dialog plugin
    if folder:
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    else:
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(OPEN_FILTER)
    places = [Path.home()]
    for loc in (QStandardPaths.StandardLocation.DesktopLocation, QStandardPaths.StandardLocation.DownloadLocation, QStandardPaths.StandardLocation.DocumentsLocation):
        d = QStandardPaths.writableLocation(loc)
        if d and Path(d).is_dir():
            places.append(Path(d))
    places += [Path(root) for _, root, _ in external_drives()]
    dialog.setSidebarUrls([QUrl.fromLocalFile(str(x)) for x in places])
    return dialog.selectedFiles()[0] if dialog.exec() else None


def table(headers, select=True):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().hide()
    t.setShowGrid(False)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection if select else QAbstractItemView.SelectionMode.NoSelection)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setHighlightSections(False)
    t.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return t


def item(text, color=None, data=None):
    it = QTableWidgetItem(str(text))
    if color:
        it.setForeground(T.color(color))
    if data is not None:
        it.setData(Qt.ItemDataRole.UserRole, data)
    return it


def clear(layout):
    while layout.count():
        w = layout.takeAt(0)
        if w.widget():
            w.widget().hide()              # deleteLater only acts on the next event loop pass
            w.widget().deleteLater()
        elif w.layout():
            clear(w.layout())


class _Signals(QObject):
    done = Signal(object)


class _Probe(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = _Signals()

    def run(self):
        result = resources.detect(paths.data_dir())
        try:
            self.signals.done.emit(result)
        except RuntimeError:         # the window closed while probing
            pass


class Page(QWidget):
    def __init__(self, title, subtitle):
        super().__init__()
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(22, 18, 22, 18)
        self.root.setSpacing(12)
        self.root.addWidget(label(title, "h1"))
        self.subtitle = label(subtitle, "muted", wrap=True)
        self.root.addWidget(self.subtitle)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atlas Engine")
        self.resize(1320, 840)
        self.setMinimumSize(1120, 720)
        cat = load_json(paths.PACKAGE_DATA / "catalog.json", {"parts": [], "entries": []})
        self.parts = cat["parts"]
        self.entries = {e["id"]: e for e in cat["entries"]}
        self.verified = load_json(paths.PACKAGE_DATA / "verified.json", {})
        self.hw = None
        self.profile = None
        self.task = None
        self.entry = None
        self.run_dir = None
        self.force_cpu = False
        self.pool = QThreadPool(self)
        self.runner = Runner(self)
        self.runner.finished.connect(self._job_finished)
        self.job_mode = None
        self.pending = None
        self.planned_device = "cpu"
        self.pending_device = None
        self.run_threads, self.run_cap = 1, 100.0
        self.limits = Limits.load(paths.load_settings().get("limits"))
        self.governor = Governor(os.cpu_count() or 1, self)
        self.governor.limits = self.limits
        self.governor.updated.connect(self.show_usage)
        self.runner.governor = self.governor
        self.hw_config, self.hw_live = HardwareControls(False), HardwareControls(True)
        for c in (self.hw_config, self.hw_live):
            c.limit_changed.connect(self.limit_changed)
        self.hw_config.accel_clicked.connect(self.accel_configure)
        self.hw_live.accel_clicked.connect(self.accel_live)
        self.save_limits = QTimer(self)
        self.save_limits.setSingleShot(True)
        self.save_limits.setInterval(400)
        self.save_limits.timeout.connect(self._store_limits)
        self.setAcceptDrops(True)
        self._build()
        self.probe()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.probe)
        self.timer.start(15000)
        self.goto(0)

    # ---- frame --------------------------------------------------------------------
    def _build(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QFrame()
        bar.setObjectName("topbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 9, 16, 9)
        brand = label("ATLAS//ENGINE", "h2")
        brand.setFont(T.font(13, True))
        h.addWidget(brand)
        h.addWidget(label(f"{len(self.entries)} algorithms, {len(registry.ALGOS)} trainable", "faint"))
        h.addStretch(1)
        self.hw_label = label("probing hardware...", "muted")
        h.addWidget(self.hw_label)
        b = QPushButton("Runs folder")
        b.setToolTip("Choose where run history is stored. An external drive is fine for this.")
        b.clicked.connect(self.choose_runs)
        h.addWidget(b)
        outer.addWidget(bar)

        body = QHBoxLayout()
        body.setSpacing(0)
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(172)
        r = QVBoxLayout(rail)
        r.setContentsMargins(0, 12, 0, 12)
        r.setSpacing(2)
        self.step_btns = []
        for i, s in enumerate(STEPS):
            btn = QPushButton(f"{i + 1:02d}  {s}")
            btn.setObjectName("step")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, n=i: self.goto(n))
            r.addWidget(btn)
            self.step_btns.append(btn)
        r.addStretch(1)
        body.addWidget(rail)

        self.stack = QStackedWidget()
        for build in (self._page_dataset, self._page_algorithm, self._page_config, self._page_train, self._page_results, self._page_predict, self._page_export):
            self.stack.addWidget(build())
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.setCentralWidget(root)

    def goto(self, n):
        self.stack.setCurrentIndex(n)
        if n == 1:
            self.refresh_algos()
        if n == 2:
            self.build_config()
        if n == 4:
            self.refresh_runs()
        if n == 5:
            self.refresh_predict()
        if n == 6:
            self.refresh_export()
        self.sync_rail()

    def sync_rail(self):
        ready = self.profile is not None and self.task is not None and self.entry in registry.ALGOS and self.task in registry.ALGOS[self.entry]["tasks"]
        enabled = [True, True, ready, True, True, self.run_dir is not None, self.run_dir is not None]
        for i, b in enumerate(self.step_btns):
            b.setEnabled(enabled[i])
            b.setChecked(i == self.stack.currentIndex())

    def probe(self):
        p = _Probe()
        p.signals.done.connect(self._probed)
        self.pool.start(p)

    def _probed(self, hw):
        self.hw = hw
        g = hw.gpu
        state = {"ok": f"{g.name}, {g.vram_free_mb}/{g.vram_total_mb} MB free", "none": "none", "no_driver": f"{g.name}: no driver",
                 "needs_reboot": f"{g.name}: reboot needed", "error": f"{g.name or 'NVIDIA'}: error"}[g.state]
        self.hw_label.setText(f"CPU {hw.cpu_threads}t   RAM {hw.ram_available / GB:.1f}/{hw.ram_total / GB:.0f} GB   DISK {hw.disk_free / GB:.0f} GB   GPU {state}")
        self.hw_label.setToolTip(g.detail)
        self.governor.enabled = hw.limits_ok
        self.sync_hardware()
        self.refresh_drives()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and not self.runner.running:
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.goto(0)
            self.load_dataset(urls[0].toLocalFile())

    def choose_runs(self):
        d = choose_path(self, "Folder for run history", paths.runs_dir(), folder=True)
        if d:
            s = paths.load_settings()
            s["runs_dir"] = d
            paths.save_settings(s)
            self.refresh_runs()

    # ---- 01 dataset ------------------------------------------------------------------
    def _page_dataset(self):
        pg = Page("Select a dataset", "Tables (CSV, Excel, JSON, Parquet, TXT, SQLite and more), folders of images, audio or video, detection and mask folders, or a graph. You can also drop a file or folder onto this window.")
        row = QHBoxLayout()
        row.addWidget(label("look in", "muted"))
        self.look_in = QComboBox()
        self.look_in.setMinimumWidth(190)
        row.addWidget(self.look_in)
        a = QPushButton("Open file")
        a.setObjectName("primary")
        a.clicked.connect(self.pick_file)
        b = QPushButton("Open folder")
        b.clicked.connect(self.pick_folder)
        row.addWidget(a)
        row.addWidget(b)
        row.addSpacing(14)
        row.addWidget(label("or simulate", "muted"))
        self.env_box = QComboBox()
        for name, about in ENVIRONMENTS.items():
            self.env_box.addItem(name, name)
            self.env_box.setItemData(self.env_box.count() - 1, about, Qt.ItemDataRole.ToolTipRole)
        row.addWidget(self.env_box)
        c = QPushButton("Use environment")
        c.setToolTip("Reinforcement learning trains in a simulation instead of on a dataset.")
        c.clicked.connect(lambda: self.set_profile(prof.environment_profile(self.env_box.currentData())))
        row.addWidget(c)
        row.addStretch(1)
        pg.root.addLayout(row)
        self.ds_name = label("no dataset selected", "muted", wrap=True)
        pg.root.addWidget(self.ds_name)
        split = QHBoxLayout()
        split.setSpacing(12)
        self.ds_table = table(["column", "type", "missing", "unique", "range or example"], select=False)
        split.addWidget(self.ds_table, 3)
        side, s = panel()
        side.setFixedWidth(330)
        s.addWidget(label("What should the engine do?", "h2"))
        self.ds_form = QFormLayout()
        self.ds_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.task_box, self.target_box, self.time_box, self.text_box, self.user_box, self.item_box = (QComboBox() for _ in range(6))
        self.task_box.currentIndexChanged.connect(self.task_changed)
        for box in (self.target_box, self.time_box, self.text_box, self.user_box, self.item_box):
            box.currentIndexChanged.connect(self.validate_choice)
        for name, box in (("task", self.task_box), ("target", self.target_box), ("time", self.time_box), ("text", self.text_box), ("user", self.user_box), ("item", self.item_box)):
            self.ds_form.addRow(name, box)
        s.addLayout(self.ds_form)
        self.ds_msg = label("", "muted", wrap=True)
        s.addWidget(self.ds_msg)
        s.addStretch(1)
        self.ds_next = QPushButton("Choose algorithm")
        self.ds_next.setObjectName("primary")
        self.ds_next.setEnabled(False)
        self.ds_next.clicked.connect(lambda: self.goto(1))
        s.addWidget(self.ds_next)
        split.addWidget(side)
        pg.root.addLayout(split, 1)
        self.drive_list = None
        self.refresh_drives()
        return pg

    def refresh_drives(self):
        """Keep the 'look in' list in step with drives being plugged in and out."""
        drives = external_drives()
        if drives == self.drive_list:
            return
        self.drive_list = drives
        current = self.look_in.currentData()
        self.look_in.blockSignals(True)
        self.look_in.clear()
        self.look_in.addItem("Home folder", str(Path.home()))
        for name, root, size in drives:
            self.look_in.addItem(f"{name}  ({size / GB:.0f} GB)", root)
        self.look_in.setCurrentIndex(max(self.look_in.findData(current), 0))
        self.look_in.blockSignals(False)

    def pick_file(self):
        f = choose_path(self, "Dataset file", self.look_in.currentData())
        if f:
            self.load_dataset(f)

    def pick_folder(self):
        d = choose_path(self, "Dataset folder", self.look_in.currentData(), folder=True)
        if d:
            self.load_dataset(d)

    def _dataset_error(self, name, text):
        self.ds_name.setText(f"could not use {name}")
        self.ds_msg.setStyleSheet(f"color: {T.ROSE};")
        self.ds_msg.setText(text)
        self.profile = None
        self.ds_next.setEnabled(False)

    def load_dataset(self, path):
        p = Path(path)
        try:
            if p.is_dir():
                self.set_profile(prof.profile_folder(p))
            else:
                try:
                    self.set_profile(prof.profile_table(p))
                except prof.NeedsConversion:
                    self._convert_table(p)
        except (prof.ProfileError, prof.NeedsConversion) as e:
            self._dataset_error(p.name, str(e))
        except OSError as e:
            self._dataset_error(p.name, e.strerror or str(e))

    def _convert_table(self, p):
        """Excel, JSON and the like need the runtime's readers, so they load through the terminal."""
        out = paths.data_dir() / "cache" / (p.name + ".profile.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            steps = runtime.setup_steps({"groups": ("core", "formats")}, "cpu", None)
        except FileNotFoundError as e:
            self._dataset_error(p.name, str(e))
            return
        steps.append(runtime.Step(f"read {p.name}", runtime.worker_argv("cpu", "convert", p, out)))
        self.job_mode = "convert"
        self.pending = (p, out)
        self.term.clear()
        self.goto(3)
        self.begin_job(steps)

    def _fill(self, box, items, none=None):
        box.blockSignals(True)
        box.clear()
        if none:
            box.addItem(none, None)
        for name in items:
            box.addItem(name, name)
        box.blockSignals(False)

    def set_profile(self, pr, source=None):
        if source:
            pr.update(path=str(source), name=Path(source).name)
        self.profile = pr
        self.entry = None
        self.ds_msg.setStyleSheet("")
        kind = pr["kind"]
        note = f"   {pr['malformed']} malformed rows skipped" if pr.get("malformed") else ""
        if kind == "environment":
            summary = f"{pr['name']}   simulated environment: {ENVIRONMENTS[pr['name']]}"
        elif kind in ("table", "graph"):
            summary = f"{pr['name']}   {pr['rows']:,} {'nodes' if kind == 'graph' else 'rows'} x {pr['cols']} columns"
        elif kind == "detection":
            summary = f"{pr['name']}   {pr['rows']:,} images, {pr['format'].upper()} detection labels"
        elif kind == "segmentation":
            summary = f"{pr['name']}   {pr['rows']:,} images with masks"
        else:
            summary = f"{pr['name']}   {pr['rows']:,} {'images' if kind == 'images' else kind + ' files'}" + (f", {len(pr['classes'])} classes" if pr["classes"] else "")
        self.ds_name.setText(summary + note + ("   untick any column the model must not see" if kind == "table" else ""))
        t = self.ds_table
        if pr["columns"]:
            t.setColumnCount(5)
            t.setHorizontalHeaderLabels(["column", "type", "missing", "unique", "range or example"])
            t.setRowCount(len(pr["columns"]))
            colors = {"numeric": T.ACCENT, "categorical": T.CYAN, "datetime": T.VIOLET, "text": T.AMBER, "empty": T.ROSE}
            for i, c in enumerate(pr["columns"]):
                t.setItem(i, 0, item(c["name"]))
                if kind == "table":
                    t.item(i, 0).setFlags(t.item(i, 0).flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                    t.item(i, 0).setCheckState(Qt.CheckState.Checked)
                t.setItem(i, 1, item(c["kind"], colors[c["kind"]]))
                t.setItem(i, 2, item(f"{c['missing']:.1%}", T.AMBER if c["missing"] > 0.2 else T.MUTED))
                t.setItem(i, 3, item(f"{c['unique']:,}", T.MUTED))
                t.setItem(i, 4, item(f"{fmt(c['min'])} .. {fmt(c['max'])}" if c["min"] is not None else c["example"][:80], T.MUTED))
        else:
            t.setColumnCount(2)
            t.setHorizontalHeaderLabels(["class", "files"])
            t.setRowCount(len(pr["classes"]))
            for i, c in enumerate(pr["classes"]):
                t.setItem(i, 0, item(c["name"]))
                t.setItem(i, 1, item(f"{c['images']:,}" if c["images"] else "-", T.MUTED))
        t.resizeColumnsToContents()
        guess = prof.suggest(pr)
        names = [c["name"] for c in pr["columns"]]
        tasks = [k for k in prof.TASKS_FOR_KIND[kind] if not (k == "image_classification" and len(pr["classes"]) < 2)]
        self.task_box.blockSignals(True)
        self.task_box.clear()
        for k in tasks:
            self.task_box.addItem(TASKS[k], k)
        self.task_box.setCurrentIndex(max(self.task_box.findData(guess["task"]), 0))
        self.task_box.blockSignals(False)
        self._fill(self.target_box, names, "(none)")
        self._fill(self.time_box, [c["name"] for c in pr["columns"] if c["kind"] in ("datetime", "numeric")], "(row order)")
        self._fill(self.text_box, [c["name"] for c in pr["columns"] if c["kind"] in ("text", "categorical")], "(choose)")
        self._fill(self.user_box, names, "(choose)")
        self._fill(self.item_box, names, "(choose)")
        for box, key in ((self.target_box, "target"), (self.time_box, "time"), (self.text_box, "text"), (self.user_box, "user"), (self.item_box, "item")):
            box.blockSignals(True)
            box.setCurrentIndex(max(box.findData(guess.get(key)), 0))
            box.blockSignals(False)
        self.task_changed()

    def task_changed(self, *_):
        self.task = self.task_box.currentData()
        t = self.task
        table_kind = bool(self.profile) and self.profile["kind"] == "table"
        needs_target = t in ("classification", "regression", "forecasting", "seq_classification", "anomaly", "text_classification", "text_generation", "recommendation", "node_classification")
        shown = dict(target=needs_target, time=table_kind and t in ("classification", "regression", "forecasting", "seq_classification", "anomaly"),
                     text=t in ("text_classification", "text_generation"), user=t == "recommendation", item=t == "recommendation")
        for i, key in enumerate(("target", "time", "text", "user", "item"), 1):
            self.ds_form.setRowVisible(i, shown[key])
        names = {"text_generation": "response", "recommendation": "rating", "text_classification": "label", "node_classification": "label"}
        self.ds_form.labelForField(self.target_box).setText(names.get(t, "target"))
        self.validate_choice()

    def validate_choice(self, *_):
        if not self.profile:
            return
        self.task = self.task_box.currentData()
        pr, t = self.profile, self.task
        find = lambda box: next((c for c in pr["columns"] if c["name"] == box.currentData()), None)
        col, text = find(self.target_box), find(self.text_box)
        msg, ok, level = "", True, "muted"
        fixed = {"object_detection": "Boxes are read from the labels. Existing train and val folders are respected.", "semantic_segmentation": "Each mask is matched to its image by file name; each mask value is one class.",
                 "audio_classification": "Each subfolder is one class of sound.", "video_classification": "Each subfolder is one class of video.",
                 "reinforcement_learning": "The agent learns by acting in this simulation. No dataset is needed.", "image_generation": "No labels are used: the model learns to create similar images.",
                 "representation": "No labels are used: the model learns image embeddings.", "image_classification": "Each subfolder is one class; images are split per class."}
        if t in fixed:
            msg = fixed[t]
        elif t == "text_classification":
            if text is None:
                msg, ok, level = "Choose the column that holds the text.", False, "warn"
            elif col is None:
                msg, ok, level = "Choose the column with the category.", False, "warn"
            elif col["name"] == text["name"]:
                msg, ok, level = "Text and label must be different columns.", False, "warn"
            elif col["unique"] > 200:
                msg, ok, level = f"'{col['name']}' has {col['unique']:,} distinct values, which is too many for categories.", False, "warn"
            else:
                msg = "Texts are split per category; the test set is never used for training."
        elif t == "text_generation":
            if text is None:
                msg, ok, level = "Choose the column that holds the text.", False, "warn"
            else:
                msg = "Response is optional. Set it to fine-tune on instruction and response pairs."
        elif t == "recommendation":
            user, it = find(self.user_box), find(self.item_box)
            if user is None or it is None or col is None:
                msg, ok, level = "Choose the user, item and rating columns.", False, "warn"
            elif len({user["name"], it["name"], col["name"]}) < 3:
                msg, ok, level = "User, item and rating must be three different columns.", False, "warn"
            elif col["kind"] != "numeric":
                msg, ok, level = f"'{col['name']}' is {col['kind']}, but a rating must be numeric.", False, "warn"
            else:
                msg = "Interactions are split at random; unseen users and items fall back to the average."
        elif t == "node_classification":
            if col is None:
                msg, ok, level = "Choose the column in nodes.csv with the node label.", False, "warn"
            else:
                msg = "Every node passes messages; only labelled nodes are scored."
        elif prof.task_requirements(t)[0] and col is None:
            msg, ok, level = "Choose the column to predict.", False, "warn"
        elif col is not None and col["name"] == self.time_box.currentData():
            msg, ok, level = "Target and time cannot be the same column.", False, "warn"
        elif t in ("forecasting", "seq_classification") and find(self.time_box) is not None and find(self.time_box)["unique"] < pr["rows"]:
            msg, ok, level = f"'{self.time_box.currentData()}' repeats, so these rows are many series mixed together. Use Regression or Classification.", False, "warn"
        elif t in ("regression", "forecasting") and col is not None and col["kind"] != "numeric":
            msg, ok, level = f"'{col['name']}' is {col['kind']}, but this task needs a numeric target.", False, "warn"
        elif t == "classification" and col is not None and col["unique"] > 50:
            msg, ok, level = f"'{col['name']}' has {col['unique']:,} distinct values, which is too many for classes. Did you mean regression?", False, "warn"
        elif t == "anomaly":
            msg = "Target is optional. A 0/1 label column lets the engine score detection quality."
            if col is not None and col["unique"] != 2:
                msg, ok, level = f"A label column needs exactly 2 values; '{col['name']}' has {col['unique']}.", False, "warn"
        elif t in ("clustering", "dimreduction"):
            msg = "Unsupervised: every usable column is an input."
        elif t in ("forecasting", "seq_classification"):
            msg = "Data is split in time order, so the future never leaks into training."
        elif self.time_box.currentData():
            msg = "A time column is set, so splits follow time order."
        else:
            msg = "Rows are split at random, stratified by class where possible."
        if pr["kind"] == "table":
            pr["target_min"] = col["min"] if col is not None and t not in ("classification", "text_classification") else None
        self.ds_msg.setStyleSheet(f"color: {T.AMBER if level == 'warn' else T.MUTED};")
        self.ds_msg.setText(msg)
        self.ds_next.setEnabled(ok)
        self.valid_choice = ok
        self.sync_rail()

    # ---- 02 algorithm ---------------------------------------------------------------
    def _page_algorithm(self):
        pg = Page("Choose an algorithm", "All 830 atlas entries are here. Only the ones this version can really train offer a Use button.")
        row = QHBoxLayout()
        self.algo_search = QLineEdit()
        self.algo_search.setPlaceholderText("search name, purpose or example")
        self.algo_search.textChanged.connect(self.refresh_algos)
        self.algo_view = QComboBox()
        self.algo_view.addItem("Suited to my task", "task")
        self.algo_view.addItem("All trainable", "ready")
        self.algo_view.addItem("Entire atlas", "all")
        self.algo_view.currentIndexChanged.connect(self.refresh_algos)
        self.algo_cat = QComboBox()
        self.algo_cat.addItem("All categories", 0)
        for p in self.parts:
            self.algo_cat.addItem(f"{p['part']:02d}  {p['title']}", p["part"])
        self.algo_cat.currentIndexChanged.connect(self.refresh_algos)
        row.addWidget(self.algo_search, 2)
        row.addWidget(self.algo_view)
        row.addWidget(self.algo_cat, 1)
        pg.root.addLayout(row)
        self.algo_count = label("", "faint")
        pg.root.addWidget(self.algo_count)
        split = QSplitter()
        split.setChildrenCollapsible(False)
        self.algo_table = table(["#", "algorithm", "category", "status"])
        self.algo_table.itemSelectionChanged.connect(self.show_algo)
        split.addWidget(self.algo_table)
        side = QFrame()
        side.setObjectName("panel")
        outer = QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setStyleSheet(f"background: {T.PANEL};")
        self.algo_detail = QVBoxLayout(inner)
        self.algo_detail.setContentsMargins(14, 14, 14, 14)
        self.algo_detail.setSpacing(8)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        foot = QHBoxLayout()
        foot.setContentsMargins(14, 8, 14, 12)
        self.algo_use = QPushButton("Use this algorithm")
        self.algo_use.setObjectName("primary")
        self.algo_use.setEnabled(False)
        self.algo_use.clicked.connect(self.use_algo)
        foot.addStretch(1)
        foot.addWidget(self.algo_use)
        outer.addLayout(foot)
        split.addWidget(side)
        split.setSizes([640, 480])
        pg.root.addWidget(split, 1)
        return pg

    def refresh_algos(self, *_):
        q = self.algo_search.text().strip().lower()
        view = self.algo_view.currentData()
        cat = self.algo_cat.currentData()
        if view == "task" and (not self.profile or not self.task):
            view = "ready"
        rows = []
        for e in self.entries.values():
            status, _ = registry.status_of(e["id"], e["part"])
            algo = registry.ALGOS.get(e["id"])
            if view == "ready" and status != "ready":
                continue
            if view == "task" and (algo is None or self.task not in algo["tasks"]):
                continue
            if cat and e["part"] != cat:
                continue
            if q and q not in f"{e['name']} {e['why']} {e['example']}".lower():
                continue
            rows.append((e, status, algo))
        t = self.algo_table
        keep = self.entry
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, (e, status, algo) in enumerate(rows):
            t.setItem(i, 0, item(e["id"], T.FAINT, e["id"]))
            t.setItem(i, 1, item(e["name"]))
            t.setItem(i, 2, item(e["category"], T.MUTED))
            tag, color = STATUS_LABEL[status], STATUS_COLOR[status]
            if algo and self.profile and self.task in algo["tasks"]:
                state, reasons = prof.compatibility(algo, self.task, self.profile)
                if state != "ok":
                    tag, color = ("LIMITED", T.AMBER) if state == "warn" else ("BLOCKED", T.ROSE)
            t.setItem(i, 3, item(tag, color))
            if e["id"] == keep:
                t.selectRow(i)
        t.blockSignals(False)
        head = t.horizontalHeader()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col, width in ((0, 52), (2, 190), (3, 96)):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            t.setColumnWidth(col, width)
        self.algo_count.setText(f"{len(rows)} shown")
        self.show_algo()

    def _detail_section(self, title, text, obj="muted"):
        self.algo_detail.addWidget(label(title, "h2"))
        lb = label(text, obj, wrap=True)
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.algo_detail.addWidget(lb)

    def show_algo(self):
        clear(self.algo_detail)
        self.algo_use.setEnabled(False)
        row = self.algo_table.currentRow()
        selected = self.algo_table.selectedItems()
        if row < 0 or not selected:
            self.algo_detail.addWidget(label("Select an algorithm to see what it needs.", "muted", wrap=True))
            self.algo_detail.addStretch(1)
            return
        eid = self.algo_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        e = self.entries[eid]
        status, why_not = registry.status_of(eid, e["part"])
        algo = registry.ALGOS.get(eid)
        title = label(f"#{eid}  {e['name']}", "h1", wrap=True)
        title.setFont(T.font(13, True))
        self.algo_detail.addWidget(title)
        self.algo_detail.addWidget(label(e["category"], "faint", wrap=True))
        self._detail_section("Used for", e["why"])
        self._detail_section("Example", e["example"])
        if algo:
            tasks = ", ".join(TASKS[t] for t in algo["tasks"])
            libs = ", ".join(g for g in algo["groups"] if g != "core") or "scikit-learn"
            hw = "CPU or NVIDIA GPU" if algo["gpu"] else "CPU only"
            if algo.get("max_rows"):
                hw += f". Practical limit about {algo['max_rows']:,} rows"
            libs = ", ".join(g for g in algo["groups"] if g not in ("core", "formats")) or "scikit-learn"
            self._detail_section("Runs on", f"{hw}\nLibraries: {libs}\nTasks: {tasks}")
            if algo.get("note"):
                self._detail_section("Note", algo["note"], "warn")
            self._detail_section("Settings", ", ".join(p["label"] for p in algo["params"]) or "No settings to tune.")
            task = self.task if self.task in algo["tasks"] else next(iter(algo["tasks"]))
            v = self.verified.get(f"{eid}:{task}", {})
            self._detail_section("Export", self._export_text(v))
            if self.profile and self.task:
                state, reasons = prof.compatibility(algo, self.task, self.profile)
                if reasons:
                    self._detail_section("With your data", " ".join(reasons), "warn" if state == "warn" else "err")
                self.algo_use.setEnabled(state != "blocked" and getattr(self, "valid_choice", False))
                if state == "blocked" and reasons:
                    self.algo_use.setToolTip(reasons[0])
                if state == "blocked":
                    self.algo_use.setToolTip("Not compatible with this dataset and task.")
            else:
                self.algo_detail.addWidget(label("Open a dataset first to use this algorithm.", "warn", wrap=True))
        else:
            self._detail_section("Status", why_not, "warn" if status == "planned" else "muted")
        code = label(e["code"], "code", wrap=True)
        code.setFont(T.font(9))
        code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.algo_detail.addWidget(label("Atlas code", "h2"))
        self.algo_detail.addWidget(code)
        self.algo_detail.addStretch(1)
        self.selected_algo = eid

    @staticmethod
    def _export_text(v):
        if not v:
            return "Not self-tested in this build."
        if v.get("onnx") == "not_applicable":
            return "Cannot process new data, so there is nothing to deploy."
        onnx = "ONNX verified against the original model" if v.get("onnx") == "ok" else "No ONNX path (use the REST API bundle)"
        return f"{onnx}. REST API bundle available."

    def use_algo(self):
        self.entry = self.selected_algo
        self.goto(2)

    # ---- 03 configure ---------------------------------------------------------------
    def _page_config(self):
        pg = Page("Configure", "Only the settings this algorithm actually has are shown.")
        split = QHBoxLayout()
        split.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.cfg_form = QVBoxLayout(inner)
        self.cfg_form.setContentsMargins(0, 0, 8, 0)
        self.cfg_form.setSpacing(12)
        scroll.setWidget(inner)
        split.addWidget(scroll, 3)
        side, s = panel()
        side.setFixedWidth(370)
        s.addWidget(label("Hardware plan", "h2"))
        self.device_box = QComboBox()
        self.device_box.addItem("Automatic", "auto")
        self.device_box.addItem("Force CPU", "cpu")
        self.device_box.addItem("Force GPU", "gpu")
        self.device_box.currentIndexChanged.connect(self.update_plan)
        s.addWidget(self.device_box)
        self.plan_label = label("", "muted", wrap=True)
        s.addWidget(self.plan_label)
        s.addStretch(1)
        s.addWidget(self.hw_config)
        self.train_btn = QPushButton("Train")
        self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self.start_training)
        s.addWidget(self.train_btn)
        split.addWidget(side)
        pg.root.addLayout(split, 1)
        self.param_widgets = {}
        self.extra_widgets = {}
        return pg

    def _spin(self, p):
        if p["type"] in ("int", "optional_int"):
            w = QSpinBox()
            w.setRange(int(p["min"]), int(p["max"]))
            w.setValue(int(p["default"]))
        else:
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(float(p["min"]), float(p["max"]))
            w.setValue(float(p["default"]))
        w.setToolTip(f"{p['min']} to {p['max']}")
        return w

    def build_config(self):
        clear(self.cfg_form)
        self.param_widgets, self.extra_widgets = {}, {}
        if self.entry not in registry.ALGOS:
            return
        algo, e = registry.ALGOS[self.entry], self.entries[self.entry]
        impl = algo["tasks"][self.task]
        self.cfg_form.addWidget(label(f"#{e['id']}  {e['name']}", "h2"))
        self.cfg_form.addWidget(label(TASKS[self.task], "muted"))

        box, lay = panel()
        lay.addWidget(label("Algorithm settings", "h2"))
        f = QFormLayout()
        for p in algo["params"]:
            if p["type"] == "choice":
                w = QComboBox()
                w.addItems(p["choices"])
                w.setCurrentText(p["default"])
            elif p["type"] == "text":
                w = QLineEdit(p["default"])
            else:
                w = self._spin(p)
            self.param_widgets[p["name"]] = w
            f.addRow(p["label"], w)
        if not algo["params"]:
            lay.addWidget(label("This algorithm has no settings to adjust.", "muted"))
        lay.addLayout(f)
        self.cfg_form.addWidget(box)

        box, lay = panel()
        lay.addWidget(label("Data handling", "h2"))
        f = QFormLayout()
        val = self._spin(dict(type="float", min=0.05, max=0.4, default=0.15))
        test = self._spin(dict(type="float", min=0.05, max=0.4, default=0.15))
        val.setDecimals(2), test.setDecimals(2)
        self.extra_widgets.update(val=val, test=test)
        f.addRow("validation share", val)
        f.addRow("test share", test)
        if self.task in ("classification", "regression"):
            cv = self._spin(dict(type="int", min=0, max=20, default=0))
            tune = self._spin(dict(type="int", min=0, max=200, default=0))
            cv.setToolTip("0 turns cross-validation off")
            tune.setToolTip("0 turns tuning off")
            self.extra_widgets.update(cv=cv, tune=tune)
            f.addRow("cross-validation folds", cv)
            f.addRow("tuning trials", tune)
        if self.task == "anomaly":
            c = self._spin(dict(type="float", min=0.001, max=0.5, default=0.05))
            c.setDecimals(3)
            self.extra_widgets["contamination"] = c
            f.addRow("expected abnormal share", c)
        if self.task == "forecasting":
            h = self._spin(dict(type="int", min=1, max=100000, default=12))
            self.extra_widgets["horizon"] = h
            f.addRow("future steps", h)
            if impl["adapter"] == "lag":
                lags = self._spin(dict(type="int", min=1, max=5000, default=12))
                self.extra_widgets["lags"] = lags
                f.addRow("lag features", lags)
        lay.addLayout(f)
        self.cfg_form.addWidget(box)
        if "torch" in algo["groups"]:
            box, lay = panel()
            f = QFormLayout()
            b = self._spin(dict(type="int", min=0, max=4096, default=0))
            b.setToolTip("0 lets the engine choose from your VRAM")
            b.valueChanged.connect(self.update_plan)
            self.extra_widgets["batch"] = b
            f.addRow("batch size (0 = automatic)", b)
            lay.addLayout(f)
            self.cfg_form.addWidget(box)
        self.cfg_form.addStretch(1)
        self.update_plan()

    def current_plan(self, choice=None):
        algo = registry.ALGOS[self.entry]
        batch = self.extra_widgets["batch"].value() if "batch" in self.extra_widgets else 0
        choice = choice or self.device_box.currentData()
        flavor = "cuda" if (choice != "cpu" and algo["gpu"] and self.hw.gpu.usable) else "cpu"
        return resources.make_plan(self.hw, algo, self.task, self.profile, choice, runtime.installed_groups(flavor, self.hw.gpu.backend), batch, self.limits)

    def update_plan(self, *_):
        if self.entry not in registry.ALGOS or self.hw is None or self.task not in registry.ALGOS[self.entry]["tasks"]:
            return
        plan = self.current_plan()
        lines = [f"Device: {'NVIDIA GPU' if plan.device == 'cuda' else 'CPU'}", f"Threads: {plan.threads}"] + plan.notes
        html = "<br>".join(lines)
        if plan.warnings:
            html += "<br><br>" + "<br>".join(f"<span style='color:{T.AMBER}'>{w}</span>" for w in plan.warnings)
        if plan.blockers:
            html += "<br><br>" + "<br>".join(f"<span style='color:{T.ROSE}'>{w}</span>" for w in plan.blockers)
        self.plan_label.setText(html)
        self.train_btn.setEnabled(not plan.blockers)
        self.hw_config.rows["vram"].setVisible(plan.device == "cuda" and "torch" in registry.ALGOS[self.entry]["groups"])
        self.sync_accel()

    def collect_params(self):
        out = {}
        for name, w in self.param_widgets.items():
            out[name] = w.currentText() if isinstance(w, QComboBox) else w.text().strip() if isinstance(w, QLineEdit) else w.value()
        return out

    # ---- 04 train ---------------------------------------------------------------------
    def _page_train(self):
        pg = Page("Train", "Setup, hardware configuration and training all run here, one fixed step at a time.")
        self.train_page = pg
        row = QHBoxLayout()
        row.setSpacing(4)
        self.chips = {}
        for s in STAGES:
            c = label(s, "faint")
            c.setAlignment(Qt.AlignmentFlag.AlignCenter)
            c.setStyleSheet(f"border: 1px solid {T.LINE}; padding: 5px;")
            row.addWidget(c, 1)
            self.chips[s] = c
        pg.root.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedHeight(8)
        self.progress.setTextVisible(False)
        pg.root.addWidget(self.progress)
        pg.root.addWidget(self.hw_live)
        self.term = Terminal()
        pg.root.addWidget(self.term, 1)
        foot = QHBoxLayout()
        self.res_label = label("idle", "muted")
        foot.addWidget(self.res_label, 1)
        self.train_msg = label("", "warn", wrap=True)
        foot.addWidget(self.train_msg, 3)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.runner.cancel)
        foot.addWidget(self.cancel_btn)
        self.to_results = QPushButton("View results")
        self.to_results.setObjectName("primary")
        self.to_results.setEnabled(False)
        self.to_results.clicked.connect(lambda: self.goto(4))
        foot.addWidget(self.to_results)
        pg.root.addLayout(foot)
        return pg

    def mark_stage(self, name):
        i = STAGES.index(name)
        for j, s in enumerate(STAGES):
            c, col = self.chips[s], (T.ACCENT_DIM if j < i else T.ACCENT if j == i else T.FAINT)
            weight = "700" if j == i else "400"
            c.setStyleSheet(f"color: {col}; border: 1px solid {T.ACCENT if j == i else T.LINE}; padding: 5px; font-weight: {weight};")
        self.progress.setValue(int(1000 * i / len(STAGES)))

    def begin_job(self, steps):
        self.cancel_btn.setEnabled(True)
        self.to_results.setEnabled(False)
        self.train_msg.setText("")
        self.runner.finished.disconnect()
        self.runner.finished.connect(self._job_finished)
        self.runner.start(steps, self.term, self.on_event)

    def start_training(self):
        if self.runner.running or self.hw is None:
            return
        self.force_cpu = False
        self.pending_device = None
        self._launch(self.device_box.currentData())

    # ---- hardware controls ------------------------------------------------------------
    def sync_hardware(self):
        """Bring both sets of controls in line with the machine and the saved limits."""
        hw = self.hw
        if hw is None:
            return
        g, lim = hw.gpu, self.limits
        no_gpu = g.detail or "No usable NVIDIA GPU."
        for c in (self.hw_config, self.hw_live):
            c.set_ranges(hw.ram_total / GB, g.vram_total_mb / 1024 if g.vram_total_mb else 6)
            c.rows["cpu"].set_value(lim.cpu_pct)
            c.rows["ram"].set_value(int(lim.ram_gb * 2) if lim.ram_gb else c.ram_top)
            c.rows["gpu"].set_value(lim.gpu_pct)
            c.rows["vram"].set_value(int(lim.vram_gb * 2) if lim.vram_gb else c.vram_top)
            c.rows["ram"].set_available(hw.limits_ok, hw.limits_why)
            c.rows["gpu"].set_available(hw.limits_ok and g.usable, hw.limits_why or no_gpu)
            c.rows["vram"].set_available(g.usable, no_gpu)
        self.sync_accel()

    def sync_accel(self):
        """The GPU accelerator appears when a GPU is found, and works when the algorithm can use it."""
        hw = self.hw
        if hw is None:
            return
        g = hw.gpu
        algo = registry.ALGOS.get(self.entry)
        running = self.runner.running and self.job_mode == "train"
        if running:
            on = self.planned_device == "cuda"
        elif algo and self.profile and self.task in algo["tasks"]:
            on = self.current_plan().device == "cuda"
        else:
            on = False
        can = bool(g.usable and algo and algo["gpu"])
        tip = (g.detail if not g.usable else "Choose an algorithm first." if not algo
               else "This algorithm runs on the CPU only, so a GPU cannot speed it up." if not algo["gpu"] else f"Train on {g.name}.")
        self.hw_config.set_accel(g.state != "none", can, on, tip)
        self.hw_live.set_accel(g.state != "none", can and running, on, "Restarts this training on the other device. Progress so far is lost." if can else tip)

    def limit_changed(self, key, value):
        source, lim = self.sender(), self.limits
        if key in ("cpu", "gpu"):
            setattr(lim, f"{key}_pct", value)
        else:
            top = source.ram_top if key == "ram" else source.vram_top
            setattr(lim, f"{key}_gb", 0.0 if value >= top else value / 2)
        for c in (self.hw_config, self.hw_live):
            if c is not source:
                c.rows[key].set_value(value)
        self.governor.set_limits(lim)
        self.save_limits.start()
        self.update_plan()

    def _store_limits(self):
        s = paths.load_settings()
        s["limits"] = asdict(self.limits)
        paths.save_settings(s)

    def accel_configure(self, on):
        self.device_box.setCurrentIndex(self.device_box.findData("gpu" if on else "cpu"))
        self.update_plan()

    def confirm(self, text):
        return QMessageBox.question(self, "Atlas Engine", text) == QMessageBox.StandardButton.Yes

    def accel_live(self, on):
        if not self.runner.running or self.job_mode != "train" or on == (self.planned_device == "cuda"):
            self.sync_accel()
            return
        target = "gpu" if on else "cpu"
        download = [n for n in self.current_plan(target).notes if n.startswith("First-time download")]
        extra = f" {download[0]}." if download else ""
        if not self.confirm(f"Restart this training on the {target.upper()}? Progress so far will be lost.{extra}"):
            self.sync_accel()
            return
        self.device_box.setCurrentIndex(self.device_box.findData(target))
        self.pending_device = target
        self.runner.cancel()

    def show_usage(self, st):
        c = self.hw_live
        if st.get("state") == "idle":
            for key in ("cpu", "ram", "gpu"):
                c.show_reading(key, "-")
            c.show_status("")
            return
        if "cpu_pct" in st:
            c.show_reading("cpu", f"{st['cpu_pct']:.0f}% of machine", st["cpu_pct"] / 100, st["cpu_slowed"])
        if "ram_bytes" in st:
            top = (self.limits.ram_gb * GB) or st["ram_total"]
            c.show_reading("ram", f"{st['ram_bytes'] / GB:.1f} GB in use", st["ram_bytes"] / top, st["ram_slowed"])
        gpu = st.get("gpu")
        c.show_reading("gpu", f"{gpu['util']:.0f}%   {gpu['used_mb'] / 1024:.1f}/{gpu['total_mb'] / 1024:.1f} GB" if gpu else "not found",
                       (gpu["util"] / 100) if gpu else 0, bool(st.get("gpu_gated")))
        if st.get("paused"):
            text = f"Paused: only {st['available'] / 2 ** 20:.0f} MB of RAM is free on this machine. Training continues by itself once memory is free."
        elif st.get("ram_raised_to"):
            text = (f"This job needs more RAM than your limit, and this machine has no swap to hold the rest. "
                    f"The limit has risen to {st['ram_raised_to'] / GB:.1f} GB so training keeps moving.")
        elif st.get("ram_slowed"):
            text = "Low on free memory: training is slower so this machine stays usable." if st.get("ram_guard") else "RAM limit reached: training is slower, and still running."
        elif st.get("cpu_slowed"):
            text = "Held to your CPU limit: training is slower, and still running."
        elif st.get("gpu_gated"):
            text = "Held to your GPU limit: training is slower, and still running."
        elif self.limits.cpu_pct > self.run_cap + 1:
            text = f"This run started with {self.run_threads} threads, so it cannot use more than {self.run_cap:.0f}% of the CPU. A higher limit applies from the next run."
        else:
            text = ""
        c.show_status(text)

    def _write_limits(self, t):
        lim, hw = self.limits, self.hw
        if lim.cpu_pct >= 100 and not lim.ram_gb and lim.gpu_pct >= 100 and not lim.vram_gb:
            return
        t.write("$ apply limits", "cmd")
        t.table([("cpu", "no limit" if lim.cpu_pct >= 100 else f"{lim.cpu_pct}% of the machine"), ("ram", f"{lim.ram_gb:.1f} GB" if lim.ram_gb else "no limit"),
                 ("gpu", "no limit" if lim.gpu_pct >= 100 else f"{lim.gpu_pct}%"), ("vram", f"{lim.vram_gb:.1f} GB" if lim.vram_gb else "no limit")])
        if hw.limits_ok:
            t.write("  limits slow this run down and never stop it; you can change them while it trains", "muted")
        else:
            t.write(f"  ! {hw.limits_why} CPU use is reduced by running fewer threads.", "warn")

    def _launch(self, choice, note=None):
        algo = registry.ALGOS[self.entry]
        e = self.entries[self.entry]
        self.hw = resources.detect(paths.data_dir())
        self._probed(self.hw)
        plan = self.current_plan(choice)
        self.term.clear()
        self.goto(3)
        self.job_mode = "train"
        self.mark_stage("hardware")
        t = self.term
        if note:
            t.write(note, "warn")
        g = self.hw.gpu
        t.write("$ probe hardware", "cmd")
        t.table([("cpu", f"{self.hw.cpu_threads} threads"), ("ram", f"{self.hw.ram_available / GB:.1f} GB free of {self.hw.ram_total / GB:.1f} GB"),
                 ("disk", f"{self.hw.disk_free / GB:.0f} GB free"),
                 ("gpu", f"{g.name}, driver {g.driver}, CUDA {g.cuda}, {g.vram_free_mb} MB free" if g.usable else (f"{g.name or 'none'}: {g.detail}"))])
        t.write("$ plan job", "cmd")
        t.table([("algo", f"#{e['id']} {e['name']}"), ("task", TASKS[self.task]), ("device", plan.device), ("threads", plan.threads), ("batch", plan.batch_size)])
        for n in plan.notes:
            t.write(f"  {n}", "muted")
        for w in plan.warnings:
            t.write(f"  ! {w}", "warn")
        if plan.blockers:
            for w in plan.blockers:
                t.write(f"  x {w}", "error")
            self.train_msg.setText(plan.blockers[0])
            return
        self._write_limits(t)
        job = self._make_job(e, plan)
        path = paths.data_dir() / "cache" / f"{job['run_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, indent=1))
        self.pending_run = Path(job["run_dir"])
        threads = str(plan.threads)
        env = dict(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        if plan.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            steps = runtime.setup_steps(algo, plan.flavor, g.backend)
        except FileNotFoundError as err:
            t.write(str(err), "error")
            self.train_msg.setText(str(err))
            return
        if steps:
            self.mark_stage("runtime")
        steps.append(runtime.Step("check runtime sees the planned device", runtime.worker_argv(plan.flavor, "selfcheck", *(["xgboost"] if plan.device == "cuda" and "boost" in algo["groups"] else [])), env=env if plan.device == "cpu" else {}))
        steps.append(runtime.Step(f"train #{e['id']} {e['name']}", runtime.worker_argv(plan.flavor, "train", path), env=env, governed=True))
        self.planned_device = plan.device
        self.last_batch = plan.batch_size
        self.run_threads, self.run_cap = plan.threads, plan.threads / self.hw.cpu_threads * 100
        self.governor.use_gpu = plan.device == "cuda"
        self.begin_job(steps)
        self.sync_accel()

    def _make_job(self, e, plan):
        x = self.extra_widgets
        pr = self.profile
        rid = time.strftime("%Y%m%d-%H%M%S") + f"-{e['id']}"
        job = dict(run_id=rid, run_dir=str(paths.runs_dir() / rid), entry=e["id"], name=e["name"], task=self.task,
                   dataset=dict(path=pr["path"], target=self.target_box.currentData(), time_column=self.time_box.currentData()),
                   params=self.collect_params(), split=dict(val=x["val"].value(), test=x["test"].value(), seed=42),
                   device=plan.device, threads=plan.threads, batch_size=plan.batch_size)
        ds = job["dataset"]
        if pr["kind"] == "table":
            ticks = [self.ds_table.item(i, 0) for i in range(self.ds_table.rowCount())]
            if any(c.checkState() != Qt.CheckState.Checked for c in ticks):
                ds["features"] = [c.text() for c in ticks if c.checkState() == Qt.CheckState.Checked]
        if self.task not in ("classification", "regression", "forecasting", "seq_classification", "anomaly"):
            ds["time_column"] = None
        if self.task in ("clustering", "dimreduction") or pr["kind"] in ("images", "detection", "segmentation", "audio", "video", "environment"):
            ds["target"] = None
        if self.task in ("text_classification", "text_generation"):
            ds["text_column"] = self.text_box.currentData()
        if self.task == "recommendation":
            ds.update(user_column=self.user_box.currentData(), item_column=self.item_box.currentData())
        if pr["kind"] == "environment":
            ds["env"] = pr["env"]
        if plan.device == "cuda" and self.limits.vram_gb:
            job["vram_limit_mb"] = int(self.limits.vram_gb * 1024)
        for key, src in (("cv", "cv"), ("tune", "tune"), ("contamination", "contamination"), ("horizon", "horizon"), ("seq_len", "lags")):
            if src in x:
                job[key] = x[src].value()
        return job

    def on_event(self, ev):
        k, t = ev["t"], self.term if self.job_mode in ("train", "convert") else self.active_term
        if k == "stage" and ev["name"] in STAGES:
            self.mark_stage(ev["name"])
            t.write(f"[{ev['name']}]", "ok")
        elif k == "log":
            t.write(f"  {ev['msg']}", ev.get("level", "info"))
        elif k == "epoch":
            t.write(f"  epoch {ev['epoch']}/{ev['total']}  train {ev['train_loss']:.5f}  val {ev['val_loss']:.5f}", "muted")
            base = STAGES.index("train") / len(STAGES)
            self.progress.setValue(int(1000 * (base + ev["epoch"] / ev["total"] / len(STAGES))))
        elif k == "res":
            gpu = f"   GPU {ev['gpu_mb']} MB" if ev.get("gpu_mb") is not None else ""
            self.res_label.setText(f"memory {ev['rss_mb']} MB{gpu}")
        elif k == "selfcheck":
            t.table([("torch", ev.get("torch") or "not needed"), ("cuda", ev.get("gpu_name") or ("available" if ev.get("cuda") else "not in use"))])
            if self.job_mode == "train" and self.planned_device == "cuda" and not ev.get("cuda"):
                t.write(f"  ! the runtime cannot reach the GPU {ev.get('error', '')}: switching to CPU", "warn")
                self.force_cpu = True
                self.runner.cancel()
        elif k == "error":
            t.write(f"x {ev['msg']}", "error")
            self.last_error = ev["msg"]
        elif k == "done":
            self.run_dir = Path(ev["run_dir"])
            s = ev["summary"]
            p = s["primary"]
            t.write(f"done in {s['train_seconds']}s   {p['name']} = {fmt(p['value'])}   saved to {ev['run_dir']}", "ok")
        elif k == "predicted":
            self.show_prediction(ev)
        elif k == "export":
            self.show_export(ev)

    def _job_finished(self, ok, cancelled):
        self.cancel_btn.setEnabled(False)
        mode = self.job_mode
        if mode == "train" and self.force_cpu:
            self.force_cpu = False
            self._launch("cpu")
            return
        if mode == "train" and self.pending_device:
            target, self.pending_device = self.pending_device, None
            self._launch(target, note=f"restarted on the {target.upper()}")
            return
        if mode == "train" and not ok and not cancelled and self._retry_after_gpu_memory():
            return
        if mode == "convert":
            if ok:
                src, out = self.pending
                self.goto(0)
                try:
                    self.set_profile(prof.profile_table(out), source=src)
                except (prof.ProfileError, prof.NeedsConversion) as e:
                    self._dataset_error(Path(src).name, str(e))
            elif not cancelled:
                self.goto(0)
                self._dataset_error(Path(self.pending[0]).name, getattr(self, "last_error", None) or "That file could not be read. The Train page's terminal shows why.")
                self.last_error = None
            return
        if mode == "train":
            if ok:
                self.progress.setValue(1000)
                for c in self.chips.values():
                    c.setStyleSheet(f"color: {T.ACCENT_DIM}; border: 1px solid {T.LINE}; padding: 5px;")
                self.to_results.setEnabled(True)
                self.res_label.setText("finished")
                self.load_run(self.run_dir)
            elif cancelled:
                self.term.write("cancelled", "warn")
                self.train_msg.setText("Training was cancelled.")
                self.res_label.setText("cancelled")
            else:
                msg = getattr(self, "last_error", None) or "A step failed. The terminal above shows why."
                self.train_msg.setText(msg)
                self.res_label.setText("failed")
                self.last_error = None
            self.sync_rail()
            return
        if mode == "predict":
            self.predict_btn.setEnabled(True)
            if not ok:
                self.predict_msg.setText(getattr(self, "last_error", None) or "Prediction failed. See the log.")
        if mode == "export":
            self.export_btn.setEnabled(True)
            if not ok and not cancelled:
                self.export_msg.setText(getattr(self, "last_error", None) or "Export failed. See the log.")

    def _retry_after_gpu_memory(self):
        """A GPU that runs out of memory costs time, not the run: smaller batches, then the CPU."""
        error = getattr(self, "last_error", None) or ""
        if self.planned_device != "cuda" or "out of memory" not in error.lower():
            return False
        self.last_error = None
        box = self.extra_widgets.get("batch")
        if box is not None and self.last_batch > 1:
            box.setValue(max(1, self.last_batch // 2))
            self._launch("gpu", note=f"the GPU ran out of memory: restarting with batch size {box.value()}")
        else:
            self._launch("cpu", note="the GPU ran out of memory at the smallest batch: restarting on the CPU")
        return True

    # ---- 05 results -----------------------------------------------------------------
    def _page_results(self):
        pg = Page("Results", "Test-set numbers come from data the model never saw during training.")
        self.result_title = label("", "h2", wrap=True)
        pg.root.addWidget(self.result_title)
        self.result_warn = label("", "warn", wrap=True)
        self.result_warn.hide()
        pg.root.addWidget(self.result_warn)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.res_box = QVBoxLayout(inner)
        self.res_box.setContentsMargins(0, 0, 8, 0)
        self.res_box.setSpacing(12)
        self.tiles = QGridLayout()
        self.tiles.setSpacing(8)
        self.res_box.addLayout(self.tiles)
        self.chart_grid = QGridLayout()
        self.chart_grid.setSpacing(10)
        self.res_box.addLayout(self.chart_grid)
        self.res_box.addStretch(1)
        scroll.setWidget(inner)
        pg.root.addWidget(scroll, 3)
        head = QHBoxLayout()
        head.addWidget(label("Compare runs", "h2"))
        head.addStretch(1)
        self.res_folder = QPushButton("Open run folder")
        self.res_folder.clicked.connect(lambda: self.run_dir and open_folder(self.run_dir))
        again = QPushButton("Try another algorithm")
        again.clicked.connect(lambda: self.goto(1))
        go_p = QPushButton("Predict")
        go_p.clicked.connect(lambda: self.goto(5))
        go_e = QPushButton("Export")
        go_e.setObjectName("primary")
        go_e.clicked.connect(lambda: self.goto(6))
        self.res_actions = [self.res_folder, go_p, go_e]
        for w in (self.res_folder, again, go_p, go_e):
            head.addWidget(w)
        pg.root.addLayout(head)
        self.runs_table = table(["run", "algorithm", "task", "metric", "test", "validation", "seconds", "peak RAM", "device"])
        self.runs_table.setMaximumHeight(190)
        self.runs_table.itemSelectionChanged.connect(self.pick_run)
        pg.root.addWidget(self.runs_table, 1)
        return pg

    def all_runs(self):
        out = []
        for d in paths.runs_dir().iterdir():
            m = load_json(d / "metrics.json", None) if d.is_dir() else None
            if m:
                out.append((d, m))
        return sorted(out, key=lambda x: x[1].get("created", ""), reverse=True)

    def refresh_runs(self):
        t = self.runs_table
        runs = self.all_runs()
        t.blockSignals(True)
        t.setRowCount(len(runs))
        for i, (d, m) in enumerate(runs):
            p = m["primary"]
            val = (m.get("val") or {}).get(p["name"])
            t.setItem(i, 0, item(m["run_id"], T.FAINT, str(d)))
            t.setItem(i, 1, item(m.get("name") or m["entry"]))
            t.setItem(i, 2, item(TASKS.get(m["task"], m["task"]), T.MUTED))
            t.setItem(i, 3, item(p["name"], T.MUTED))
            t.setItem(i, 4, item(fmt(p["value"]), T.ACCENT))
            t.setItem(i, 5, item(fmt(val), T.MUTED))
            t.setItem(i, 6, item(m["train_seconds"], T.MUTED))
            t.setItem(i, 7, item(f"{m.get('peak_rss_mb', 0)} MB", T.MUTED))
            t.setItem(i, 8, item(m.get("device", ""), T.MUTED))
            if self.run_dir and Path(d) == Path(self.run_dir):
                t.selectRow(i)
        t.blockSignals(False)
        t.resizeColumnsToContents()
        if self.run_dir:
            self.load_run(self.run_dir)
        else:
            self.result_title.setText("No run selected. Train a model or pick a past run below." if not runs else "Pick a run below to see its results.")
        for w in self.res_actions:
            w.setEnabled(self.run_dir is not None)

    def pick_run(self):
        r = self.runs_table.currentRow()
        if r >= 0 and self.runs_table.selectedItems():
            self.run_dir = Path(self.runs_table.item(r, 0).data(Qt.ItemDataRole.UserRole))
            self.load_run(self.run_dir)
            self.sync_rail()
            for w in self.res_actions:
                w.setEnabled(True)

    def load_run(self, d):
        m = load_json(Path(d) / "metrics.json", None)
        if not m:
            return
        charts = load_json(Path(d) / "charts.json", [])
        p = m["primary"]
        self.result_title.setText(f"{m.get('name')}   {TASKS.get(m['task'], m['task'])}   {m['created']}")
        clear(self.tiles)
        items = [(p["name"], p["value"], True)]
        test = m.get("test") or {}
        items += [(k, v, False) for k, v in test.items() if k != p["name"] and isinstance(v, (int, float))][:11]
        for i, (k, v, main) in enumerate(items):
            f, lay = panel(margin=10, spacing=2)
            lay.addWidget(label(k.replace("_", " "), "faint"))
            val = label(fmt(v), "tileValue")
            if not main:
                val.setStyleSheet(f"color: {T.TEXT}; font-size: 13pt;")
            lay.addWidget(val)
            self.tiles.addWidget(f, i // 6, i % 6)
        self.result_warn.setText("  ".join(m.get("warnings") or []))
        self.result_warn.setVisible(bool(m.get("warnings")))
        clear(self.chart_grid)
        for i, c in enumerate(charts):
            self.chart_grid.addWidget(Chart(c, base=d), i // 2, i % 2)

    # ---- 06 predict -------------------------------------------------------------------
    def _page_predict(self):
        pg = Page("Predict", "Run the selected model on new data.")
        self.pred_title = label("", "h2", wrap=True)
        pg.root.addWidget(self.pred_title)
        row = QHBoxLayout()
        self.pred_input = QLineEdit()
        self.pred_input.setPlaceholderText("choose new data")
        self.pred_pick = QPushButton("File")
        self.pred_pick.clicked.connect(lambda: self.pick_predict(False))
        self.pred_pick_dir = QPushButton("Folder")
        self.pred_pick_dir.clicked.connect(lambda: self.pick_predict(True))
        self.pred_prompt = QLineEdit()
        self.pred_prompt.setPlaceholderText("type a prompt")
        self.pred_steps = self._spin(dict(type="int", min=1, max=100000, default=12))
        self.pred_steps_label = label("future steps", "muted")
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setObjectName("primary")
        self.predict_btn.clicked.connect(lambda: self.run_predict())
        for w in (self.pred_input, self.pred_prompt, self.pred_pick, self.pred_pick_dir, self.pred_steps_label, self.pred_steps, self.predict_btn):
            row.addWidget(w, 1 if w in (self.pred_input, self.pred_prompt) else 0)
        pg.root.addLayout(row)
        self.single_label = label("Or type one row and predict it:", "muted")
        pg.root.addWidget(self.single_label)
        self.single = QTableWidget(1, 0)
        self.single.verticalHeader().hide()
        self.single.setMaximumHeight(68)
        pg.root.addWidget(self.single)
        self.single_btn = QPushButton("Predict this row")
        self.single_btn.clicked.connect(self.run_single)
        self.single_row = QHBoxLayout()
        self.single_row.addWidget(self.single_btn)
        self.single_row.addStretch(1)
        pg.root.addLayout(self.single_row)
        self.predict_msg = label("", "warn", wrap=True)
        pg.root.addWidget(self.predict_msg)
        self.pred_table = table(["result"], select=False)
        pg.root.addWidget(self.pred_table, 2)
        self.pred_term = Terminal()
        self.pred_term.setMaximumHeight(120)
        pg.root.addWidget(self.pred_term)
        return pg

    def _run_meta(self):
        return load_json(Path(self.run_dir) / "model" / "meta.json", {})

    def refresh_predict(self):
        m = self._run_meta()
        s = load_json(Path(self.run_dir) / "metrics.json", {})
        self.pred_title.setText(f"{s.get('name')}   {s.get('run_id')}")
        kind = m.get("kind")
        needs = NEEDS.get(kind, "rows")
        count = {"steps": "future steps", "samples": "images to generate", "episodes": "test episodes", "prompt": "max new tokens"}.get(needs)
        files = needs in ("rows", "images", "audio", "video", "graph")
        self.pred_input.setVisible(files)
        self.pred_pick.setVisible(files and needs != "graph")
        self.pred_pick_dir.setVisible(needs in ("images", "audio", "video", "graph"))
        self.pred_prompt.setVisible(needs == "prompt")
        self.pred_steps.setVisible(count is not None and not (needs == "episodes" and kind == "bandit"))
        self.pred_steps_label.setVisible(count is not None and not (needs == "episodes" and kind == "bandit"))
        if count:
            self.pred_steps_label.setText(count)
            self.pred_steps.setValue({"steps": 12, "samples": 8, "episodes": 10, "prompt": 80}[needs])
        hints = {"rows": "a table with the same columns as training", "images": "one image, or a folder of images", "audio": "one audio file, or a folder", "video": "one video, or a folder", "graph": "a folder with nodes.csv and edges.csv"}
        self.pred_input.setPlaceholderText(hints.get(needs, ""))
        rows_only = kind in ("supervised", "cluster", "reducer", "anomaly", "tab_ae")
        for w in (self.single, self.single_btn, self.single_label):
            w.setVisible(rows_only and m.get("inductive", True))
        if rows_only:
            names = [f["name"] for f in m["features"]]
            self.single.setColumnCount(len(names))
            self.single.setHorizontalHeaderLabels(names)
            self.single.setRowCount(1)
            self.single.resizeColumnsToContents()
        cant = m.get("inductive") is False
        self.predict_msg.setStyleSheet(f"color: {T.AMBER};")
        self.predict_msg.setText("This algorithm only labels the data it was fitted on, so it cannot score new data." if cant else "")
        self.predict_btn.setEnabled(not cant)

    def pick_predict(self, folder):
        f = choose_path(self, "New data", self.look_in.currentData(), folder=folder)
        if f:
            self.pred_input.setText(f)

    def _flavor_of_run(self):
        job = load_json(Path(self.run_dir) / "job.json", {})
        return "cuda" if job.get("device") == "cuda" and paths.env_python("cuda").exists() else "cpu"

    def _ensure_steps(self):
        job = load_json(Path(self.run_dir) / "job.json", {})
        flavor = self._flavor_of_run()
        g = self.hw.gpu if self.hw else None
        return flavor, runtime.setup_steps(registry.ALGOS[job["entry"]], flavor, g.backend if g else None)

    def run_predict(self, source=None):
        if self.runner.running:
            return
        m = self._run_meta()
        needs = NEEDS.get(m.get("kind"), "rows")
        stamp = time.strftime("%H%M%S")
        out = Path(self.run_dir) / f"predicted_{stamp}.csv"
        args = ["predict", self.run_dir, out]
        if needs == "prompt":
            prompt = self.pred_prompt.text().strip()
            if not prompt:
                self.predict_msg.setText("Type a prompt first.")
                return
            args += ["--prompt", prompt, "--steps", self.pred_steps.value()]
        elif needs in ("steps", "samples", "episodes"):
            args += ["--steps", self.pred_steps.value()]
            if needs == "samples":
                args += ["--out-dir", Path(self.run_dir) / f"predicted_{stamp}"]
        else:
            src = source or self.pred_input.text().strip()
            if not src:
                self.predict_msg.setText("Choose the new data first.")
                return
            args += ["--input", src]
            if needs == "images":
                args += ["--out-dir", Path(self.run_dir) / f"predicted_{stamp}"]
        self.predict_msg.setText("")
        self.job_mode = "predict"
        self.active_term = self.pred_term
        self.predict_btn.setEnabled(False)
        flavor, steps = self._ensure_steps()
        steps.append(runtime.Step("predict", runtime.worker_argv(flavor, *args), env={"CUDA_VISIBLE_DEVICES": ""}))
        self.pred_term.clear()
        self.runner.finished.disconnect()
        self.runner.finished.connect(self._job_finished)
        self.runner.start(steps, self.pred_term, self.on_event)

    def run_single(self):
        m = self._run_meta()
        values = [self.single.item(0, i).text() if self.single.item(0, i) else "" for i in range(self.single.columnCount())]
        path = Path(self.run_dir) / "input_row.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([x["name"] for x in m["features"]])
            w.writerow(values)
        self.run_predict(str(path))

    def show_prediction(self, ev):
        t = self.pred_table
        t.setColumnCount(len(ev["columns"]))
        t.setHorizontalHeaderLabels(ev["columns"])
        t.setRowCount(len(ev["head"]))
        for i, r in enumerate(ev["head"]):
            for j, v in enumerate(r):
                t.setItem(i, j, item(fmt(v) if isinstance(v, float) else v, T.ACCENT if ev["columns"][j] in ("prediction", "forecast", "anomaly", "cluster") else None))
        t.resizeColumnsToContents()
        self.predict_msg.setStyleSheet(f"color: {T.MUTED};")
        self.predict_msg.setText(f"{ev['rows']:,} predictions saved to {ev['out']}")

    # ---- 07 export ---------------------------------------------------------------------
    def _page_export(self):
        pg = Page("Export", "Package the model for other software and devices. ONNX is only kept if it reproduces the original model's output.")
        self.exp_title = label("", "h2", wrap=True)
        pg.root.addWidget(self.exp_title)
        split = QHBoxLayout()
        split.setSpacing(12)
        left = QVBoxLayout()
        box, lay = panel()
        lay.addWidget(label("Formats", "h2"))
        self.exp_onnx = QCheckBox("ONNX model, checked against the original")
        self.exp_api = QCheckBox("REST API bundle: serve.py, Dockerfile, requirements.txt")
        self.exp_onnx.setChecked(True)
        self.exp_api.setChecked(True)
        lay.addWidget(self.exp_onnx)
        lay.addWidget(self.exp_api)
        row = QHBoxLayout()
        self.exp_dir = QLineEdit()
        pick = QPushButton("Browse")
        pick.clicked.connect(self.pick_export)
        row.addWidget(self.exp_dir, 1)
        row.addWidget(pick)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("primary")
        self.export_btn.clicked.connect(self.run_export)
        self.exp_open = QPushButton("Open folder")
        self.exp_open.setEnabled(False)
        self.exp_open.clicked.connect(lambda: open_folder(self.exp_dir.text()))
        row.addWidget(self.export_btn)
        row.addWidget(self.exp_open)
        row.addStretch(1)
        lay.addLayout(row)
        left.addWidget(box)
        self.export_msg = label("", "muted", wrap=True)
        left.addWidget(self.export_msg)
        self.exp_term = Terminal()
        left.addWidget(self.exp_term, 1)
        split.addLayout(left, 5)
        self.target_table = table(["where it will run", "what to use", "status"], select=False)
        split.addWidget(self.target_table, 6)
        pg.root.addLayout(split, 1)
        return pg

    TARGETS = [("Any software, over HTTP", "REST API bundle", "api"), ("Python, Java, C#, C++, Node", "ONNX Runtime", "onnx"),
               ("Android and iOS apps", "ONNX Runtime Mobile", "onnx"), ("Web browser", "ONNX Runtime Web", "onnx"),
               ("Intel edge devices", "OpenVINO", "onnx"), ("NVIDIA Jetson", "ONNX Runtime with CUDA", "onnx")]

    def fill_targets(self, report=None):
        t = self.target_table
        t.setRowCount(len(self.TARGETS))
        for i, (where, what, needs) in enumerate(self.TARGETS):
            t.setItem(i, 0, item(where))
            t.setItem(i, 1, item(what, T.MUTED))
            if report is None:
                s, c = "-", T.FAINT
            elif (report.get(needs) or {}).get("status") == "ok":
                s, c = "FILE READY", T.ACCENT
            else:
                s, c = "NOT AVAILABLE", T.ROSE
            t.setItem(i, 2, item(s, c))
        t.resizeColumnsToContents()

    def refresh_export(self):
        s = load_json(Path(self.run_dir) / "metrics.json", {})
        m = self._run_meta()
        self.exp_title.setText(f"{s.get('name')}   {s.get('run_id')}")
        self.exp_dir.setText(str(Path.home() / "atlas-exports" / Path(self.run_dir).name))
        cant = m.get("inductive") is False
        self.export_btn.setEnabled(not cant)
        self.export_msg.setStyleSheet(f"color: {T.AMBER if cant else T.MUTED};")
        self.export_msg.setText("This algorithm cannot process new data, so there is nothing to deploy." if cant else "")
        self.exp_open.setEnabled(False)
        self.fill_targets()

    def pick_export(self):
        d = choose_path(self, "Export to", self.look_in.currentData(), folder=True)
        if d:
            self.exp_dir.setText(d)

    def run_export(self):
        if self.runner.running:
            return
        formats = [f for f, b in (("onnx", self.exp_onnx), ("api", self.exp_api)) if b.isChecked()]
        if not formats:
            self.export_msg.setText("Pick at least one format.")
            return
        self.job_mode = "export"
        self.active_term = self.exp_term
        self.export_btn.setEnabled(False)
        self.export_msg.setText("")
        self.exp_term.clear()
        flavor, steps = self._ensure_steps()
        steps.append(runtime.Step("export and verify", runtime.worker_argv(flavor, "export", self.run_dir, self.exp_dir.text(), ",".join(formats)), env={"CUDA_VISIBLE_DEVICES": ""}))
        self.runner.finished.disconnect()
        self.runner.finished.connect(self._job_finished)
        self.runner.start(steps, self.exp_term, self.on_event)

    def show_export(self, ev):
        if ev.get("status") == "blocked":
            self.export_msg.setText(ev["msg"])
            return
        rep = ev["report"]
        self.fill_targets(rep)
        self.exp_open.setEnabled(True)
        onnx = rep.get("onnx") or {}
        text = f"Saved to {ev['out']}."
        if onnx.get("status") == "ok":
            text += f" ONNX matched the original on {onnx['check'].get('rows')} held-out samples."
        elif onnx:
            text += " ONNX is not available for this model. Use the REST API bundle."
        self.export_msg.setStyleSheet(f"color: {T.ACCENT};")
        self.export_msg.setText(text)

    def closeEvent(self, e):
        if self.runner.running:
            self.runner.cancel()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Atlas Engine")
    app.setWindowIcon(QIcon(str(paths.PACKAGE_DATA / "icon.png")))
    T.load_fonts()
    app.setFont(T.font(10))
    app.setStyleSheet(T.STYLE)
    w = MainWindow()
    w.show()
    if "--selftest" in sys.argv:
        from . import selftest

        selftest.quick(app, w)
    elif "--selftest-train" in sys.argv:
        from . import selftest

        selftest.full(app, w)
    sys.exit(app.exec())
