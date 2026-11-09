"""The hardware controls: limit sliders, live meters and the GPU accelerator button."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget

from .widgets import label, panel

NO_LIMIT = "no limit"


class LimitRow(QWidget):
    """A named slider and its value. `text` turns a slider position into words."""

    changed = Signal(int)

    def __init__(self, name, lo, hi, step, text):
        super().__init__()
        self.text, self.step = text, step
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.name = label(name, "muted")
        self.name.setFixedWidth(40)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step)
        self.value_label = label("")
        self.value_label.setFixedWidth(62)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.name)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._moved)

    def _moved(self, v):
        snapped = round(v / self.step) * self.step
        if snapped != v:
            self.slider.setValue(snapped)
            return
        self.value_label.setText(self.text(v))
        self.changed.emit(v)

    def set_range(self, lo, hi):
        self.slider.blockSignals(True)
        self.slider.setRange(lo, hi)
        self.slider.blockSignals(False)
        self.value_label.setText(self.text(self.slider.value()))

    def set_value(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.value_label.setText(self.text(v))

    def value(self):
        return self.slider.value()

    def set_available(self, ok, why=""):
        self.setEnabled(ok)
        self.setToolTip("" if ok else why)


def half_gb(top):
    """Text for a slider counted in half gigabytes, whose top position means no limit."""
    return lambda v: NO_LIMIT if v >= top() else f"{v / 2:.1f} GB"


class HardwareControls(QWidget):
    """Used twice: stacked on the Configure page, and as live meters on the Train page."""

    limit_changed = Signal(str, int)
    accel_clicked = Signal(bool)

    def __init__(self, live: bool):
        super().__init__()
        self.live = live
        self.ram_top, self.vram_top = 32, 12
        self.rows = {"cpu": LimitRow("CPU", 5, 100, 5, lambda v: NO_LIMIT if v >= 100 else f"{v}%"),
                     "ram": LimitRow("RAM", 1, self.ram_top, 1, half_gb(lambda: self.ram_top)),
                     "gpu": LimitRow("GPU", 10, 100, 5, lambda v: NO_LIMIT if v >= 100 else f"{v}%"),
                     "vram": LimitRow("VRAM", 1, self.vram_top, 1, half_gb(lambda: self.vram_top))}
        for key, row in self.rows.items():
            row.changed.connect(lambda v, k=key: self.limit_changed.emit(k, v))
        self.accel = QPushButton("GPU accelerator: OFF")
        self.accel.setObjectName("accel")
        self.accel.setCheckable(True)
        self.accel.clicked.connect(self.accel_clicked)
        self.readings, self.bars = {}, {}
        self.status = label("", "warn", wrap=True)
        self.status.hide()
        if live:
            self._build_live()
        else:
            self._build_stacked()

    def _build_stacked(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)
        lay.addWidget(self.accel)
        lay.addWidget(label("Limits", "h2"))
        for row in self.rows.values():
            lay.addWidget(row)
        self.note = label("A limit slows training down. It never stops it.", "muted", wrap=True)
        lay.addWidget(self.note)

    def _build_live(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5)
        box, row = panel(QHBoxLayout, margin=8, spacing=16)
        for key in ("cpu", "ram", "gpu"):
            col = QVBoxLayout()
            col.setSpacing(3)
            reading = label("-", "muted")
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFixedHeight(6)
            bar.setTextVisible(False)
            col.addWidget(reading)
            col.addWidget(bar)
            col.addWidget(self.rows[key])
            self.readings[key], self.bars[key] = reading, bar
            row.addLayout(col, 1)
        self.rows["vram"].hide()
        self.accel.setFixedWidth(216)
        row.addWidget(self.accel)
        outer.addWidget(box)
        outer.addWidget(self.status)

    def show_reading(self, key, text, fraction=0.0, slowed=False):
        self.readings[key].setText(text)
        self.readings[key].setObjectName("warn" if slowed else "muted")
        self.readings[key].style().unpolish(self.readings[key])
        self.readings[key].style().polish(self.readings[key])
        self.bars[key].setValue(int(min(1.0, max(0.0, fraction)) * 1000))

    def show_status(self, text):
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def set_accel(self, visible, enabled, on, tip):
        self.accel.blockSignals(True)
        self.accel.setChecked(on)
        self.accel.blockSignals(False)
        self.accel.setText(f"GPU accelerator: {'ON' if on else 'OFF'}")
        self.accel.setVisible(visible)
        self.accel.setEnabled(enabled)
        self.accel.setToolTip(tip)

    def set_ranges(self, ram_gb, vram_gb):
        self.ram_top, self.vram_top = max(2, int(round(ram_gb * 2))), max(2, int(round(vram_gb * 2)))
        self.rows["ram"].set_range(1, self.ram_top)
        self.rows["vram"].set_range(1, self.vram_top)
