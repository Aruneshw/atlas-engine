"""Charts and small shared widgets. Drawn with QPainter so the app ships no plotting library."""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from . import theme as T


def label(text="", obj=None, wrap=False):
    w = QLabel(text)
    if obj:
        w.setObjectName(obj)
    w.setWordWrap(wrap)
    return w


def panel(layout_cls=QVBoxLayout, margin=12, spacing=8):
    f = QFrame()
    f.setObjectName("panel")
    lay = layout_cls(f)
    lay.setContentsMargins(margin, margin, margin, margin)
    lay.setSpacing(spacing)
    return f, lay


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return "n/a"
        if isinstance(v, int) or float(v).is_integer() and abs(v) < 1e6:
            return f"{int(v):,}"
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.1f}"
    return str(v)


def _ticks(lo, hi, n=4):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 5, 10) if s * mag >= raw)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + step * 1e-6:
        out.append(round(t, 10))
        t += step
    return out


def _short(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-3:
        return f"{v:.1e}"
    if a >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.3g}"


def _nums(seq):
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in seq]


class Chart(QWidget):
    """Renders one chart description from a run's charts.json."""

    def __init__(self, spec, base=None):
        super().__init__()
        self.spec = spec
        self.base = base
        self.pixmap = None
        if spec.get("type") == "image" and base is not None:
            self.pixmap = QPixmap(str(Path(base) / spec["path"]))
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), T.color(T.PANEL))
        p.setPen(T.color(T.LINE))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.setFont(T.font(10, True))
        p.setPen(T.color(T.ACCENT))
        p.drawText(QRectF(12, 8, self.width() - 24, 20), Qt.AlignmentFlag.AlignLeft, self.spec.get("title", ""))
        p.setFont(T.font(8.5))
        try:
            {"line": self._line, "scatter": self._scatter, "bars": self._bars, "heatmap": self._heat, "image": self._image, "text": self._text}[self.spec["type"]](p)
        except (ValueError, KeyError, IndexError, ZeroDivisionError):
            p.setPen(T.color(T.MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no data to chart")

    def _image(self, p):
        if self.pixmap is None or self.pixmap.isNull():
            raise ValueError("no image")
        area = QRectF(12, 34, self.width() - 24, self.height() - 44)
        scaled = self.pixmap.scaled(int(area.width()), int(area.height()), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        p.drawPixmap(int(area.left() + (area.width() - scaled.width()) / 2), int(area.top()), scaled)

    def _text(self, p):
        p.setPen(T.color(T.TEXT))
        p.drawText(QRectF(12, 34, self.width() - 24, self.height() - 44), Qt.TextFlag.TextWordWrap, str(self.spec.get("text", "")))

    def _area(self):
        return QRectF(58, 38, max(self.width() - 76, 10), max(self.height() - 74, 10))

    def _frame(self, p, a, xlo, xhi, ylo, yhi, xl="", yl=""):
        for t in _ticks(ylo, yhi):
            y = a.bottom() - (t - ylo) / ((yhi - ylo) or 1) * a.height()
            p.setPen(QPen(T.color(T.RAISED)))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            p.setPen(T.color(T.FAINT))
            p.drawText(QRectF(2, y - 8, 52, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _short(t))
        for t in _ticks(xlo, xhi, 5):
            x = a.left() + (t - xlo) / ((xhi - xlo) or 1) * a.width()
            p.setPen(T.color(T.FAINT))
            p.drawText(QRectF(x - 30, a.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, _short(t))
        if xl:
            p.setPen(T.color(T.MUTED))
            p.drawText(QRectF(a.left(), a.bottom() + 16, a.width(), 14), Qt.AlignmentFlag.AlignCenter, xl)
        if yl:
            p.save()
            p.translate(10, a.center().y())
            p.rotate(-90)
            p.setPen(T.color(T.MUTED))
            p.drawText(QRectF(-a.height() / 2, -7, a.height(), 14), Qt.AlignmentFlag.AlignCenter, yl)
            p.restore()

    def _legend(self, p, names, colors):
        fm = QFontMetrics(p.font())
        x = self.width() - 12
        for n, c in reversed(list(zip(names, colors))):
            w = fm.horizontalAdvance(n)
            x -= w
            p.setPen(T.color(T.MUTED))
            p.drawText(QPointF(x, 22), n)
            x -= 15
            p.fillRect(QRectF(x, 15, 10, 8), T.color(c))
            x -= 12

    def _line(self, p):
        s = self.spec
        series = []
        shared_x = s.get("x")
        for one in s["series"]:
            y = _nums(one["y"])
            x = _nums(one.get("x") or shared_x or list(range(len(y))))
            series.append((one["name"], list(zip(x, y))))
        xs = [a for _, pts in series for a, b in pts if b is not None]
        ys = [b for _, pts in series for a, b in pts if b is not None]
        if s.get("hline") is not None:
            ys.append(float(s["hline"]))
        xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
        if yhi == ylo:
            yhi += 1
        pad = (yhi - ylo) * 0.06
        ylo, yhi = ylo - pad, yhi + pad
        a = self._area()
        self._frame(p, a, xlo, xhi, ylo, yhi, s.get("xlabel", ""))
        fx = lambda v: a.left() + (v - xlo) / ((xhi - xlo) or 1) * a.width()
        fy = lambda v: a.bottom() - (v - ylo) / (yhi - ylo) * a.height()
        colors = []
        for i, (name, pts) in enumerate(series):
            c = T.SERIES[i % len(T.SERIES)]
            colors.append(c)
            p.setPen(QPen(T.color(c), 1.6))
            p.drawPolyline(QPolygonF([QPointF(fx(x), fy(y)) for x, y in pts if y is not None]))
        names = [n for n, _ in series]
        if s.get("hline") is not None:
            p.setPen(QPen(T.color(T.ROSE), 1, Qt.PenStyle.DashLine))
            y = fy(float(s["hline"]))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            names.append(s.get("hline_name", ""))
            colors.append(T.ROSE)
        if len(names) > 1:
            self._legend(p, names, colors)

    def _scatter(self, p):
        s = self.spec
        x, y = _nums(s["x"]), _nums(s["y"])
        xlo, xhi, ylo, yhi = min(x), max(x), min(y), max(y)
        if s.get("diagonal"):
            xlo = ylo = min(xlo, ylo)
            xhi = yhi = max(xhi, yhi)
        a = self._area()
        self._frame(p, a, xlo, xhi, ylo, yhi, s.get("xlabel", ""), s.get("ylabel", ""))
        fx = lambda v: a.left() + (v - xlo) / ((xhi - xlo) or 1) * a.width()
        fy = lambda v: a.bottom() - (v - ylo) / ((yhi - ylo) or 1) * a.height()
        if s.get("diagonal"):
            p.setPen(QPen(T.color(T.FAINT), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(a.left(), a.bottom()), QPointF(a.right(), a.top()))
        groups = s.get("groups")
        uniq = sorted(set(groups)) if groups else [None]
        p.setPen(Qt.PenStyle.NoPen)
        for i, (vx, vy) in enumerate(zip(x, y)):
            k = uniq.index(groups[i]) if groups else 0
            c = QColor(T.FAINT if groups and groups[i] == -1 else T.SERIES[k % len(T.SERIES)])
            c.setAlpha(190)
            p.setBrush(c)
            p.drawEllipse(QPointF(fx(vx), fy(vy)), 2.6, 2.6)

    def _bars(self, p):
        s = self.spec
        labels, values = s["labels"], [float(v) for v in s["values"]]
        n = len(labels)
        top, bottom = 36.0, self.height() - 12.0
        row = min((bottom - top) / max(n, 1), 26.0)
        fm = QFontMetrics(p.font())
        lw = min(max(fm.horizontalAdvance(str(l)) for l in labels) + 10, int(self.width() * 0.45))
        vmax = max(max(values), 1e-12)
        for i, (l, v) in enumerate(zip(labels, values)):
            y = top + i * row
            p.setPen(T.color(T.MUTED))
            p.drawText(QRectF(10, y, lw - 8, row), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fm.elidedText(str(l), Qt.TextElideMode.ElideMiddle, lw - 8))
            w = max(v / vmax, 0) * (self.width() - lw - 70)
            p.fillRect(QRectF(lw + 6, y + 3, w, row - 6), T.color(T.ACCENT_DIM))
            p.setPen(T.color(T.TEXT))
            p.drawText(QRectF(lw + 10 + w, y, 60, row), Qt.AlignmentFlag.AlignVCenter, _short(v))

    def _heat(self, p):
        s = self.spec
        z, labels = s["z"], s["labels"]
        n = len(z)
        lw = 74.0
        size = min((self.width() - lw - 18) / n, (self.height() - 66) / n)
        vmax = max(max(r) for r in z) or 1
        fm = QFontMetrics(p.font())
        x0, y0 = max(lw, (self.width() - n * size) / 2), 36.0
        base, full = T.color(T.RAISED), T.color(T.ACCENT)
        for i, row in enumerate(z):
            p.setPen(T.color(T.MUTED))
            p.drawText(QRectF(4, y0 + i * size, lw - 10, size), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fm.elidedText(str(labels[i]), Qt.TextElideMode.ElideRight, int(lw - 10)))
            p.drawText(QRectF(x0 + i * size, y0 + n * size + 3, size, 14), Qt.AlignmentFlag.AlignCenter, fm.elidedText(str(labels[i]), Qt.TextElideMode.ElideRight, int(size)))
            for j, v in enumerate(row):
                t = (v / vmax) ** 0.6
                c = QColor(int(base.red() + (full.red() - base.red()) * t), int(base.green() + (full.green() - base.green()) * t), int(base.blue() + (full.blue() - base.blue()) * t))
                r = QRectF(x0 + j * size + 1, y0 + i * size + 1, size - 2, size - 2)
                p.fillRect(r, c)
                if size > 22:
                    p.setPen(T.color(T.BG if t > 0.55 else T.TEXT))
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(v))
        p.setPen(T.color(T.FAINT))
        p.drawText(QRectF(0, y0 + n * size + 17, self.width(), 14), Qt.AlignmentFlag.AlignCenter, "predicted across, actual down")
