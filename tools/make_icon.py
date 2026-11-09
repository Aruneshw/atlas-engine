#!/usr/bin/env python3
"""Draw the app icon. Usage: python3 tools/make_icon.py  (writes packaging/ and atlas_engine/data/)."""
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

app = QApplication([])
S = 256
img = QImage(S, S, QImage.Format.Format_ARGB32)
img.fill(Qt.GlobalColor.transparent)
p = QPainter(img)
p.setRenderHint(QPainter.RenderHint.Antialiasing)
p.setBrush(QColor("#0d120f"))
p.setPen(QPen(QColor("#2a3630"), 6))
p.drawRoundedRect(QRectF(8, 8, S - 16, S - 16), 34, 34)
pen = QPen(QColor("#a9e4bb"), 20, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
p.setPen(pen)
p.drawLine(58, 192, 104, 64)          # A, left leg
p.drawLine(104, 64, 150, 192)         # A, right leg
p.drawLine(77, 146, 131, 146)         # A, bar
pen.setColor(QColor("#6fa583"))
p.setPen(pen)
p.drawLine(168, 192, 206, 64)         # slash
p.end()
for path in (ROOT / "packaging/atlas-engine.png", ROOT / "atlas_engine/data/icon.png"):
    img.save(str(path))
