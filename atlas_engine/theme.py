"""Pale phosphor terminal theme: soft desaturated greens on near-black, no neon."""
from PySide6.QtGui import QColor, QFont, QFontDatabase

BG = "#0d120f"
PANEL = "#131a16"
RAISED = "#1a231e"
LINE = "#2a3630"
TEXT = "#d3e4d8"
MUTED = "#8ba394"
FAINT = "#5f7468"
ACCENT = "#a9e4bb"
ACCENT_DIM = "#6fa583"
CYAN = "#a8d8e0"
AMBER = "#e8d59a"
ROSE = "#e8aaa5"
VIOLET = "#c5b8e6"

SERIES = [ACCENT, CYAN, AMBER, ROSE, VIOLET, "#b5cf9e", "#9fbfe8", "#e0bfa0"]
FAMILIES = ["JetBrains Mono", "Cascadia Mono", "Consolas", "Hack", "DejaVu Sans Mono", "Liberation Mono", "monospace"]


def load_fonts():
    """Register the bundled typeface so the app looks the same on every system."""
    from . import paths

    for f in sorted((paths.PACKAGE_DATA / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(f))


def font(size=10, bold=False) -> QFont:
    f = QFont()
    f.setFamilies([x for x in FAMILIES])
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setFixedPitch(True)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def color(hex_):
    return QColor(hex_)


STYLE = f"""
* {{ color: {TEXT}; outline: 0; }}
QWidget {{ background: {BG}; }}
QFrame#panel, QWidget#panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 3px; }}
QFrame#panel QLabel, QFrame#panel QCheckBox {{ background: transparent; }}
QLabel {{ background: transparent; }}
QLabel#h1 {{ color: {ACCENT}; font-size: 16pt; font-weight: 700; }}
QLabel#h2 {{ color: {ACCENT}; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#faint {{ color: {FAINT}; }}
QLabel#warn {{ color: {AMBER}; }}
QLabel#err {{ color: {ROSE}; }}
QLabel#ok {{ color: {ACCENT}; }}
QLabel#tileValue {{ color: {ACCENT}; font-size: 17pt; font-weight: 700; }}
QLabel#code {{ background: {BG}; border: 1px solid {LINE}; padding: 8px; color: {CYAN}; }}
QFrame#topbar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QFrame#rail {{ background: {PANEL}; border-right: 1px solid {LINE}; }}

QPushButton {{ background: {RAISED}; border: 1px solid {LINE}; padding: 7px 14px; border-radius: 2px; }}
QPushButton:hover {{ border-color: {ACCENT_DIM}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:disabled {{ color: {FAINT}; border-color: {RAISED}; background: {PANEL}; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: {BG}; border: 1px solid {ACCENT}; font-weight: 700; }}
QPushButton#primary:hover {{ background: #bdeccb; color: {BG}; }}
QPushButton#primary:disabled {{ background: {RAISED}; color: {FAINT}; border-color: {LINE}; }}
QPushButton#danger {{ color: {ROSE}; border-color: #5d4442; }}
QPushButton#danger:hover {{ color: {ROSE}; border-color: {ROSE}; }}
QPushButton#danger:disabled {{ color: {FAINT}; border-color: {RAISED}; background: {PANEL}; }}
QPushButton#step {{ text-align: left; background: transparent; border: none; border-left: 2px solid transparent; padding: 9px 14px; color: {MUTED}; }}
QPushButton#step:hover {{ color: {TEXT}; }}
QPushButton#step:checked {{ color: {ACCENT}; border-left: 2px solid {ACCENT}; background: {RAISED}; }}
QPushButton#step:disabled {{ color: {FAINT}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BG}; border: 1px solid {LINE}; padding: 5px 7px; border-radius: 2px; selection-background-color: {ACCENT_DIM}; selection-color: {BG}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {LINE}; selection-background-color: {ACCENT_DIM}; selection-color: {BG}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {ACCENT_DIM}; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; }}

QTableWidget, QListWidget, QTreeWidget, QTreeView, QListView {{ background: {PANEL}; border: 1px solid {LINE}; gridline-color: {RAISED}; alternate-background-color: {RAISED};
    selection-background-color: #2c4235; selection-color: {ACCENT}; }}
QHeaderView::section {{ background: {RAISED}; color: {MUTED}; border: none; border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: 5px 7px; }}
QTableCornerButton::section {{ background: {RAISED}; border: none; }}
QPlainTextEdit#terminal {{ background: #090d0b; border: 1px solid {LINE}; color: {TEXT}; padding: 8px; selection-background-color: {ACCENT_DIM}; selection-color: {BG}; }}
QProgressBar {{ background: {BG}; border: 1px solid {LINE}; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT_DIM}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 28px; border-radius: 3px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {LINE}; min-width: 28px; border-radius: 3px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; height: 0; width: 0; }}
QScrollArea {{ border: none; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {ACCENT_DIM}; padding: 4px; }}
QSplitter::handle {{ background: {LINE}; }}
"""
