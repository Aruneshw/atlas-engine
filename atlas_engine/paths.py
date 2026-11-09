"""Where the app keeps things, and where its bundled files are when frozen."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
PACKAGE_DATA = BUNDLE / "atlas_engine" / "data"


def data_dir() -> Path:
    """Runtimes live here. It must be an internal disk: exFAT cannot hold Python environments."""
    override = os.environ.get("ATLAS_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AtlasEngine"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "atlas-engine"


def env_dir(flavor: str) -> Path:
    return data_dir() / "envs" / flavor


def env_python(flavor: str) -> Path:
    d = env_dir(flavor)
    return d / "Scripts" / "python.exe" if sys.platform == "win32" else d / "bin" / "python"


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(json.dumps(s, indent=1))


def runs_dir() -> Path:
    """Run history can move to an external drive through Settings."""
    custom = load_settings().get("runs_dir")
    d = Path(custom) if custom else data_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def uv_path() -> str | None:
    name = "uv.exe" if sys.platform == "win32" else "uv"
    bundled = BUNDLE / "bin" / name
    return str(bundled) if bundled.exists() else shutil.which("uv")


def worker_root() -> Path:
    """Folder holding the atlas_worker package. Separate when frozen, so the runtime sees only that."""
    return BUNDLE / "worker_src" if FROZEN else BUNDLE


def worker_env() -> dict:
    """Environment for worker processes: find the worker package, ignore any outer Python setup."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "LD_LIBRARY_PATH", "QT_PLUGIN_PATH", "VIRTUAL_ENV")}
    env["PYTHONPATH"] = str(worker_root())
    env["PYTHONUNBUFFERED"] = "1"
    return env
