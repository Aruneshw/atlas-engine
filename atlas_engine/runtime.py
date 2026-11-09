"""Builds the list of fixed setup steps the terminal panel runs for a job.

Every argv is assembled here from pinned package names. Nothing typed by the user reaches a command.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from atlas_worker.runtime_pins import PINS, PYTHON, XGBOOST_CUDA12

from . import paths


@dataclass
class Step:
    title: str
    argv: list
    env: dict = field(default_factory=dict)
    stamp: tuple | None = None      # (flavor, group) recorded once this step succeeds
    governed: bool = False          # runs under the live CPU, RAM and GPU limits


def _stamp_file(flavor):
    return paths.env_dir(flavor) / ".atlas_groups.json"


def _old_driver_boost(group: str, flavor: str, backend: str | None) -> bool:
    return group == "boost" and flavor == "cuda" and backend == "cu126"


def _fingerprint(group: str, flavor: str = "cpu", backend: str | None = None) -> str:
    pins = " ".join(PINS[group]) + (f" {XGBOOST_CUDA12}" if _old_driver_boost(group, flavor, backend) else "")
    return hashlib.sha256(pins.encode()).hexdigest()[:12]


def _stamps(flavor: str) -> dict:
    try:
        data = json.loads(_stamp_file(flavor).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def installed_groups(flavor: str, backend: str | None = None) -> set:
    """Groups already installed at exactly the versions this build expects."""
    if not paths.env_python(flavor).exists():
        return set()
    return {g for g, fp in _stamps(flavor).items() if g in PINS and fp == _fingerprint(g, flavor, backend)}


def record_group(flavor: str, group: str, backend: str | None = None) -> None:
    stamps = _stamps(flavor)
    stamps[group] = _fingerprint(group, flavor, backend)
    _stamp_file(flavor).write_text(json.dumps(stamps))


def _packages(group: str, flavor: str, backend: str | None = None) -> list:
    pkgs = PINS[group]
    if _old_driver_boost(group, flavor, backend):
        pkgs = [XGBOOST_CUDA12 if p.startswith("xgboost==") else p for p in pkgs]
    if flavor == "cuda" and group in ("torch", "vision"):
        # Older drivers can only use older CUDA builds, so let uv choose the newest torch for that backend.
        pkgs = [p.split("==")[0] if p.startswith(("torch==", "torchvision==")) else p for p in pkgs]
    return pkgs


def setup_steps(algo: dict, flavor: str, backend: str | None) -> list[Step]:
    """Steps to make `flavor` runtime ready for this algorithm. Empty when already ready."""
    uv = paths.uv_path()
    if uv is None:
        raise FileNotFoundError("The bundled package manager (uv) is missing from this build.")
    steps = []
    py = paths.env_python(flavor)
    if not py.exists():
        steps.append(Step(f"create {flavor} runtime (Python {PYTHON})", [uv, "venv", "--python", PYTHON, str(paths.env_dir(flavor))]))
    have = installed_groups(flavor, backend)
    torch_flag = ["--torch-backend", "cpu" if flavor == "cpu" else (backend or "cu126")]
    for group in algo["groups"]:
        if group in have:
            continue
        extra = torch_flag if group in ("torch", "vision", "timm", "ultra", "smp", "rl", "graph", "text") else []
        steps.append(Step(f"install {group} libraries", [uv, "pip", "install", "--python", str(py), *extra, *_packages(group, flavor, backend)],
                          stamp=(flavor, group, backend)))
    return steps


def worker_argv(flavor: str, *args) -> list:
    return [str(paths.env_python(flavor)), "-u", "-m", "atlas_worker", *map(str, args)]
