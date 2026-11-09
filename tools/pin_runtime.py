#!/usr/bin/env python3
"""Rewrite atlas_worker/runtime_pins.py with the versions installed in a runtime env.

Usage: <env>/bin/python tools/pin_runtime.py
torch/torchvision stay unpinned at runtime on CUDA; see atlas_engine/runtime.py.
"""
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from atlas_worker.runtime_pins import PINS, PYTHON

body = "".join(f'    "{g}": {[f"{p.split("==")[0]}=={version(p.split("==")[0]).split("+")[0]}" for p in pkgs]!r},\n' for g, pkgs in PINS.items())
(ROOT / "atlas_worker/runtime_pins.py").write_text(
    '"""Exact package versions for each runtime group. Run tools/smoke_test.py after changing any."""\n\n'
    f"PINS = {{\n{body}}}\nPYTHON = \"{PYTHON}\"\n")
