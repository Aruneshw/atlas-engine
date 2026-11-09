"""Hardware detection and the resource plan for a job. Standard library only.

This runs inside the packaged app before any training runtime exists.
"""
from __future__ import annotations

import ctypes
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# CUDA builds run on any driver of the same major version, and PyTorch wheels carry their own runtime.
# Every 12.x driver therefore uses cu126: older 12.x builds stop at a PyTorch too old for the other libraries.
CUDA_BACKENDS = [(13.2, "cu132"), (13.0, "cu130"), (12.0, "cu126")]
GB = 1024 ** 3


@dataclass
class GPU:
    state: str                    # ok | none | no_driver | needs_reboot | error
    name: str = ""
    driver: str = ""
    cuda: float = 0.0
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    detail: str = ""

    @property
    def usable(self):
        return self.state == "ok"

    @property
    def backend(self):
        return next((b for v, b in CUDA_BACKENDS if self.cuda >= v), None)


@dataclass
class Hardware:
    cpu_threads: int
    ram_total: int
    ram_available: int
    disk_free: int
    gpu: GPU
    platform: str = field(default_factory=lambda: sys.platform)
    limits_ok: bool = False       # live CPU, RAM and GPU limits can be applied here
    limits_why: str = ""          # why not, in words for the interface


def _run(cmd, timeout=8):
    try:
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}   # keep bundled libraries away from system tools
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def _nvidia_on_bus() -> str:
    """Name of an NVIDIA adapter known to the OS, even with no working driver."""
    if sys.platform == "win32":
        _, out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"], 15)
    else:
        _, out = _run(["lspci"])
    m = re.search(r".*(?:NVIDIA).*", out, re.I)
    if not m:
        return ""
    line = m.group(0).strip()
    bracket = re.search(r"\[([^\]]*(?:GeForce|RTX|GTX|Quadro|Tesla)[^\]]*)\]", line)
    return bracket.group(1) if bracket else line.split(": ")[-1]


def detect_gpu() -> GPU:
    if shutil.which("nvidia-smi") is None:
        name = _nvidia_on_bus()
        if name:
            return GPU("no_driver", name=name, detail="NVIDIA card found, but its driver is not installed.")
        return GPU("none", detail="No NVIDIA GPU found.")
    code, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"])
    if code != 0 or "Failed to initialize" in out or "mismatch" in out.lower():
        name = _nvidia_on_bus()
        if "mismatch" in out.lower():
            return GPU("needs_reboot", name=name, detail="The NVIDIA driver was updated and the old one is still loaded. Reboot to use the GPU.")
        return GPU("error", name=name, detail=out.strip().splitlines()[0] if out.strip() else "nvidia-smi failed.")
    first = out.strip().splitlines()[0]
    try:
        name, driver, total, free = [x.strip() for x in first.split(",")]
        total, free = int(float(total)), int(float(free))
    except ValueError:
        return GPU("error", detail=f"Could not read nvidia-smi output: {first}")
    _, head = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", head)
    cuda = float(".".join(m.group(1).split(".")[:2])) if m else 0.0
    if not any(cuda >= v for v, _ in CUDA_BACKENDS):
        return GPU("error", name=name, driver=driver, cuda=cuda, vram_total_mb=total, vram_free_mb=free,
                   detail=f"Driver only supports CUDA {cuda}; version 12 or newer is needed. Update the NVIDIA driver.")
    return GPU("ok", name=name, driver=driver, cuda=cuda, vram_total_mb=total, vram_free_mb=free)


def memory() -> tuple[int, int]:
    """(total, available) RAM in bytes."""
    if sys.platform == "win32":
        class S(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong), ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong),
                        ("ex", ctypes.c_ulonglong)]

        s = S()
        s.len = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.total, s.avail
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k] = int(v.split()[0]) * 1024
    return info["MemTotal"], info.get("MemAvailable", info["MemFree"])


def _disk_probe(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


_limit_support = None


def limit_support() -> tuple[bool, str]:
    """Whether this machine can apply live limits: (yes or no, why not). Probed once."""
    global _limit_support
    if _limit_support is None:
        _limit_support = _probe_limits()
    return _limit_support


def _probe_limits() -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, "Live limits need Linux with systemd."
    if shutil.which("systemd-run") is None:
        return False, "Live limits need systemd, which this system does not use."
    uid = os.getuid()
    try:
        have = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers").read_text().split()
    except OSError:
        return False, "This system does not give your user its own control groups."
    if not {"cpu", "memory"} <= set(have):
        return False, "This system does not let your user limit CPU and memory."
    code, _ = _run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "true"])
    return (True, "") if code == 0 else (False, "Could not start a limited process on this system.")


def detect(data_dir: Path) -> Hardware:
    total, avail = memory()
    ok, why = limit_support()
    return Hardware(cpu_threads=os.cpu_count() or 1, ram_total=total, ram_available=avail,
                    disk_free=shutil.disk_usage(_disk_probe(data_dir)).free, gpu=detect_gpu(), limits_ok=ok, limits_why=why)


# Rough install footprint in GB of each runtime group.
GROUP_DISK = {"core": 1.2, "formats": 0.2, "boost": 0.5, "catboost": 0.3, "stats": 0.1, "pyod": 0.2, "umap": 0.1, "vision": 0.1, "text": 0.5, "peft": 0.05,
              "timm": 0.05, "ultra": 0.6, "smp": 0.05, "audio": 0.02, "video": 0.2, "rl": 0.2, "graph": 0.1}
# Starting batch size: CPU, GPU under 4 GB, under 8 GB, 8 GB or more.
BATCHES = {"image_classification": (8, 16, 32, 64), "object_detection": (2, 4, 8, 16), "semantic_segmentation": (4, 8, 16, 32), "video_classification": (2, 4, 8, 16),
           "image_generation": (16, 32, 64, 128), "representation": (16, 32, 64, 128), "text_classification": (8, 16, 32, 64), "text_generation": (4, 8, 16, 32)}
TORCH_DISK = {"cpu": 0.8, "cuda": 6.0}


@dataclass
class Plan:
    device: str
    flavor: str
    threads: int
    batch_size: int
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    blockers: list = field(default_factory=list)


def estimate_ram(profile, algo, task) -> int:
    """Rough peak bytes for the job, enough to catch clearly impossible requests."""
    rows, cols = profile.get("rows", 0), max(profile.get("cols", 1), 1)
    base = 600 * 1024 ** 2                                     # interpreter and libraries
    matrix = rows * cols * 8 * 6
    limit = algo.get("max_rows")
    if limit and rows > 0:                                     # kernel and distance methods grow with n squared
        matrix = max(matrix, min(rows, 200_000) ** 2 * 8)
    if "torch" in algo["groups"]:
        base += 1500 * 1024 ** 2
    return base + matrix


def make_plan(hw: Hardware, algo: dict, task: str, profile: dict, choice: str = "auto", installed: set | None = None,
              batch: int = 0, limits=None) -> Plan:
    """Decide where and how a job runs. `choice` is auto, cpu or gpu."""
    gpu = hw.gpu
    device, notes, warns, block = "cpu", [], [], []
    want_gpu = choice == "gpu" or (choice == "auto" and algo["gpu"])
    if want_gpu and not algo["gpu"]:
        warns.append("This algorithm has no GPU support: running on CPU.")
    elif want_gpu and gpu.usable:
        device = "cuda"
        notes.append(f"GPU: {gpu.name}, {gpu.vram_free_mb} MB free, CUDA {gpu.cuda}")
    elif want_gpu:
        warns.append(f"GPU unavailable: {gpu.detail} Running on CPU.")
    deep = "torch" in algo["groups"]
    if deep and device == "cpu" and task in ("image_classification", "object_detection", "semantic_segmentation", "video_classification", "text_generation", "image_generation"):
        warns.append("This kind of model is slow on CPU: use few epochs or a small variant.")

    threads = max(1, hw.cpu_threads - 2 if hw.cpu_threads > 4 else hw.cpu_threads - 1)
    if limits and limits.cpu_pct < 100:     # measured: threads beyond the limit waste it spinning, and ran 4x slower
        threads = min(threads, max(1, math.ceil(hw.cpu_threads * limits.cpu_pct / 100)))
    vram_free = min(gpu.vram_free_mb, int(limits.vram_gb * 1024)) if limits and limits.vram_gb else gpu.vram_free_mb
    size = batch
    if not size:
        size = 64
        if task in BATCHES:
            cpu, small, mid, big = BATCHES[task]
            size = cpu if device == "cpu" else small if vram_free < 4000 else mid if vram_free < 8000 else big
        elif device == "cuda" and vram_free < 2000:
            size = 32
    if deep:
        notes.append(f"Batch size {size}")

    need = estimate_ram(profile, algo, task)
    notes.append(f"Estimated RAM {need / GB:.1f} GB of {hw.ram_available / GB:.1f} GB available")
    if limits and limits.ram_gb and need > limits.ram_gb * GB:
        warns.append(f"This job may need about {need / GB:.1f} GB and your limit is {limits.ram_gb:.1f} GB. It will keep running, but expect it to take much longer.")
    if need > hw.ram_available:
        limit = algo.get("max_rows")
        if limit and profile.get("rows", 0) > limit:
            block.append(f"{profile['rows']:,} rows needs about {need / GB:.0f} GB RAM with this algorithm (practical limit {limit:,} rows). "
                         "Pick a tree or linear model, or use fewer rows.")
        else:
            warns.append("This job may use more RAM than is free: close other applications if it fails.")
    elif algo.get("max_rows") and profile.get("rows", 0) > algo["max_rows"]:
        warns.append(f"Above this algorithm's practical limit of {algo['max_rows']:,} rows: training may be very slow.")

    flavor = "cuda" if device == "cuda" else "cpu"
    missing = [g for g in algo["groups"] if g not in (installed or set())]
    disk = sum(TORCH_DISK[flavor] if g == "torch" else GROUP_DISK.get(g, 0.2) for g in missing)
    if disk * GB > hw.disk_free:
        block.append(f"Needs {disk:.1f} GB disk for its libraries; only {hw.disk_free / GB:.1f} GB is free.")
    elif missing:
        notes.append(f"First-time download: {', '.join(missing)} (about {disk:.1f} GB)")
    return Plan(device, flavor, threads, size, notes, warns, block)
