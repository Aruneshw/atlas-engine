"""Live CPU, RAM and GPU limits for one training process.

A limit slows a job down. It never stops one. On Linux the job runs in its own cgroup scope:

  cpu.max        caps its CPU share, and can change while it runs
  memory.high    makes the kernel throttle allocations above the limit instead of killing the job.
                 With no swap the kernel has nowhere to put the excess and a throttled job all but stops,
                 so when that is seen the limit is raised in small steps: slower, but it finishes.
  cgroup.freeze  pauses it when the machine is about to run out of memory, and spaces out GPU work

The scope starts with the limits already set, so the first instruction runs under them. After that the
files are written directly, which needs no extra process and takes effect at once.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

GB = 1024 ** 3
MB = 1024 ** 2
CG_ROOT = Path("/sys/fs/cgroup")
CPU_PERIOD = 100000
TICK_MS = 500
GPU_CYCLE_MS = 250           # short, so nvidia-smi's samples average over whole cycles rather than catching one phase
MIN_DUTY = 0.05
MIN_HIGH = 64 * MB             # below this a Python process cannot make progress at all
STALL_TICKS = 2                # consecutive throttled, idle ticks before the limit gives way
MIN_SWAP_ROOM = 256 * MB       # less free swap than this counts as having none


@dataclass
class Limits:
    cpu_pct: int = 100         # share of the whole machine; 100 means no limit
    ram_gb: float = 0.0        # 0 means no limit
    gpu_pct: int = 100         # share of GPU time; 100 means no limit
    vram_gb: float = 0.0       # 0 means no limit; chosen before a run starts

    @classmethod
    def load(cls, data):
        known = {k: data[k] for k in asdict(cls()) if k in (data or {})}
        return cls(**known)


def memory_floor(total: int) -> int:
    """Memory kept free for the rest of the machine."""
    return max(300 * MB, int(total * 0.04))


def ceiling(job_bytes: int, available: int, total: int, user_limit: int) -> int:
    """The memory.high value: the user's limit, or lower when the machine is short of memory."""
    safe = job_bytes + available - memory_floor(total)
    return max(MIN_HIGH, min(safe, user_limit) if user_limit else safe)


def relax_step(user_limit: int) -> int:
    """How much a stalled job's limit rises at a time."""
    return max(32 * MB, user_limit // 10)


def next_duty(duty: float, smoothed_use: float, limit: int) -> float:
    """Share of each GPU cycle the job may run. Moves a little each second, so one noisy reading cannot swing it."""
    if limit >= 100:
        return 1.0
    return min(1.0, max(MIN_DUTY, duty + 0.25 * (limit - smoothed_use) / 100))


def wrap(argv: list, unit: str, lim: Limits, cpus: int) -> list:
    """The command that starts `argv` inside a new scope, with the limits already in force."""
    props = ["--property=CPUAccounting=yes", "--property=MemoryAccounting=yes"]
    if lim.cpu_pct < 100:
        props.append(f"--property=CPUQuota={lim.cpu_pct * cpus}%")
    if lim.ram_gb:
        props.append(f"--property=MemoryHigh={int(lim.ram_gb * GB)}")
    return [shutil.which("systemd-run"), "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", *props, "--", *map(str, argv)]


def _flat(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def _meminfo() -> tuple[int, int, int]:
    """(total RAM, available RAM, free swap) in bytes."""
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k] = int(v.split()[0]) * 1024
    return info["MemTotal"], info.get("MemAvailable", info["MemFree"]), info.get("SwapFree", 0)


class Scope:
    """Reads and writes one job's cgroup. Every method tolerates the job exiting at any moment."""

    def __init__(self, path: Path):
        self.path = path

    def _read(self, name):
        try:
            return (self.path / name).read_text()
        except OSError:
            return ""

    def _write(self, name, value):
        if self._read(name).strip() == str(value):
            return True
        try:
            (self.path / name).write_text(str(value))
            return True
        except OSError:
            return False

    def stat(self, name):
        return _flat(self._read(name))

    def set_cpu(self, pct, cpus):
        return self._write("cpu.max", "max 100000" if pct >= 100 else f"{int(pct / 100 * cpus * CPU_PERIOD)} {CPU_PERIOD}")

    def set_high(self, value):
        return self._write("memory.high", value)

    def freeze(self, on):
        return self._write("cgroup.freeze", "1" if on else "0")

    def memory(self):
        """(bytes of real working memory, bytes including file cache)."""
        current = self._read("memory.current").strip()
        current = int(current) if current.isdigit() else 0
        return max(0, current - self.stat("memory.stat").get("inactive_file", 0)), current


class Governor(QObject):
    """Applies Limits to the running job and reports what it is using."""

    updated = Signal(dict)

    def __init__(self, cpus: int, parent=None):
        super().__init__(parent)
        self.cpus, self.limits = cpus, Limits()
        self.enabled = False       # set from the hardware probe
        self.use_gpu = False       # the current job runs on the GPU
        self.scope, self.pid, self.unit = None, None, None
        self._n = 0
        self._cpu_last = None
        self._events = (0, 0)
        self.paused = False
        self.gpu_phase_off = False
        self.duty = 1.0
        self.ram_extra = 0         # how far a stalled job's RAM limit has been raised
        self._stalled = 0
        self.gpu_stats = None
        self.gpu_avg = None
        self._nvsmi = None
        self._nvbuf = ""
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(TICK_MS)
        self._tick_timer.timeout.connect(self._tick)
        self._duty_timer = QTimer(self)
        self._duty_timer.setSingleShot(True)
        self._duty_timer.timeout.connect(self._duty_phase)

    @property
    def attached(self):
        return self.pid is not None

    def prepare(self, argv: list) -> list:
        """Called just before the job starts. Returns the command to run."""
        if not self.enabled:
            return argv
        self._n += 1
        self.unit = f"atlas-train-{os.getpid()}-{self._n}"
        return wrap(argv, self.unit, self.limits, self.cpus)

    def attach(self, pid: int):
        self.pid, self.scope, self._cpu_last, self._events = pid, None, None, (0, 0)
        self.paused, self.gpu_phase_off, self.duty = False, False, max(MIN_DUTY, self.limits.gpu_pct / 100)
        self.gpu_stats = None
        self.gpu_avg = None
        self.ram_extra, self._stalled = 0, 0
        self._start_nvsmi()
        self._tick_timer.start()

    def detach(self):
        self._tick_timer.stop()
        self._duty_timer.stop()
        self._stop_nvsmi()
        self.pid, self.scope = None, None
        self.updated.emit({"state": "idle"})

    def set_limits(self, lim: Limits):
        self.limits = lim
        self.ram_extra, self._stalled = 0, 0
        if self.scope:
            self.scope.set_cpu(lim.cpu_pct, self.cpus)
            self._arm_gpu()

    # ---- GPU readings: one long-lived nvidia-smi rather than a new process every tick ---------
    def _start_nvsmi(self):
        exe = shutil.which("nvidia-smi")
        if not exe or self._nvsmi is not None:
            return
        p = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.remove("LD_LIBRARY_PATH")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(self._read_nvsmi)
        self._nvsmi = p
        p.start(exe, ["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits", "-l", "1"])

    def _stop_nvsmi(self):
        if self._nvsmi is not None:
            self._nvsmi.kill()
            self._nvsmi.waitForFinished(500)
            self._nvsmi.deleteLater()
            self._nvsmi = None

    def _read_nvsmi(self):
        self._nvbuf += bytes(self._nvsmi.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._nvbuf = self._nvbuf.split("\n")
        for line in lines:
            try:
                util, used, total = [float(x) for x in line.split(",")]
            except ValueError:
                continue
            self.gpu_stats = dict(util=util, used_mb=used, total_mb=total)
            if self.use_gpu and self.scope:
                self.gpu_avg = util if self.gpu_avg is None else self.gpu_avg * 0.75 + util * 0.25
                self.duty = next_duty(self.duty, self.gpu_avg, self.limits.gpu_pct)
                self._arm_gpu()

    # ---- GPU duty cycle ------------------------------------------------------------------------
    def _arm_gpu(self):
        """Start the on/off cycle if a GPU limit applies, or end it if not."""
        wanted = self.use_gpu and self.limits.gpu_pct < 100
        if wanted and not self._duty_timer.isActive():
            self.gpu_phase_off = False
            self._duty_timer.start(int(GPU_CYCLE_MS * self.duty))
        elif not wanted and (self._duty_timer.isActive() or self.gpu_phase_off):
            self._duty_timer.stop()
            self.gpu_phase_off = False
            self._apply_freeze()

    def _duty_phase(self):
        if not self.scope:
            return
        self.gpu_phase_off = not self.gpu_phase_off
        self._apply_freeze()
        share = (1 - self.duty) if self.gpu_phase_off else self.duty
        self._duty_timer.start(max(30, int(GPU_CYCLE_MS * share)))

    def _apply_freeze(self):
        if self.scope:
            self.scope.freeze(self.paused or self.gpu_phase_off)

    # ---- main loop -----------------------------------------------------------------------------
    def _find_scope(self):
        try:
            rel = Path(f"/proc/{self.pid}/cgroup").read_text().strip().split("::")[-1]
        except OSError:
            return
        if self.unit and rel.endswith(f"{self.unit}.scope"):
            self.scope = Scope(CG_ROOT / rel.lstrip("/"))
            self.scope.set_cpu(self.limits.cpu_pct, self.cpus)
            self._arm_gpu()

    def _tick(self):
        stats = {"state": "running", "gpu": self.gpu_stats, "limited": self.enabled}
        if self.enabled and self.scope is None:
            self._find_scope()
        if self.scope is not None:
            stats.update(self._measure())
        self.updated.emit(stats)

    def _measure(self):
        s, lim = self.scope, self.limits
        out = {}
        cpu = s.stat("cpu.stat")
        now = cpu.get("usage_usec", 0), time.monotonic()
        cores_busy = 1.0
        if self._cpu_last:
            spent, wall = now[0] - self._cpu_last[0], now[1] - self._cpu_last[1]
            if wall > 0:
                out["cpu_pct"] = spent / 1e6 / wall / self.cpus * 100
                cores_busy = spent / 1e6 / wall
        self._cpu_last = now

        working, _ = s.memory()
        total, available, swap_free = _meminfo()
        user_limit = int(lim.ram_gb * GB) if lim.ram_gb else 0
        rose = s.stat("memory.events").get("high", 0) > self._events[0]
        frozen = self.paused or self.gpu_phase_off
        self._stalled = self._stalled + 1 if rose and cores_busy < 0.1 and not frozen and swap_free < MIN_SWAP_ROOM else 0
        if user_limit and self._stalled >= STALL_TICKS:
            self.ram_extra += relax_step(user_limit)
            self._stalled = 0
        user_limit += self.ram_extra if user_limit else 0
        high = ceiling(working, available, total, user_limit)
        s.set_high(high)
        floor = memory_floor(total)
        if not self.paused and available < floor * 0.6 and working > 200 * MB:
            self.paused = True
        elif self.paused and available > floor * 1.2:
            self.paused = False
        self._apply_freeze()

        events = s.stat("memory.events").get("high", 0), cpu.get("nr_throttled", 0)
        out.update(ram_bytes=working, ram_total=total, available=available, paused=self.paused, ram_raised_to=user_limit if self.ram_extra else 0,
                   ram_slowed=events[0] > self._events[0], ram_guard=bool(not user_limit or high < user_limit),
                   cpu_slowed=events[1] > self._events[1], gpu_gated=self.gpu_phase_off or self.duty < 1.0 and self.use_gpu and lim.gpu_pct < 100)
        self._events = events
        return out
