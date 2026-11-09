"""Worker-to-GUI event stream: JSON lines prefixed with @@ on stdout.

Anything else a library prints to stdout goes to the terminal panel unchanged.
"""
import json
import sys
import threading
import time

PREFIX = "@@"
STAGES = ["validate", "prepare", "resources", "build", "train", "evaluate", "save"]
_peak = {"rss": 0.0}


def emit(event, /, **kw):
    sys.stdout.write(PREFIX + json.dumps({"t": event, **kw}, default=_jsonable) + "\n")
    sys.stdout.flush()


def _jsonable(v):
    if hasattr(v, "item"):
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    return str(v)


def stage(name):
    emit("stage", name=name)


def log(msg, level="info"):
    emit("log", msg=msg, level=level)


def start_monitor(interval=2.0):
    """Report this process tree's memory, plus GPU memory when torch has it loaded."""
    import psutil

    me = psutil.Process()

    def loop():
        while True:
            try:
                rss = me.memory_info().rss
                for c in me.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except psutil.Error:
                        pass
                mb = rss / 1048576
                _peak["rss"] = max(_peak["rss"], mb)
                gpu = None
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    gpu = torch.cuda.memory_allocated() / 1048576
                emit("res", rss_mb=round(mb), gpu_mb=None if gpu is None else round(gpu))
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


def peak_rss():
    return round(_peak["rss"])
