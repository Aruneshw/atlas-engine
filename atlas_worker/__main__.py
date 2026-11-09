"""Worker entry point. The GUI runs this inside a managed runtime, one job per process.

Commands:  train <job.json> | predict <run_dir> <out.csv> [--input FILE|DIR] [--steps N]
           export <run_dir> <out_dir> <formats> | selfcheck | convert <in> <out.csv>
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path


NO_TABLE = {"vision": ("vision", "run_image"), "detect_tv": ("detect", "run_torchvision"), "detect_ultra": ("detect", "run_ultralytics"), "segment": ("segment", "run"),
            "audio": ("media", "run_audio"), "video": ("media", "run_video"), "gen": ("gen", "run_generative"), "selfsup": ("gen", "run_selfsup"),
            "bandit": ("rl", "run_bandit"), "rl_table": ("rl", "run_tabular"), "sb3": ("rl", "run_sb3"), "graph": ("relational", "run_graph")}
TABLE_ONLY = {"text_sk": ("nlp", "run_text_sk"), "hf_text": ("nlp", "run_hf_text"), "llm": ("nlp", "run_llm"), "recsys": ("relational", "run_recommender")}


def _load_fn(spec):
    import importlib

    return getattr(importlib.import_module(f"atlas_worker.{spec[0]}"), spec[1])


def cmd_train(job_path):
    from . import registry
    from .protocol import emit, log, peak_rss, stage, start_monitor

    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="A single label was found")     # reported properly by warn_user
    job = json.loads(Path(job_path).read_text())
    start_monitor()
    from .data import DataError, file_hash, load_table, validate

    current = "validate"
    try:
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "job.json").write_text(json.dumps(job, indent=1))
        algo = registry.get(job["entry"])
        task = job["task"]
        if task not in algo["tasks"]:
            raise DataError(f"Atlas entry #{job['entry']} does not support the task '{task}'.")
        impl = algo["tasks"][task]
        params = registry.resolve_params(algo, job.get("params"))
        ctx = {}
        t0 = time.time()
        stage("validate")
        adapter = impl["adapter"]
        if adapter in NO_TABLE:
            res = _load_fn(NO_TABLE[adapter])(job, algo, impl, params, ctx)
        elif adapter in TABLE_ONLY:
            res = _load_fn(TABLE_ONLY[adapter])(load_table(job["dataset"]["path"]), job, algo, impl, params, ctx)
        else:
            df = load_table(job["dataset"]["path"])
            df, features = validate(df, job)
            if algo.get("max_rows") and len(df) > algo["max_rows"]:
                log(f"{len(df)} rows is above this algorithm's practical limit of {algo['max_rows']}: "
                    "it may be slow or run out of memory", "warn")
            res = _dispatch(df, features, job, algo, impl, params, ctx, task, adapter)
        name, value, higher = res["primary"]
        import numpy as np
        if res.get("sample") is not None:
            import joblib
            joblib.dump(res["sample"], run_dir / "model" / "sample.joblib")
        summary = dict(run_id=job["run_id"], entry=job["entry"], name=job.get("name"), task=task, dataset=job["dataset"]["path"],
                       dataset_hash=file_hash(job["dataset"]["path"]) if job["dataset"].get("path") else job["dataset"].get("env"), primary=dict(name=name, value=value, higher_better=higher),
                       val=res.get("val"), test=res.get("test"), cv=res.get("cv"), params=res.get("params"), tuned=ctx.get("tuned"),
                       device=job.get("device"), threads=job.get("threads"), batch_size=job.get("batch_size"),
                       train_seconds=round(time.time() - t0, 2), peak_rss_mb=peak_rss(), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       libraries=_versions(algo), warnings=ctx.get("warnings"))
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=_num))
        (run_dir / "charts.json").write_text(json.dumps(res["charts"], default=_num))
        emit("done", run_dir=str(run_dir), summary=summary)
    except DataError as e:
        emit("error", kind="data", msg=str(e))
        sys.exit(2)
    except MemoryError:
        emit("error", kind="memory", msg="Ran out of RAM. Use fewer rows, a lighter algorithm, or close other applications.")
        sys.exit(3)
    except Exception as e:
        log(traceback.format_exc(), "error")
        emit("error", kind="bug", msg=f"{type(e).__name__}: {e}")
        sys.exit(4)


def _num(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    return str(v)


def _versions(algo):
    from importlib.metadata import PackageNotFoundError, version

    from .runtime_pins import PINS

    out = {}
    for g in algo["groups"]:
        for p in PINS[g]:
            name = p.split("==")[0]
            try:
                out[name] = version(name)
            except PackageNotFoundError:
                pass
    return out


def _dispatch(df, features, job, algo, impl, params, ctx, task, adapter):
    from . import forecast, tabular

    if adapter in ("seq", "tabae"):
        from . import deep

        if adapter == "tabae":
            return deep.run_tab_ae(df, features, job, algo, impl, params, ctx)
        if task == "forecasting":
            return deep.run_seq_forecast(df, features, job, algo, impl, params, ctx)
        if task == "seq_classification":
            return deep.run_seq_classification(df, features, job, algo, impl, params, ctx)
        return deep.run_seq_anomaly(df, features, job, algo, impl, params, ctx)
    if task == "forecasting":
        return forecast.run_forecast(df, features, job, algo, impl, params, ctx)
    fn = {"classification": tabular.run_supervised, "regression": tabular.run_supervised, "clustering": tabular.run_clustering,
          "dimreduction": tabular.run_reduction, "anomaly": tabular.run_anomaly}[task]
    return fn(df, features, job, algo, impl, params, ctx)


def cmd_predict(args):
    from .data import DataError, load_table
    from .predictor import Predictor
    from .protocol import emit

    run_dir, out = Path(args[0]), Path(args[1])
    opts = dict(zip(args[2::2], args[3::2]))
    try:
        p = Predictor(run_dir / "model")
        out_dir = opts.get("--out-dir")
        if p.needs in ("steps", "samples", "episodes"):
            res = p.predict(steps=int(opts.get("--steps", 0)) or None, out_dir=out_dir)
        elif p.needs == "prompt":
            res = p.predict(prompt=opts.get("--prompt"), steps=int(opts.get("--steps", 0)) or None)
        elif p.needs == "graph":
            res = p.predict(path=opts["--input"])
        elif p.needs in ("images", "audio", "video"):
            from .fileinfo import AUDIO_EXT, IMAGE_EXT, VIDEO_EXT

            ext = {"images": IMAGE_EXT, "audio": AUDIO_EXT, "video": VIDEO_EXT}[p.needs]
            src = Path(opts["--input"])
            files = sorted(f for f in src.rglob("*") if f.suffix.lower() in ext) if src.is_dir() else [src]
            if not files:
                raise DataError(f"No {p.needs} files found in that folder.")
            res = p.predict(images=[str(f) for f in files], out_dir=out_dir)
            if "file" not in res.columns and "image" not in res.columns:
                res.insert(0, "file", [str(f) for f in files])
        else:
            df = load_table(opts["--input"])
            res = p.predict(df=df)
            if len(res) == len(df) and p.kind not in ("seq_class", "seq_anomaly"):
                res = df.reset_index(drop=True).join(res.add_prefix("atlas_") if set(res.columns) & set(df.columns) else res)
        res.to_csv(out, index=False)
        emit("predicted", out=str(out), rows=len(res), head=json_safe(res.head(50)), columns=list(res.columns))
    except (DataError, KeyError) as e:
        emit("error", kind="data", msg=str(e))
        sys.exit(2)
    except Exception as e:
        print(traceback.format_exc())
        emit("error", kind="bug", msg=f"{type(e).__name__}: {e}")
        sys.exit(4)


def json_safe(df):
    return json.loads(df.to_json(orient="values"))


def cmd_export(args):
    import warnings

    warnings.filterwarnings("ignore")
    from .export import run_export
    from .protocol import emit

    try:
        run_export(args[0], args[1], args[2].split(","))
    except Exception as e:
        print(traceback.format_exc())
        emit("error", kind="bug", msg=f"{type(e).__name__}: {e}")
        sys.exit(4)


def cmd_selfcheck():
    from .protocol import emit

    info = dict(torch=None, cuda=False, gpu_name=None, cuda_version=None)
    try:
        import torch

        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(), cuda_version=torch.version.cuda)
        if info["cuda"]:
            torch.zeros(1).cuda()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception as e:
        info.update(cuda=False, error=f"{type(e).__name__}: {e}")
    emit("selfcheck", **info)


def cmd_convert(src, dst):
    from .data import DataError, load_table
    from .protocol import emit

    try:
        load_table(src).to_csv(dst, index=False)
    except DataError as e:
        emit("error", kind="data", msg=str(e))
        sys.exit(2)


def main():
    if os.name == "posix":
        try:
            os.setsid()            # own process group, so Cancel can stop every child at once
        except OSError:
            pass
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        cmd_train(args[0])
    elif cmd == "predict":
        cmd_predict(args)
    elif cmd == "export":
        cmd_export(args)
    elif cmd == "selfcheck":
        cmd_selfcheck()
    elif cmd == "convert":
        cmd_convert(*args)
    else:
        sys.exit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
