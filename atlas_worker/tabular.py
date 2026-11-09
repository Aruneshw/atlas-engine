"""Table-based training: supervised, clustering, dimensionality reduction, anomaly detection."""
from __future__ import annotations

import importlib
import inspect
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import registry
from .data import DataError, coerce, split_indices
from .metrics import anomaly_metrics, classification_metrics, regression_metrics
from .protocol import log, stage
from .simple import SIMPLE

MAX_CHART_POINTS = 400


def load_class(path):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _resolve(value):
    if isinstance(value, dict) and "$cls" in value:
        return _instantiate(load_class(value["$cls"]), dict(value["$kw"]))
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve(v) for v in value)
    return value


def _instantiate(cls, kw):
    accepted = inspect.signature(cls.__init__).parameters
    if "random_state" in accepted:
        kw.setdefault("random_state", 42)
    return cls(**{k: _resolve(v) for k, v in kw.items()})


def make_preprocessor(features, scale):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

    num = [f["name"] for f in features if f["type"] == "num"]
    cat = [f["name"] for f in features if f["type"] == "cat"]
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale == "standard":
        steps.append(("scale", StandardScaler()))
    elif scale == "minmax":
        steps.append(("scale", MinMaxScaler()))
    parts = []
    if num:
        parts.append(("num", Pipeline(steps), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def build_model(algo, impl, params, job):
    """Instantiate the estimator named by the registry, applying validated parameters."""
    cls = load_class(impl["cls"])
    kw = dict(impl.get("fixed", {}))
    kw.update({p["name"]: params[p["name"]] for p in algo["params"] if p["to"] == "model"})
    for p in algo["params"]:
        if p["type"] == "optional_int" and p["to"] == "model" and kw.get(p["name"]) == 0:
            kw[p["name"]] = None
    if impl.get("hidden_arg"):
        kw["hidden_layer_sizes"] = (kw.pop("hidden"),)
    accepted = inspect.signature(cls.__init__).parameters
    if "n_jobs" in accepted:
        kw.setdefault("n_jobs", job.get("threads", 1))
    if impl.get("gbm") == "xgb":
        kw.update(tree_method="hist", device="cuda" if job.get("device") == "cuda" else "cpu",
                  n_jobs=job.get("threads", 1))
    if impl.get("gbm") == "lgb":
        kw["n_jobs"] = job.get("threads", 1)
    if "thread_count" in accepted or impl["cls"].startswith("catboost"):
        kw["thread_count"] = job.get("threads", 1)
    return _instantiate(cls, kw)


def extra_steps(impl, params, n_features):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

    pre = impl.get("pre")
    if pre == "poly":
        return [("poly", PolynomialFeatures(degree=params["degree"], include_bias=False))]
    if pre == "pca":
        return [("pca", PCA(n_components=min(params["n_components"], n_features)))]
    if pre == "spline":
        return [("spline", SplineTransformer(n_knots=params["n_knots"]))]
    return []


def _pipeline(algo, impl, params, job, features, model):
    from sklearn.pipeline import Pipeline

    pre = make_preprocessor(features, algo["scale"])
    return Pipeline([("pre", pre), *extra_steps(impl, params, len(features)), ("model", model)])


def warn_user(ctx, msg):
    """A warning serious enough to show beside the results, not only in the log."""
    log(msg, "warn")
    ctx.setdefault("warnings", []).append(msg)


def _thin(n, limit=MAX_CHART_POINTS):
    return np.linspace(0, n - 1, min(n, limit)).astype(int)


def _clamp_components(params, algo, n_rows, n_cols):
    for p in algo["params"]:
        if p["name"] == "n_components" and params[p["name"]] > min(n_rows, n_cols):
            params[p["name"]] = min(n_rows, n_cols)
            log(f"n_components reduced to {params[p['name']]} to fit the data", "warn")


def _importance(pipe, top=15):
    """(names, values) of the strongest features, when the model exposes them."""
    model = pipe.named_steps["model"]
    try:
        names = list(pipe[:-1].get_feature_names_out())
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, float)
    elif hasattr(model, "coef_"):
        c = np.abs(np.asarray(model.coef_, float))
        vals = c.mean(0) if c.ndim > 1 and c.shape[0] > 1 else c.ravel()
    else:
        return None
    if len(vals) != len(names):
        return None
    order = np.argsort(vals)[::-1][:top]
    return [names[i] for i in order], [float(vals[i]) for i in order]


def _save(run_dir, meta, pipe=None):
    d = Path(run_dir) / "model"
    d.mkdir(parents=True, exist_ok=True)
    if pipe is not None:
        joblib.dump(pipe, d / "pipeline.joblib")
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


# ---- supervised ------------------------------------------------------------------
def run_supervised(df, features, job, algo, impl, params, ctx):
    task, target = job["task"], job["dataset"]["target"]
    stage("prepare")
    raw_y = df[target]
    classes = None
    if task == "classification":
        classes = sorted(raw_y.astype(str).unique().tolist())
        if len(classes) < 2:
            raise DataError("The target has only one class, so there is nothing to classify.")
        if len(classes) > 0.5 * len(df):
            raise DataError(f"Target has {len(classes)} distinct values in {len(df)} rows. "
                            "This looks like a regression target or an ID column.")
        lookup = {c: i for i, c in enumerate(classes)}
        y = raw_y.astype(str).map(lookup).to_numpy()
    else:
        y = pd.to_numeric(raw_y, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise DataError(f"Target '{target}' has non-numeric values, so regression cannot use it.")
        if impl.get("target") == "nonnegative" and (y < 0).any():
            raise DataError(f"This algorithm needs a non-negative target; '{target}' has negative values.")

    tr, va, te, chrono = split_indices(len(df), y, job)
    log(f"split: train {len(tr)}, validation {len(va)}, test {len(te)}" + (" (time-ordered)" if chrono else ""))
    if task == "classification":
        missing = set(y[tr]) ^ set(range(len(classes)))
        if missing:
            warn_user(ctx, f"{len(missing)} class(es) never appear in the training split, so the model cannot learn them.")
        counts = np.bincount(y[te], minlength=len(classes))
        for name, n in zip(classes, counts):
            if n < 5:
                warn_user(ctx, f"The test split has only {n} rows of class '{name}', so its metrics are unreliable."
                          + (" Use a larger test share or more data." if chrono else ""))
    X = coerce(df, features)
    log("preprocessing is fitted on the training split only")

    stage("build")
    _clamp_components(params, algo, len(tr), len(features))
    model = build_model(algo, impl, params, job)
    pipe = _pipeline(algo, impl, params, job, features, model)
    log(f"pipeline: {' > '.join(n for n, _ in pipe.steps)} ({type(model).__name__})")

    stage("train")
    cv_info = None
    folds, tune = int(job.get("cv") or 0), int(job.get("tune") or 0)
    splitter = None
    if folds > 1 or tune > 0:
        from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

        k = max(2, folds or 3)
        splitter = (TimeSeriesSplit(n_splits=k) if chrono else
                    StratifiedKFold(n_splits=k, shuffle=True, random_state=42) if task == "classification" else
                    KFold(n_splits=k, shuffle=True, random_state=42))
    scoring = "f1_macro" if task == "classification" else "r2"
    if tune > 0:
        pipe = _tuned(pipe, algo, tune, splitter, scoring, job, X.iloc[tr], y[tr], ctx)
    elif folds > 1:
        from sklearn.model_selection import cross_val_score

        log(f"cross-validating on {folds} folds...")
        scores = cross_val_score(pipe, X.iloc[tr], y[tr], cv=splitter, scoring=scoring)
        cv_info = {"scoring": scoring, "mean": float(scores.mean()), "std": float(scores.std()), "folds": [float(s) for s in scores]}
        log(f"cv {scoring}: {scores.mean():.4f} +/- {scores.std():.4f}")
    t0 = time.time()
    log("fitting final model...")
    pipe.fit(X.iloc[tr], y[tr])
    log(f"fit finished in {time.time() - t0:.2f}s")

    stage("evaluate")
    charts = []
    if task == "classification":
        val = classification_metrics(pipe, X.iloc[va], y[va], classes)
        test = classification_metrics(pipe, X.iloc[te], y[te], classes, charts=charts)
        primary = ("f1_macro", test["f1_macro"], True)
    else:
        val = regression_metrics(pipe.predict(X.iloc[va]), y[va])
        pred = np.ravel(pipe.predict(X.iloc[te]))
        test = regression_metrics(pred, y[te])
        primary = ("r2", test["r2"], True)
        i = _thin(len(te))
        charts.append(dict(type="scatter", title="Predicted vs actual (test)", x=y[te][i], y=pred[i],
                           xlabel="actual", ylabel="predicted", diagonal=True))
    imp = _importance(pipe)
    if imp:
        charts.append(dict(type="bars", title="Most influential features", labels=imp[0], values=imp[1]))

    stage("save")
    pred_col = np.ravel(pipe.predict(X.iloc[te]))
    out = df.iloc[te].copy()
    out["prediction"] = [classes[int(p)] for p in pred_col] if classes else pred_col
    out.to_csv(Path(job["run_dir"]) / "predictions.csv", index=False)
    meta = dict(kind="supervised", task=task, features=features, classes=classes, target=target)
    _save(job["run_dir"], meta, pipe)
    return dict(primary=primary, val=val, test=test, cv=cv_info, charts=charts, sample=X.iloc[te].head(32),
                params=params)


def _tuned(pipe, algo, n_iter, splitter, scoring, job, X, y, ctx):
    from sklearn.model_selection import RandomizedSearchCV

    grid = {}
    for p in algo["params"]:
        if not p.get("tune"):
            continue
        prefix = "model__" if p["to"] == "model" else next(n for n, _ in pipe.steps[1:-1]) + "__"
        values = [None if (p["type"] == "optional_int" and v == 0) else v for v in p["tune"]]
        grid[prefix + p["name"]] = values
    if not grid:
        log("this algorithm has no tunable grid, so tuning was skipped", "warn")
        return pipe
    n_total = np.prod([len(v) for v in grid.values()])
    log(f"tuning: {min(n_iter, int(n_total))} trials over {', '.join(grid)}")
    search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, int(n_total)), cv=splitter, scoring=scoring,
                                random_state=42, refit=False, n_jobs=1)
    search.fit(X, y)
    log(f"best {scoring}: {search.best_score_:.4f} with {search.best_params_}")
    ctx["tuned"] = {k: v for k, v in search.best_params_.items()}
    ctx["cv_best"] = float(search.best_score_)
    return pipe.set_params(**search.best_params_)


# ---- clustering / dimensionality reduction ----------------------------------------
def _fit_matrix(df, features, job, algo):
    X = coerce(df, features)
    pre = make_preprocessor(features, algo["scale"])
    return X, pre


def run_clustering(df, features, job, algo, impl, params, ctx):
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.pipeline import Pipeline

    stage("prepare")
    X, pre = _fit_matrix(df, features, job, algo)
    Xp = pre.fit_transform(X)
    log(f"fitting on all {len(X)} rows: clustering has no held-out labels")
    stage("build")
    model = build_model(algo, impl, params, job)
    stage("train")
    t0 = time.time()
    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(Xp)
    else:
        model.fit(Xp)
        labels = model.predict(Xp)
    log(f"fit finished in {time.time() - t0:.2f}s")
    stage("evaluate")
    uniq = [u for u in np.unique(labels) if u != -1]
    noise = float(np.mean(labels == -1))
    m = dict(clusters=len(uniq), noise_fraction=noise)
    if len(uniq) >= 2:
        mask = labels != -1
        sample = np.random.RandomState(42).choice(np.flatnonzero(mask), min(5000, mask.sum()), replace=False)
        m["silhouette"] = float(silhouette_score(Xp[sample], labels[sample]))
        m["davies_bouldin"] = float(davies_bouldin_score(Xp[mask], labels[mask]))
        m["calinski_harabasz"] = float(calinski_harabasz_score(Xp[mask], labels[mask]))
    else:
        log("found fewer than two clusters: try different parameters", "warn")
        m["silhouette"] = float("nan")
    stage("save")
    charts = _projection_chart(Xp, labels, "Clusters (first two PCA directions)")
    sizes = pd.Series(labels).value_counts().sort_index()
    charts.append(dict(type="bars", title="Cluster sizes", labels=[str(i) for i in sizes.index], values=sizes.tolist()))
    out = df.copy()
    out["cluster"] = labels
    out.to_csv(Path(job["run_dir"]) / "predictions.csv", index=False)
    inductive = impl.get("inductive", True) and hasattr(model, "predict")
    if not inductive:
        log(f"{type(model).__name__} cannot assign new rows to clusters, so it has no prediction or export path", "warn")
    pipe = Pipeline([("pre", pre), ("model", model)])
    _save(job["run_dir"], dict(kind="cluster", task=job["task"], features=features, inductive=inductive), pipe)
    return dict(primary=("silhouette", m["silhouette"], True), val=None, test=m, charts=charts, sample=X.head(32), params=params)


def _projection_chart(Xp, labels, title):
    from sklearn.decomposition import PCA

    Z = Xp[:, :2] if Xp.shape[1] <= 2 else PCA(n_components=2, random_state=42).fit_transform(Xp)
    if Z.shape[1] < 2:
        Z = np.c_[Z, np.zeros(len(Z))]
    i = _thin(len(Z))
    return [dict(type="scatter", title=title, x=Z[i, 0], y=Z[i, 1], groups=np.asarray(labels)[i] if labels is not None else None,
                 xlabel="component 1", ylabel="component 2")]


def run_reduction(df, features, job, algo, impl, params, ctx):
    from sklearn.manifold import trustworthiness
    from sklearn.pipeline import Pipeline

    stage("prepare")
    X, pre = _fit_matrix(df, features, job, algo)
    Xp = pre.fit_transform(X)
    stage("build")
    _clamp_components(params, algo, len(X), Xp.shape[1])
    model = build_model(algo, impl, params, job)
    stage("train")
    t0 = time.time()
    Z = model.fit_transform(Xp)
    log(f"fit finished in {time.time() - t0:.2f}s")
    stage("evaluate")
    n = min(2000, len(Xp))
    sample = np.random.RandomState(42).choice(len(Xp), n, replace=False)
    neighbours = min(5, max(1, n // 2 - 1))
    m = dict(trustworthiness=float(trustworthiness(Xp[sample], Z[sample], n_neighbors=neighbours)), output_dimensions=int(Z.shape[1]))
    if hasattr(model, "explained_variance_ratio_"):
        m["explained_variance"] = float(np.sum(model.explained_variance_ratio_))
    stage("save")
    charts = _projection_chart(Z, None, "Reduced data")
    if hasattr(model, "explained_variance_ratio_"):
        v = np.asarray(model.explained_variance_ratio_)
        charts.append(dict(type="bars", title="Explained variance", labels=[f"PC{i + 1}" for i in range(len(v))][:15], values=v[:15].tolist()))
    pd.concat([df.reset_index(drop=True), pd.DataFrame(Z, columns=[f"dim_{i + 1}" for i in range(Z.shape[1])])], axis=1).to_csv(
        Path(job["run_dir"]) / "predictions.csv", index=False)
    inductive = impl.get("inductive", True) and hasattr(model, "transform")
    if not inductive:
        log(f"{type(model).__name__} cannot transform new rows, so it has no prediction or export path", "warn")
    pipe = Pipeline([("pre", pre), ("model", model)])
    _save(job["run_dir"], dict(kind="reducer", task=job["task"], features=features, inductive=inductive), pipe)
    return dict(primary=("trustworthiness", m["trustworthiness"], True), val=None, test=m, charts=charts, sample=X.head(32), params=params)


# ---- anomaly -----------------------------------------------------------------------
class Scorer:
    """Unifies every table-based detector: `score(X)` is higher for more abnormal rows."""

    def __init__(self, est, method):
        self.est, self.method = est, method

    def fit(self, X):
        self.est.fit(X)
        return self

    def score(self, X):
        if self.method == "own":
            return self.est.score(X)
        if self.method == "neg_samples":
            return -self.est.score_samples(X)
        return self.est.decision_function(X)


def anomaly_labels(df, job, target):
    """Optional 0/1 labels. Normal class is the majority one."""
    if not target:
        return None
    s = df[target].astype(str)
    counts = s.value_counts()
    if len(counts) != 2:
        raise DataError(f"For anomaly detection the label column must have exactly 2 values; '{target}' has {len(counts)}.")
    normal = counts.index[0]
    log(f"label column '{target}': treating '{normal}' as normal and '{counts.index[1]}' as abnormal")
    return (s != normal).to_numpy().astype(int)


def anomaly_report(scores_te, scores_tr, y_te, contamination, df_te, job, ctx=None):
    thr = float(np.quantile(scores_tr, 1 - contamination))
    flags = scores_te > thr
    m = dict(threshold=thr, flagged_rate=float(flags.mean()))
    if y_te is not None:
        m.update(anomaly_metrics(scores_te, flags, y_te))
    i = _thin(len(scores_te))
    charts = [dict(type="line", title="Anomaly score (test)", series=[dict(name="score", y=scores_te[i])], hline=thr, hline_name="threshold")]
    out = df_te.copy()
    out["anomaly_score"] = scores_te
    out["anomaly"] = flags.astype(int)
    out.to_csv(Path(job["run_dir"]) / "predictions.csv", index=False)
    primary = ("pr_auc", m["pr_auc"], True) if y_te is not None and "pr_auc" in m else ("flagged_rate", m["flagged_rate"], False)
    return m, primary, charts, thr


def run_anomaly(df, features, job, algo, impl, params, ctx):
    from sklearn.pipeline import Pipeline

    target = job["dataset"].get("target")
    contamination = float(job.get("contamination", 0.05))
    stage("prepare")
    y = anomaly_labels(df, job, target)
    tr, va, te, chrono = split_indices(len(df), None, job)
    if y is not None:
        before = len(tr)
        tr = tr[y[tr] == 0]
        log(f"training on the {len(tr)} normal rows of {before} in the train split")
    X = coerce(df, features)
    pre = make_preprocessor(features, algo["scale"])
    stage("build")
    if impl["adapter"] == "simple":
        kw = {p["name"]: params[p["name"]] for p in algo["params"] if p["to"] == "model"}
        scorer = Scorer(SIMPLE[impl["cls"]](**kw), "own")
    else:
        scorer = Scorer(build_model(algo, impl, params, job), impl.get("score", "neg_samples"))
    stage("train")
    t0 = time.time()
    Xtr = pre.fit_transform(X.iloc[tr])
    scorer.fit(Xtr)
    log(f"fit finished in {time.time() - t0:.2f}s")
    stage("evaluate")
    s_tr, s_te = scorer.score(Xtr), scorer.score(pre.transform(X.iloc[te]))
    y_te = y[te] if y is not None else None
    m, primary, charts, thr = anomaly_report(s_te, s_tr, y_te, contamination, df.iloc[te], job)
    stage("save")
    decision = None
    est = scorer.est
    if hasattr(est, "decision_function") and impl["adapter"] == "sk":
        decision = float(np.quantile(est.decision_function(Xtr), contamination))
    pipe = Pipeline([("pre", pre), ("model", scorer)])
    _save(job["run_dir"], dict(kind="anomaly", task=job["task"], features=features, threshold=thr,
                               decision_threshold=decision, adapter=impl["adapter"]), pipe)
    return dict(primary=primary, val=None, test=m, charts=charts, sample=X.iloc[te].head(32), params=params)
