"""Statistical and lag-feature forecasting.

Every forecaster, deep ones included, is scored the same way: fit on the earlier data, then
forecast the whole held-out segment from one origin with no peeking. That keeps run
comparisons fair and matches what predicting the real future does.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DataError, split_indices
from .metrics import forecast_metrics
from .protocol import log, stage
from .tabular import build_model


def series_from(df, target):
    y = pd.to_numeric(df[target], errors="coerce")
    bad = int(y.isna().sum())
    if bad > 0.2 * len(y):
        raise DataError(f"Target '{target}' has {bad} non-numeric or missing values out of {len(y)}.")
    if bad:
        log(f"filling {bad} gaps in the series by interpolation", "warn")
        y = y.interpolate(limit_direction="both")
    return y.to_numpy(dtype=float)


class StatsForecaster:
    """Fits the named statsmodels-backed method and forecasts `steps` from the end."""

    def __init__(self, method, params):
        self.method, self.params = method, params

    def fit(self, y):
        from statsmodels.tsa.ar_model import AutoReg
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        m, p = self.method, self.params
        self.tail = np.asarray(y)[-max(int(p.get("window", 1)), 1):]
        period = p.get("seasonal_periods")
        if period and len(y) < 2 * period:
            raise DataError(f"Season length {period} needs at least {2 * period} training points; there are {len(y)}.")
        if m == "sma":
            self.res = None
        elif m == "ses":
            self.res = SimpleExpSmoothing(y, initialization_method="estimated").fit()
        elif m == "holt":
            self.res = Holt(y, initialization_method="estimated").fit()
        elif m == "holt_winters":
            self.res = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=period,
                                            initialization_method="estimated").fit()
        elif m == "ets":
            self.res = ETSModel(y, error="add", trend="add", seasonal="add", seasonal_periods=period).fit(disp=False)
        elif m == "ar":
            self.res = AutoReg(y, lags=min(p["lags"], len(y) // 2 - 1)).fit()
        elif m == "arima":
            self.res = ARIMA(y, order=(p["p"], p["d"], p["q"])).fit()
        elif m == "sarima":
            self.res = SARIMAX(y, order=(p["p"], p["d"], p["q"]), seasonal_order=(1, 1, 1, period)).fit(disp=False)
        self.n = len(y)
        return self

    def forecast(self, steps):
        if self.method == "sma":
            return np.full(steps, float(np.mean(self.tail)))
        if self.method == "ar":
            return np.asarray(self.res.predict(start=self.n, end=self.n + steps - 1))
        return np.asarray(self.res.forecast(steps))


class LagForecaster:
    """Any regressor over lag and rolling-mean features, forecast recursively."""

    def __init__(self, model, lags):
        self.model, self.lags = model, lags

    def _row(self, hist):
        h = hist[-self.lags:]
        return np.r_[h[::-1], h.mean(), h.std(), h[-min(3, len(h)):].mean()]

    def fit(self, y):
        Xs = np.array([self._row(y[:i]) for i in range(self.lags, len(y))])
        self.model.fit(Xs, y[self.lags:])
        self.tail = np.asarray(y)[-self.lags:]
        return self

    def forecast(self, steps, tail=None):
        hist = list(self.tail if tail is None else tail)
        out = []
        for _ in range(steps):
            v = float(np.ravel(self.model.predict(self._row(np.asarray(hist)).reshape(1, -1)))[0])
            out.append(v)
            hist.append(v)
        return np.asarray(out)


def forecast_charts(y_tr, y_te, pred, future, label="forecast"):
    n_hist = min(len(y_tr), 150)
    hist = y_tr[-n_hist:]
    x_t = list(range(n_hist, n_hist + len(y_te)))
    thin = np.linspace(0, len(y_te) - 1, min(len(y_te), 300)).astype(int)
    series = [dict(name="history", x=list(range(n_hist)), y=hist),
              dict(name="actual", x=[x_t[i] for i in thin], y=y_te[thin]),
              dict(name=label, x=[x_t[i] for i in thin], y=np.ravel(pred)[thin])]
    return [dict(type="line", title="Forecast vs actual (test segment)", series=series, xlabel="step"),
            dict(type="line", title="Future forecast", series=[dict(name="future", y=future)], xlabel="steps ahead")]


def run_forecast(df, features, job, algo, impl, params, ctx):
    target = job["dataset"]["target"]
    stage("prepare")
    y = series_from(df, target)
    tr, va, te, _ = split_indices(len(y), None, job)
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    log(f"chronological split: train {len(tr)}, validation {len(va)}, test {len(te)}")
    horizon = int(job.get("horizon", 12))
    stage("build")
    if impl["adapter"] == "lag":
        lags = int(job.get("seq_len") or 12)
        if len(y_tr) < 2 * lags + 5:
            raise DataError(f"{lags} lags need more than {2 * lags + 5} training points; there are {len(y_tr)}.")
        fc = lambda: LagForecaster(build_model(algo, impl, params, job), lags)
        log(f"regressor over {lags} lag features, forecast recursively")
    else:
        fc = lambda: StatsForecaster(impl["cls"], params)
        log(f"statistical method: {impl['cls']}")
    stage("train")
    t0 = time.time()
    model = fc().fit(y_tr)
    log(f"fit finished in {time.time() - t0:.2f}s")
    stage("evaluate")
    val = forecast_metrics(model.forecast(len(y_va)), y_va, y_tr)
    refit = fc().fit(np.r_[y_tr, y_va])
    pred = refit.forecast(len(y_te))
    test = forecast_metrics(pred, y_te, np.r_[y_tr, y_va])
    log("refitting on all data for the future forecast")
    final = fc().fit(y)
    future = final.forecast(horizon)
    stage("save")
    d = Path(job["run_dir"]) / "model"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, d / "forecaster.joblib")
    (d / "meta.json").write_text(json.dumps(dict(kind="forecaster", task=job["task"], target=target, horizon=horizon), indent=1))
    pd.DataFrame({"actual": y_te, "forecast": pred}).to_csv(Path(job["run_dir"]) / "predictions.csv", index=False)
    pd.DataFrame({"step": range(1, horizon + 1), "forecast": future}).to_csv(Path(job["run_dir"]) / "future.csv", index=False)
    return dict(primary=("mae", test["mae"], False), val=val, test=test,
                charts=forecast_charts(np.r_[y_tr, y_va], y_te, pred, future), sample=None, params=params)
