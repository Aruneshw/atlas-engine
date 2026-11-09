"""Checks that decide whether a score can be trusted. All of them read training rows or finished predictions only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import log

CORR_LEAK = 0.995             # one column that is almost the target
PAIR_LEAK = 0.9999            # two columns that rebuild the target exactly, as raw_bias + forecast does
TOO_GOOD = 0.995
BOOTSTRAPS = 400


def find_leaks(X: pd.DataFrame, y: np.ndarray, features: list, task: str) -> list[str]:
    """Plain-English findings about columns that give the answer away. `X` and `y` are training rows."""
    found = []
    num = [f["name"] for f in features if f["type"] == "num"]
    if task == "regression" or len(np.unique(y)) == 2:
        data = X[num].apply(lambda c: c.fillna(c.median())).to_numpy(dtype=float) if num else np.empty((len(y), 0))
        keep = [i for i in range(data.shape[1]) if data[:, i].std() > 0]
        if keep and np.std(y) > 0:
            names = [num[i] for i in keep]
            r = np.corrcoef(np.column_stack([data[:, keep], y]), rowvar=False)
            to_y, among = r[-1, :-1], r[:-1, :-1]
            single = {n for n, v in zip(names, to_y) if abs(v) >= CORR_LEAK}
            found += [f"'{n}' is almost identical to the target (correlation {abs(to_y[names.index(n)]):.3f})." for n in sorted(single)]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if names[i] in single or names[j] in single or abs(among[i, j]) > 0.9999:
                        continue
                    r2 = (to_y[i] ** 2 + to_y[j] ** 2 - 2 * to_y[i] * to_y[j] * among[i, j]) / (1 - among[i, j] ** 2)
                    if r2 >= PAIR_LEAK:
                        found.append(f"'{names[i]}' and '{names[j]}' together rebuild the target exactly, so at least one is probably calculated from it.")
    if task == "classification":
        for f in features:
            if f["type"] == "cat" and 2 <= X[f["name"]].nunique() < len(X) / 2 and pd.Series(y, index=X.index).groupby(X[f["name"]]).nunique().max() == 1:
                found.append(f"'{f['name']}' decides the target exactly.")
    return found


def split_problems(X: pd.DataFrame, features: list, tr, te) -> list[str]:
    """Rows repeated across the split, and categories the model will meet for the first time in the test set."""
    out = []
    hashed = pd.util.hash_pandas_object(X, index=False).to_numpy()
    repeated = float(np.isin(hashed[te], hashed[tr]).mean())
    if repeated >= 0.05:
        out.append(f"{repeated:.0%} of test rows also appear in training with identical inputs. If they are the same records, the score is too high.")
    for f in features:
        if f["type"] == "cat":
            seen = set(X[f["name"]].iloc[tr])
            new = X[f["name"]].iloc[te][~X[f["name"]].iloc[te].isin(seen)]
            if len(new) >= max(1, 0.01 * len(te)):
                top = new.value_counts().index[0]
                out.append(f"{len(new) / len(te):.0%} of test rows have a '{f['name']}' value training never saw, such as '{top}'. Train on data that covers it.")
    return out


def _interval(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def regression_extras(pred, y_tr, y_te, seed=42) -> dict:
    """Score to beat, and how far the result could move on a different sample of test rows."""
    from sklearn.metrics import r2_score

    base = float(np.mean(np.abs(y_te - np.mean(y_tr))))
    r = np.random.default_rng(seed)
    n = len(y_te)
    r2s, maes = [], []
    for _ in range(BOOTSTRAPS):
        i = r.integers(0, n, n)
        if np.ptp(y_te[i]) > 0:
            r2s.append(r2_score(y_te[i], pred[i]))
        maes.append(float(np.mean(np.abs(y_te[i] - pred[i]))))
    r2_low, r2_high = _interval(r2s) if r2s else (float("nan"), float("nan"))
    mae_low, mae_high = _interval(maes)
    return dict(baseline_mae=base, r2_low=r2_low, r2_high=r2_high, mae_low=mae_low, mae_high=mae_high)


def classification_extras(pred, y_tr, y_te, seed=42) -> dict:
    base = float(np.mean(y_te == np.bincount(y_tr).argmax()))
    right = (np.asarray(pred) == y_te).astype(float)
    r = np.random.default_rng(seed)
    low, high = _interval([right[r.integers(0, len(right), len(right))].mean() for _ in range(BOOTSTRAPS)])
    return dict(baseline_accuracy=base, accuracy_low=low, accuracy_high=high)


def report_extras(task, test, primary_value, warn):
    """Say in the terminal what the numbers mean, and flag a score that is too good."""
    if task == "regression":
        log(f"always guessing the training average would miss by {test['baseline_mae']:.4g}; this model misses by {test['mae']:.4g}")
        log(f"95% range on other test samples: r2 {test['r2_low']:.4f} to {test['r2_high']:.4f}, mae {test['mae_low']:.4g} to {test['mae_high']:.4g}")
        if test["mae"] >= test["baseline_mae"]:
            warn("This model is no better than always guessing the average. Check the inputs before using it.")
    else:
        log(f"always guessing the commonest class scores accuracy {test['baseline_accuracy']:.4f}; 95% range for this model {test['accuracy_low']:.4f} to {test['accuracy_high']:.4f}")
        if test["accuracy"] <= test["baseline_accuracy"]:
            warn("This model is no better than always guessing the commonest class.")
    if primary_value >= TOO_GOOD:
        warn(f"A score of {primary_value:.4f} is unusually high. Check every input column: could you really know it when making the prediction?")
