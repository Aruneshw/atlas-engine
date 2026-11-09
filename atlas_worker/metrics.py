"""Metrics. Each returns a flat dict of plain floats; chart data is appended separately."""
import numpy as np


def _scores(model, X, n_classes):
    """Class-probability-like scores, or None if the model offers none."""
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X))
        except Exception:
            pass
    return None


def from_labels(pred, y, classes, proba=None, charts=None):
    from sklearn import metrics as M

    labels = list(range(len(classes)))
    out = dict(
        accuracy=float(M.accuracy_score(y, pred)),
        balanced_accuracy=float(M.balanced_accuracy_score(y, pred)),
        precision_macro=float(M.precision_score(y, pred, average="macro", labels=labels, zero_division=0)),
        recall_macro=float(M.recall_score(y, pred, average="macro", labels=labels, zero_division=0)),
        f1_macro=float(M.f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
    )
    if len(classes) == 2:
        counts = np.bincount(y.astype(int), minlength=2)
        rare = int(np.argmin(counts))
        out["rare_class"] = classes[rare]
        out["rare_class_recall"] = float(M.recall_score(y == rare, pred == rare, zero_division=0))
        out["rare_class_precision"] = float(M.precision_score(y == rare, pred == rare, zero_division=0))
        if proba is not None and len(np.unique(y)) == 2:
            s = proba[:, 1] if proba.ndim == 2 else proba
            s_rare = s if rare == 1 else -s
            out["roc_auc"] = float(M.roc_auc_score(y == rare, s_rare))
            out["pr_auc"] = float(M.average_precision_score(y == rare, s_rare))
            if charts is not None:
                p, r, _ = M.precision_recall_curve(y == rare, s_rare)
                i = np.linspace(0, len(p) - 1, min(len(p), 200)).astype(int)
                charts.append(dict(type="line", title=f"Precision vs recall for '{classes[rare]}'",
                                   x=r[i], series=[dict(name="precision", y=p[i])], xlabel="recall"))
    if charts is not None:
        charts.insert(0, dict(type="heatmap", title="Confusion matrix (test)", labels=classes[:30],
                              z=M.confusion_matrix(y, pred, labels=labels[:30]).tolist()))
    return out


def classification_metrics(model, X, y, classes, charts=None):
    return from_labels(np.ravel(model.predict(X)).astype(int), y, classes, _scores(model, X, len(classes)), charts)


def regression_metrics(pred, y):
    from sklearn import metrics as M

    pred = np.ravel(pred)
    mae = float(M.mean_absolute_error(y, pred))
    return dict(r2=float(M.r2_score(y, pred)), mae=mae, rmse=float(np.sqrt(M.mean_squared_error(y, pred))),
                median_ae=float(M.median_absolute_error(y, pred)))


def anomaly_metrics(scores, flags, y):
    from sklearn import metrics as M

    out = dict(precision=float(M.precision_score(y, flags, zero_division=0)),
               recall=float(M.recall_score(y, flags, zero_division=0)),
               f1=float(M.f1_score(y, flags, zero_division=0)))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(M.roc_auc_score(y, scores))
        out["pr_auc"] = float(M.average_precision_score(y, scores))
    return out


def forecast_metrics(pred, y, train):
    """Point errors plus skill against a last-value forecast from the same origin."""
    pred, y = np.ravel(pred), np.ravel(y)
    mae = float(np.mean(np.abs(pred - y)))
    naive_mae = float(np.mean(np.abs(train[-1] - y)))
    denom = np.abs(y) + np.abs(pred)
    return dict(mae=mae, rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
                smape=float(np.mean(2 * np.abs(pred - y) / np.where(denom == 0, 1, denom))),
                naive_mae=naive_mae, skill_vs_naive=float(1 - mae / naive_mae) if naive_mae else float("nan"))
