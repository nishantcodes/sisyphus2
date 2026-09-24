"""One place that knows what the competition's metric means.

Everything else in the system — verdicts, champion selection, blend weights,
the analyst's per-row loss — asks two questions: *what does this prediction
score* and *is bigger better*. Keeping both answers here is what makes the rest
of the agent problem-agnostic: pointing it at an RMSE regression instead of an
AUC classification changes this file's lookup and nothing else.

`score` deliberately never raises on an unknown name. An agent that dies at 3am
because a competition called its metric "roc_auc_score" instead of "auc" has
failed for no reason; unknown names fall back to AUC-or-RMSE by inspecting the
labels, and `direction` defaults to higher-is-better, which is right for most
of the board.
"""

from __future__ import annotations

import numpy as np

# Substring -> (sklearn callable name, higher_is_better). Matched loosely so
# "roc_auc", "AUC", and "area under the curve" all land in the same place.
_TABLE: dict[str, tuple[str, bool]] = {
    "auc": ("roc_auc_score", True),
    "roc": ("roc_auc_score", True),
    "logloss": ("log_loss", False),
    "log_loss": ("log_loss", False),
    "cross_entropy": ("log_loss", False),
    "rmse": ("_rmse", False),
    "mse": ("mean_squared_error", False),
    "mae": ("mean_absolute_error", False),
    "rmsle": ("_rmsle", False),
    "accuracy": ("accuracy_score", True),
    "f1": ("f1_score", True),
    "r2": ("r2_score", True),
    "map": ("average_precision_score", True),
    "average_precision": ("average_precision_score", True),
}


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _rmsle(y, p):
    y = np.clip(np.asarray(y, float), 0, None)
    p = np.clip(np.asarray(p, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


def _resolve(metric: str) -> tuple[str, bool]:
    # Spaces and hyphens both normalise to underscores, so "log loss",
    # "log-loss", and "logloss" are the same metric.
    m = (metric or "").lower().strip().replace("-", "_").replace(" ", "_")
    for key, val in _TABLE.items():
        if key in m:
            return val
    return ("", True)          # unknown: decided at call time by the labels


def direction(metric: str) -> bool:
    """True when a higher score is better."""
    return _resolve(metric)[1]


def score(y, p, metric: str = "auc") -> float:
    """Score predictions. Never raises — a metric that throws mid-campaign
    costs more than one that degrades to a sensible default."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    name, _ = _resolve(metric)
    if not name:
        # Unknown metric: binary labels almost always mean AUC, otherwise RMSE.
        name = "roc_auc_score" if set(np.unique(y)) <= {0.0, 1.0} else "_rmse"
    try:
        if name == "_rmse":
            return _rmse(y, p)
        if name == "_rmsle":
            return _rmsle(y, p)
        import sklearn.metrics as sk
        fn = getattr(sk, name)
        if name in ("accuracy_score", "f1_score"):
            return float(fn(y, (p >= 0.5).astype(int)))
        return float(fn(y, p))
    except Exception:
        return float("nan")


def demo() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert score(y, perfect, "auc") == 1.0
    assert score(y, 1 - perfect, "auc") == 0.0
    assert direction("auc") is True
    assert direction("RMSE") is False
    assert direction("log loss") is False
    # An unknown name still returns a usable number rather than exploding.
    assert score(y, perfect, "some_new_metric") == 1.0
    assert abs(score(np.array([1.0, 2.0]), np.array([1.0, 3.0]), "rmse") - 0.7071) < 1e-3
    assert np.isnan(score(y, np.array([np.nan] * 4), "auc"))
    print("metrics.py: 7 checks passed")


if __name__ == "__main__":
    demo()


def encode_target(col):
    """Target column -> float array, with the MINORITY class as 1.

    This has now been written wrong in three separate places (`agent._brief`,
    `agent._y`, `blend.build`), and each time it failed the same way: s6e9
    ships "Yes"/"No", `.astype(float)` raises, and the caller's try/except
    turned a broken core feature into a one-line warning nobody read. The
    error analysis silently did not run for 14 experiments because of it.
    One implementation, used everywhere.
    """
    import pandas as pd
    if pd.api.types.is_numeric_dtype(col):
        return col.values.astype(float)
    pos = col.value_counts().index[-1]          # minority class is the positive one
    return (col == pos).values.astype(float)


def fold_scores(y, p, folds, metric: str = "auc"):
    """Per-fold scores of one OOF array on the shared folds."""
    y, p, folds = np.asarray(y, float), np.asarray(p, float), np.asarray(folds)
    return np.array([score(y[folds == k], p[folds == k], metric)
                     for k in np.unique(folds)])


def paired(y, new, old, folds, metric: str = "auc") -> tuple[float, float, float]:
    """(mean delta, standard error, t) of `new` over `old`, fold by fold.

    The verdict used to compare the mean gain against the raw fold spread
    (`max(cv_std)`). That spread is mostly *fold difficulty* — every model
    finds fold 3 hard — and it cancels exactly when two models are scored on
    the same folds, which here they always are. Measured on the bank proxy: a
    +0.0016 gain against a 0.0027 fold spread was called `inconclusive`, yet
    all five folds improved and the paired t was 6.3. On s6e9 the fold spread
    (0.0008) was wider than the whole top-10% band of the board (0.0005), so
    the old rule could essentially never say `better`.

    Positive delta always means `new` is better, whatever the metric direction.
    """
    a = fold_scores(y, new, folds, metric)
    b = fold_scores(y, old, folds, metric)
    d = (a - b) if direction(metric) else (b - a)
    k = len(d)
    se = float(np.std(d, ddof=1) / np.sqrt(k)) if k > 1 else float("inf")
    mean = float(np.mean(d))
    return mean, se, (mean / se if se > 0 else (float("inf") if mean > 0 else 0.0))


def t_crit(k: int, conf: float = 0.95) -> float:
    """One-sided Student-t critical value for k folds (k-1 dof)."""
    try:
        from scipy.stats import t
        return float(t.ppf(conf, max(1, k - 1)))
    except Exception:
        return 2.0


def _paired_demo() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.3).astype(float)
    folds = np.arange(n) % 5
    base = y + rng.normal(scale=1.0, size=n)
    # A small real improvement shared across folds is detected...
    better = base + (y - 0.5) * 0.15
    m, se, t = paired(y, better, base, folds, "auc")
    assert m > 0 and t > t_crit(5), (m, se, t)
    # ...an identical model is exactly zero...
    m0, _, _ = paired(y, base, base, folds, "auc")
    assert m0 == 0.0
    # ...and for a lower-is-better metric the sign still means "new is better".
    m2, _, _ = paired(y, y + 0.1, y + 0.3, folds, "rmse")
    assert m2 > 0, m2
    print("metrics.py paired: 3 checks passed")


if __name__ == "__main__":
    _paired_demo()
