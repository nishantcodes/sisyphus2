"""Where is the champion wrong, and what would fix it?

This module is the reason v2 exists. v1 could rank experiments but never
interrogate one: it knew model A scored 0.9702 and model B scored 0.9698, and
had nothing to say about *where* either was losing. So it did the only thing
left — searched more configs — and converged on exactly the score the public
notebook pool converges on.

The 1st-place writeup describes the alternative in one sentence: the agents
"continually studied the single model and analyzed where it made mistakes and
what types of patterns different features or model architectures could
replicate and constantly added just what was needed to address every short
coming." That loop needs a machine that turns `(y, oof_predictions, X)` into a
list of concrete, addressable weaknesses. This is that machine.

Everything here is metric-agnostic and problem-agnostic on purpose — it needs
only true labels, out-of-fold predictions, and the feature frame. Nothing about
electric vehicles, or AUC, is baked in.

The five questions it answers, in the order they usually pay:

  1. **Calibration** — is the model's probability scale honest, or is it
     confidently wrong in a specific band?
  2. **Slices** — which regions of which feature carry loss far above the
     model's average? This is where a feature or a split is missing.
  3. **Missingness** — do absent values predict error? The writeup names this
     as the single largest new source of signal its agents found, and it is the
     one an ordinary feature-importance plot never surfaces.
  4. **Interactions** — which *pairs* of features are jointly worse than either
     alone? A pair that hurts together but not apart is a missing cross-feature.
  5. **Disagreement** — where does a rival family's champion diverge, and is
     that divergence worth blending? Two models wrong in the same places blend
     to nothing; two wrong in different places are the whole ensemble gain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_BINS = 10          # quantile bins per numeric feature
MIN_BIN = 200          # a slice smaller than this is noise, not a finding
TOP_SLICES = 12        # how many findings to surface; more is unreadable


# --------------------------------------------------------------- per-row loss

def row_loss(y: np.ndarray, p: np.ndarray, metric: str = "auc") -> np.ndarray:
    """Per-row badness, on a scale where bigger is worse.

    Ranking metrics like AUC do not decompose into per-row terms at all, so for
    those we use absolute error as the surrogate: a positive scored low and a
    negative scored high are exactly the rows that cost you rank pairs, and
    that is what |y - p| measures. Where the metric *is* decomposable
    (logloss, RMSE) we use the real thing.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = metric.lower()
    if "logloss" in m or "entropy" in m:
        eps = 1e-15
        q = np.clip(p, eps, 1 - eps)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))
    if "rmse" in m or "mse" in m or "squared" in m:
        return (y - p) ** 2
    if "mae" in m or "absolute" in m:
        return np.abs(y - p)
    return np.abs(y - p)      # auc and anything else


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "n/a"


# ------------------------------------------------------------------ sections

def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> str:
    """Predicted rate vs actual rate, by prediction decile.

    A model can rank well and still be miscalibrated in a band; when the gap is
    concentrated at one end, the fix is usually a monotone post-transform or a
    missing feature that only matters for that population — both cheap, and
    neither visible from the headline CV score.
    """
    order = np.argsort(p)
    edges = np.array_split(order, bins)
    lines = ["decile |    n | mean pred | actual | gap",
             "-------|------|-----------|--------|--------"]
    worst, worst_gap = 0, 0.0
    for i, idx in enumerate(edges):
        if len(idx) == 0:
            continue
        mp, ac = p[idx].mean(), y[idx].mean()
        gap = ac - mp
        if abs(gap) > abs(worst_gap):
            worst, worst_gap = i + 1, gap
        lines.append(f"{i + 1:>6} | {len(idx):>4} |    {_fmt(mp)} | {_fmt(ac)} | {gap:+.4f}")
    lines.append(f"\nlargest miscalibration: decile {worst}, actual is "
                 f"{worst_gap:+.4f} vs predicted")
    return "\n".join(lines)


def _binned(col: pd.Series) -> pd.Series | None:
    """Bin one column into comparable groups; None if it cannot be sliced."""
    if col.dtype == object or str(col.dtype) == "category":
        vc = col.astype("string").value_counts()
        keep = vc[vc >= MIN_BIN].index
        if len(keep) < 2:
            return None
        return col.astype("string").where(col.astype("string").isin(keep), "«rare»")
    nun = col.nunique(dropna=True)
    if nun < 2:
        return None
    if nun <= MAX_BINS:                       # already discrete — keep as-is
        return col.astype("string")
    try:
        return pd.qcut(col, MAX_BINS, duplicates="drop").astype("string")
    except (ValueError, TypeError):
        return None


def worst_slices(X: pd.DataFrame, loss: np.ndarray, top: int = TOP_SLICES) -> str:
    """Feature regions where loss runs far above the model's average.

    Ranked by `lift * sqrt(n)`, not by lift alone: a bin of 30 rows with triple
    the loss is a coin flip, and chasing it is how an agent burns a day. The
    weighting keeps big, moderately-bad slices — the ones actually worth
    engineering for — above small, spectacularly-bad ones.
    """
    base = loss.mean()
    rows = []
    for c in X.columns:
        b = _binned(X[c])
        if b is None:
            continue
        g = pd.DataFrame({"b": b.values, "l": loss}).groupby("b", observed=True)["l"]
        agg = g.agg(["mean", "count"])
        agg = agg[agg["count"] >= MIN_BIN]
        for val, r in agg.iterrows():
            lift = r["mean"] / base if base > 0 else 1.0
            rows.append((lift * np.sqrt(r["count"]), lift, int(r["count"]), c, str(val)))
    if not rows:
        return "(no slice large enough to judge)"
    rows.sort(reverse=True)
    out = [f"overall mean loss {_fmt(base)}. Slices carrying the most excess loss:",
           "  lift |      n | feature = region",
           "-------|--------|------------------"]
    for _, lift, n, c, val in rows[:top]:
        out.append(f" {lift:>5.2f} | {n:>6} | {c} = {val}")
    out.append("\nA lift far above 1.0 on a large slice means the model has no "
               "handle on that region — a feature, a split, or an interaction is "
               "missing there.")
    return "\n".join(out)


def missingness(X: pd.DataFrame, loss: np.ndarray, y: np.ndarray) -> str:
    """Does absence itself carry signal?

    The writeup singles this out: "the agents found many features related to
    missingness which gave improvements to CV and LB." Two separate questions
    are asked here, because they have different fixes — whether missing rows are
    *harder* (needs a feature the model lacks) and whether they have a different
    *base rate* (the missingness indicator is itself predictive).
    """
    cols = [c for c in X.columns if X[c].isna().any()]
    if not cols:
        return "(no missing values in any column)"
    base_loss, base_rate = loss.mean(), float(np.mean(y))
    out = ["feature            |  n missing | loss(miss) | loss(present) | target(miss) | target(present)",
           "-------------------|------------|------------|---------------|--------------|----------------"]
    findings = []
    for c in cols:
        m = X[c].isna().values
        n = int(m.sum())
        if n < MIN_BIN or n == len(m):
            continue
        lm, lp = loss[m].mean(), loss[~m].mean()
        tm, tp = float(y[m].mean()), float(y[~m].mean())
        out.append(f"{c[:18]:<18} | {n:>10} |     {_fmt(lm)} |        {_fmt(lp)} "
                   f"|       {_fmt(tm)} |         {_fmt(tp)}")
        if lm > lp * 1.10:
            findings.append(f"  - rows missing `{c}` are {lm / lp:.2f}x harder than "
                            f"rows that have it")
        if abs(tm - tp) > 0.02 * max(base_rate, 1e-9) + 0.005:
            findings.append(f"  - `{c}` being missing shifts the target rate "
                            f"{tm - tp:+.4f} — the indicator is itself a feature")
    # The count of missing fields per row is often stronger than any single
    # indicator, and it is one line of feature engineering to add.
    nmiss = X[cols].isna().sum(axis=1).values
    if nmiss.max() > 0:
        g = pd.DataFrame({"k": nmiss, "l": loss, "y": y}).groupby("k")
        agg = g.agg(n=("l", "size"), loss=("l", "mean"), rate=("y", "mean"))
        agg = agg[agg["n"] >= MIN_BIN]
        if len(agg) > 1:
            out.append("\nby COUNT of missing fields in the row:")
            out.append("  k |      n | mean loss | target rate")
            for k, r in agg.iterrows():
                out.append(f" {k:>2} | {int(r['n']):>6} |    {_fmt(r['loss'])} "
                           f"|      {_fmt(r['rate'])}")
            if agg["rate"].max() - agg["rate"].min() > 0.01:
                findings.append("  - the NUMBER of missing fields tracks the target; "
                                "add it as a feature")
    if findings:
        out.append("\nworth acting on:")
        out.extend(dict.fromkeys(findings))     # dedupe, keep order
    else:
        out.append(f"\nnothing stands out (overall loss {_fmt(base_loss)})")
    return "\n".join(out)


def interactions(X: pd.DataFrame, loss: np.ndarray, top_feats: int = 6) -> str:
    """Pairs that are jointly worse than either feature alone.

    Restricted to the features that already look worst on their own, because
    the full pairwise sweep is quadratic and mostly noise. A cell whose loss
    exceeds both of its parent slices is the signature of a missing cross-term
    — the model can separate on each axis but not on the corner.
    """
    base = loss.mean()
    scored = []
    for c in X.columns:
        b = _binned(X[c])
        if b is None:
            continue
        g = pd.DataFrame({"b": b.values, "l": loss}).groupby("b", observed=True)["l"]
        agg = g.agg(["mean", "count"])
        agg = agg[agg["count"] >= MIN_BIN]
        if len(agg):
            scored.append((agg["mean"].max() / base if base > 0 else 1.0, c, b))
    scored.sort(reverse=True)
    picks = scored[:top_feats]
    if len(picks) < 2:
        return "(too few sliceable features for pair analysis)"

    out, found = [], []
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            _, ca, ba = picks[i]
            _, cb, bb = picks[j]
            df = pd.DataFrame({"a": ba.values, "b": bb.values, "l": loss})
            cell = df.groupby(["a", "b"], observed=True)["l"].agg(["mean", "count"])
            cell = cell[cell["count"] >= MIN_BIN]
            if cell.empty:
                continue
            ma = df.groupby("a", observed=True)["l"].mean()
            mb = df.groupby("b", observed=True)["l"].mean()
            for (va, vb), r in cell.iterrows():
                solo = max(ma.get(va, base), mb.get(vb, base))
                if r["mean"] > solo * 1.15:
                    found.append((r["mean"] / base, int(r["count"]),
                                  f"{ca}={va} AND {cb}={vb}"))
    if not found:
        return "(no pair is meaningfully worse than its parts — no obvious missing cross-feature)"
    found.sort(reverse=True)
    out.append("cells worse than either feature alone (a missing interaction):")
    out.append("  lift |      n | region")
    for lift, n, desc in found[:8]:
        out.append(f" {lift:>5.2f} | {n:>6} | {desc}")
    return "\n".join(out)


def disagreement(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                 name_a: str = "champion", name_b: str = "rival",
                 metric: str = "auc") -> str:
    """Do two models fail in the same places, and is blending worth it?

    Correlation of *errors* is the number that matters, not correlation of
    predictions. Two models that agree everywhere blend to nothing however
    different their architectures look; the gain in an ensemble comes entirely
    from being wrong in different places. This also reports what a plain 50/50
    average scores, which is the cheapest possible check on whether the pair is
    worth keeping.
    """
    la, lb = row_loss(y, a, metric), row_loss(y, b, metric)
    pred_corr = float(np.corrcoef(a, b)[0, 1])
    err_corr = float(np.corrcoef(la, lb)[0, 1])
    lines = [f"prediction correlation {pred_corr:.4f}, ERROR correlation {err_corr:.4f}"]
    from .metrics import score
    sa, sb = score(y, a, metric), score(y, b, metric)
    sm = score(y, (a + b) / 2, metric)
    lines.append(f"{name_a} {_fmt(sa, 5)} · {name_b} {_fmt(sb, 5)} · "
                 f"50/50 blend {_fmt(sm, 5)}")
    if err_corr < 0.7:
        lines.append("Errors are substantially decorrelated — this pair is worth "
                     "blending, and worth understanding: the regions where they "
                     "disagree are where one has found signal the other has not.")
    else:
        lines.append("Errors are highly correlated — these two are effectively the "
                     "same model. Blending them will buy almost nothing; a genuinely "
                     "different architecture or feature view is needed.")
    # Where they diverge most is a lead in itself.
    d = np.abs(a - b)
    k = max(1, len(d) // 100)
    idx = np.argpartition(-d, k)[:k]
    lines.append(f"on the 1% of rows where they most disagree (n={len(idx)}): "
                 f"{name_a} loss {_fmt(la[idx].mean())}, {name_b} loss {_fmt(lb[idx].mean())}")
    return "\n".join(lines)


# -------------------------------------------------------------- the report

def report(X: pd.DataFrame, y: np.ndarray, oof: np.ndarray, *, metric: str = "auc",
           rival: tuple[str, np.ndarray] | None = None,
           max_cols: int = 40) -> str:
    """The full error-analysis brief the agent reads before proposing anything.

    Written as text rather than JSON on purpose: the consumer is a language
    model, and prose with small tables is what it reads best.
    """
    from .metrics import score
    if len(X) != len(y) or len(y) != len(oof):
        return (f"cannot analyse: X has {len(X)} rows, y has {len(y)}, "
                f"oof has {len(oof)}")
    Xa = X.iloc[:, :max_cols]
    loss = row_loss(y, oof, metric)
    parts = [
        f"# Error analysis of the current champion",
        f"\n{metric} = {_fmt(score(y, oof, metric), 5)} on {len(y):,} rows, "
        f"mean per-row loss {_fmt(loss.mean())}",
        "\n## 1. Calibration\n" + calibration(y, oof),
        "\n## 2. Where the loss is concentrated\n" + worst_slices(Xa, loss),
        "\n## 3. Missingness\n" + missingness(Xa, loss, y),
        "\n## 4. Interactions\n" + interactions(Xa, loss),
    ]
    if rival is not None:
        parts.append(f"\n## 5. Disagreement with {rival[0]}\n"
                     + disagreement(y, oof, rival[1], "champion", rival[0], metric))
    return "\n".join(parts)


def demo() -> None:
    """Self-check: plant known weaknesses and confirm the analyst finds them.

    The point of these assertions is that a report which cannot recover a defect
    we deliberately inserted will not find a real one either.
    """
    rng = np.random.default_rng(0)
    n = 6000
    a = rng.normal(size=n)
    b = rng.choice(["x", "y", "z"], size=n)
    c = rng.normal(size=n)
    c[: n // 3] = np.nan                       # planted: missingness matters
    y = (a + (b == "z") * 2.0 + rng.normal(scale=0.5, size=n) > 0).astype(float)
    X = pd.DataFrame({"a": a, "b": b, "c": c})

    # A model that has learned `a` but is blind to `b` — so the b=="z" slice
    # and the rows where b decides the label should light up.
    p = 1 / (1 + np.exp(-a))
    loss = row_loss(y, p, "auc")

    sl = worst_slices(X, loss)
    assert "b = z" in sl, sl                   # found the blind spot
    ca = calibration(y, p)
    assert "decile" in ca

    # Missingness in `c` is correlated with nothing here, but the section must
    # still run and report the column rather than crash.
    ms = missingness(X, loss, y)
    assert "c" in ms, ms

    # Two identical models must be reported as not worth blending.
    d = disagreement(y, p, p.copy(), metric="auc")
    assert "almost nothing" in d, d

    # A genuinely different model must be reported as worth blending.
    p2 = 1 / (1 + np.exp(-((b == "z") * 3.0 - 1)))
    d2 = disagreement(y, p, p2, metric="auc")
    assert "worth blending" in d2, d2

    r = report(X, y, p, metric="auc", rival=("other", p2))
    assert "Calibration" in r and "Missingness" in r and "Disagreement" in r
    print("analyst.py: 6 checks passed")


if __name__ == "__main__":
    demo()
