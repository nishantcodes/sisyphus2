"""Known-good baselines, run without asking a model to write them.

Every campaign so far spent its first three to five rival turns — the most
expensive calls in the system — writing LightGBM / XGBoost / CatBoost baselines
that differ from the last campaign's only in column names. That is
transcription, not judgement. These scripts read the target and metric from
the environment, so one file serves any binary or regression table, and the
first LLM turn starts from a scored, blendable pool instead of from nothing.

They obey the same contract as generated code (shared folds, SIS_FRAC, OOF and
PRED saved, CV_SCORE/CV_STD printed) and go through the same runner, so they
are ordinary experiments in the ledger — parents a rival can extend.
"""

from __future__ import annotations

_HEAD = r'''
import os, numpy as np, pandas as pd
DATA, OUT = os.environ["SIS_DATA"], os.environ["SIS_OUT"]
FRAC = float(os.environ.get("SIS_FRAC", "1"))
THREADS = int(os.environ.get("SIS_THREADS", 4))
TARGET, METRIC = os.environ["SIS_TARGET"], os.environ.get("SIS_METRIC", "auc")
SEED = 42
tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
folds = np.load(os.path.join(DATA, "folds.npy"))
if os.environ.get("SIS_IDX"):                          # harness-chosen smoke rows
    idx = np.load(os.environ["SIS_IDX"])
    tr, folds = tr.iloc[idx].reset_index(drop=True), folds[idx]
elif FRAC < 1:
    idx = np.random.default_rng(SEED).choice(len(tr), max(200, int(len(tr) * FRAC)), replace=False)
    tr, folds = tr.iloc[idx].reset_index(drop=True), folds[idx]
t = tr[TARGET]
LABELS = []
if pd.api.types.is_numeric_dtype(t) and t.nunique() > 2:
    TASK, y = "reg", t.values.astype(float)
elif t.nunique() > 2:
    LABELS = sorted(t.astype(str).unique())          # canonical column order
    TASK, y = "multi", pd.Categorical(t.astype(str), categories=LABELS).codes.astype(int)
else:
    pos = t.value_counts().index[-1]                 # minority class is positive
    TASK, y = "bin", (t == pos).values.astype(float)
K = max(len(LABELS), 1)
feats = [c for c in tr.columns if c not in ("id", TARGET)]
cats = [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]
X, Xt = tr[feats].copy(), te[feats].copy()
for c in cats:
    levels = sorted(set(X[c].astype(str)) | set(Xt[c].astype(str)))
    X[c] = pd.Categorical(X[c].astype(str), categories=levels)
    Xt[c] = pd.Categorical(Xt[c].astype(str), categories=levels)

def score(yy, pp):
    from sklearn import metrics as M
    import re
    m = METRIC.lower()
    if TASK == "multi":
        mk = re.search(r"map\s*[@_]\s*(\d+)", m)
        if mk:
            kk = int(mk.group(1)); top = np.argsort(-pp, 1)[:, :kk]
            return float(((top == yy[:, None]) / np.arange(1, kk + 1)).sum(1).mean())
        if "acc" in m: return float((pp.argmax(1) == yy).mean())
        return -M.log_loss(yy, pp, labels=list(range(K)))
    if "auc" in m: return M.roc_auc_score(yy, pp)
    if "log" in m: return -M.log_loss(yy, np.clip(pp, 1e-7, 1 - 1e-7))
    if "rmsle" in m: return -np.sqrt(np.mean((np.log1p(np.clip(yy, 0, None)) - np.log1p(np.clip(pp, 0, None))) ** 2))
    if "mae" in m: return -M.mean_absolute_error(yy, pp)
    if "rmse" in m or TASK == "reg": return -np.sqrt(np.mean((yy - pp) ** 2))
    return M.roc_auc_score(yy, pp)
# Scale-aware boosting budget. Timed at s6e9 size (668k rows x 10 folds, 4
# cores) with the small-table settings: LGBM 27 min, XGB 36 min, CatBoost past
# a 60-min timeout (upper bounds — early stopping never fired on that data).
# A large table needs fewer, bigger steps; the gain from lr 0.03 over 0.08 is
# below the paired-test floor at that size anyway.
BIG = len(tr) > 200_000
LR = 0.08 if BIG else 0.03
ROUNDS = 3000 if BIG else 5000
PATIENCE = 100 if BIG else 200
CB_LR, CB_ITERS = (0.12, 2000) if BIG else (0.08, 3000)
if TASK == "multi":
    SIGN = 1 if ("map" in METRIC.lower() or "acc" in METRIC.lower()) else -1
else:
    SIGN = -1 if any(k in METRIC.lower() for k in ("log", "rmse", "mae", "mse")) or (TASK == "reg" and "auc" not in METRIC.lower()) else 1

def proba(m, Z):
    if TASK == "bin": return m.predict_proba(Z)[:, 1]
    if TASK == "multi": return m.predict_proba(Z)     # columns = classes present
    return m.predict(Z)
'''

_TAIL = r'''
shape = (lambda n: (n, K)) if TASK == "multi" else (lambda n: n)
oof, pred, scores = np.zeros(shape(len(tr))), np.zeros(shape(len(te))), []
ks = np.unique(folds)
for k in ks:
    va = folds == k
    if TASK == "multi":
        # Boosters want contiguous labels 0..k'-1, and a small subsample (the
        # 5% smoke run) routinely lacks a rare class. Fit on the classes that
        # are present, then spread the columns back to all K.
        present = np.unique(y[~va])
        lut = -np.ones(K, int); lut[present] = np.arange(len(present))
        ya, ye = lut[y[~va]], lut[y[va]]
        keep = ye >= 0
        pv, pt = fit_predict(X[~va], ya, X[va][keep], ye[keep], [X[va], Xt])
        p_va, p_te = np.zeros((va.sum(), K)), np.zeros((len(Xt), K))
        p_va[:, present], p_te[:, present] = pv, pt
    else:
        p_va, p_te = fit_predict(X[~va], y[~va], X[va], y[va], [X[va], Xt])
    oof[va] = p_va
    pred += p_te / len(ks)
    scores.append(SIGN * score(y[va], p_va))
np.save(os.path.join(OUT, "oof.npy"), oof)
np.save(os.path.join(OUT, "test.npy"), pred)
sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
if TASK != "multi":
    sub[sub.columns[1]] = pred
elif len(sub.columns) > 2:
    cols = list(sub.columns[1:]); sub[cols] = pred[:, [LABELS.index(str(c)) for c in cols]]
else:
    import re
    mk = re.search(r"map\s*[@_]\s*(\d+)", METRIC.lower())
    order = np.argsort(-pred, 1)
    kk = int(mk.group(1)) if mk else 1
    sub[sub.columns[1]] = [" ".join(LABELS[j] for j in r[:kk]) for r in order]
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
print(f"CV_SCORE: {np.mean(scores):.6f}")
print(f"CV_STD: {np.std(scores):.6f}")
print("OOF: oof.npy"); print("PRED: test.npy"); print("SUBMISSION: submission.csv")
'''

_MODELS = {
    "lgbm": r'''
import lightgbm as lgb
def fit_predict(Xa, ya, Xv, yv, outs):
    params = dict(n_estimators=ROUNDS, learning_rate=LR, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=1.0, max_bin=255,
                  random_state=SEED, n_jobs=THREADS, verbose=-1)
    m = (lgb.LGBMRegressor if TASK == "reg" else lgb.LGBMClassifier)(**params)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return [proba(m, Z) for Z in outs]
''',
    "xgb": r'''
import xgboost as xgb
def fit_predict(Xa, ya, Xv, yv, outs):
    params = dict(n_estimators=ROUNDS, learning_rate=LR, max_depth=7,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                  max_cat_to_onehot=1, early_stopping_rounds=PATIENCE,
                  random_state=SEED, n_jobs=THREADS)
    m = (xgb.XGBRegressor if TASK == "reg" else xgb.XGBClassifier)(**params)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    return [proba(m, Z) for Z in outs]
''',
    "catboost": r'''
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
def fit_predict(Xa, ya, Xv, yv, outs):
    def prep(Z):
        Z = Z.copy()
        for c in cats: Z[c] = Z[c].astype(str)
        return Z
    params = dict(iterations=CB_ITERS, learning_rate=CB_LR, depth=6,
                  l2_leaf_reg=3, random_seed=SEED, thread_count=THREADS,
                  od_type="Iter", od_wait=PATIENCE, verbose=False)
    if TASK == "multi" and len(np.unique(ya)) > 2: params["loss_function"] = "MultiClass"
    m = (CatBoostRegressor if TASK == "reg" else CatBoostClassifier)(**params)
    m.fit(Pool(prep(Xa), ya, cat_features=cats), eval_set=Pool(prep(Xv), yv, cat_features=cats),
          use_best_model=True)
    return [proba(m, prep(Z)) for Z in outs]
''',
}

DESCRIPTIONS = {
    "lgbm": "LGBM baseline (recipe): raw features, native categoricals",
    "xgb": "XGB baseline (recipe): raw features, native categoricals",
    "catboost": "CatBoost baseline (recipe): raw features, native categoricals",
}


def script(model: str) -> str:
    return _HEAD + _MODELS[model] + _TAIL


def names() -> list[str]:
    return list(_MODELS)


def demo() -> None:
    """Run every recipe on a tiny synthetic competition through the real runner."""
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from . import metrics, run
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = np.random.default_rng(0)
        n = 1200
        x = rng.normal(size=n)
        c = rng.choice(["a", "b", "c"], n)
        tgt = np.where(x + (c == "a") + rng.normal(size=n) > 1, "yes", "no")
        df = pd.DataFrame({"id": np.arange(n), "x": x, "c": c, "t": tgt})
        df.iloc[:900].to_csv(d / "train.csv", index=False)
        df.iloc[900:].drop(columns="t").to_csv(d / "test.csv", index=False)
        pd.DataFrame({"id": df.id[900:], "t": 0.5}).to_csv(d / "sample_submission.csv", index=False)
        np.save(d / "folds.npy", np.arange(900) % 5)
        os.environ["SIS_TARGET"], os.environ["SIS_METRIC"] = "t", "auc"
        y = metrics.encode_target(df.t[:900])
        for m in names():
            r = run.run_local(script(m), d / m, data_dir=d, n_train=900, n_test=300,
                              y=y, metric="auc", timeout=600)
            assert not r.get("failure"), (m, r)
            assert r["cv"] > 0.7, (m, r["cv"])
        # Multiclass, MAP@3, with a rare class the smoke subsample will miss.
        lab = np.where(x > 1.2, "hi", np.where(x < -1, "lo", np.where(c == "a", "a", "mid")))
        lab[:3] = "rare"
        df["t"] = lab
        df.iloc[:900].to_csv(d / "train.csv", index=False)
        pd.DataFrame({"id": df.id[900:], "t": "a hi lo"}).to_csv(d / "sample_submission.csv", index=False)
        os.environ["SIS_METRIC"] = "map@3"
        ym = metrics.encode_target(df.t[:900])
        for m in names():
            for frac in (0.25, 1.0):
                r = run.run_local(script(m), d / f"{m}_mc{frac}", data_dir=d, frac=frac,
                                  n_train=900 if frac == 1.0 else 0, n_test=300,
                                  y=ym if frac == 1.0 else None, metric="map@3", timeout=600)
                assert not r.get("failure"), (m, frac, r.get("error", "")[-800:])
            assert np.load(r["oof_path"]).shape == (900, 5), m
            assert r["cv"] > 0.6, (m, r["cv"])
    print(f"recipes.py: {3 * len(names())} checks passed")


if __name__ == "__main__":
    demo()
