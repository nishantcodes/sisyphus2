"""A local stand-in for Kaggle: hidden test labels, a public/private board.

Exists for two situations that both came up for real:

* **No route to Kaggle.** A cloud container whose network policy blocks
  kaggle.com (or that has no credentials) cannot download a competition or
  submit to one. Without this module the whole loop is dead there, including
  the part that most needs exercising — the submit / read-the-board / trust-CV
  cycle.
* **Testing the agent, not the model.** Changing how rivals talk to each other,
  or which model sits in which seat, needs a benchmark where one campaign takes
  an hour, not a month, and where the *private* score is knowable afterwards.
  Kaggle never tells you the private score until the competition closes; here
  it is sitting in a file the agent cannot read.

The layout mirrors a real competition exactly, so nothing downstream knows the
difference:

    input/train.csv              features + target
    input/test.csv               features only
    input/sample_submission.csv  id + target column, constant predictions
    .lb/labels.npy               test labels       <- NOT under input/
    .lb/public.npy               public-board mask (a fixed ~20% of test rows)
    .lb/private.jsonl            private scores, for the operator only

The labels live OUTSIDE `input/`, which is the directory scripts get as
$SIS_DATA, so a generated script cannot read them even by accident. Private
scores go to a JSONL file rather than the ledger because everything in the
ledger is eventually rendered into a rival's prompt, and an agent that can see
the private board will overfit it exactly the way people overfit the public one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Datasets reachable over plain HTTPS and matching real playground sources.
# `bank` is the UCI bank-marketing data Kaggle synthesised s5e8 from; `adult`
# is UCI census income. Both are binary, mixed numeric/categorical, AUC-scored.
KNOWN = {
    "bank": {
        "url": "https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv",
        "target": "y", "sep": ";",
        # `duration` is only known after the call ends — the source docs say
        # it should be dropped for a realistic model. s5e8 kept it; so do we,
        # to stay faithful to the playground this stands in for.
    },
    # s4e4 was synthesised from abalone and scored by RMSLE on Rings.
    "abalone": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
        "target": "Rings", "sep": ",", "header": None, "metric": "rmsle",
        "names": ["Sex", "Length", "Diameter", "Height", "Whole_weight",
                  "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"],
    },
    # Multiclass: quality as a LABEL (q3..q9), scored MAP@3 like s5e6.
    "wine": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv",
        "target": "quality", "sep": ",", "header": None, "metric": "map@3",
        "as_label": "q",
        "names": ["fixed_acidity", "volatile_acidity", "citric_acid",
                  "residual_sugar", "chlorides", "free_sulfur_dioxide",
                  "total_sulfur_dioxide", "density", "pH", "sulphates",
                  "alcohol", "quality"],
    },
    "adult": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
        "target": "income", "sep": ",", "header": None,
        "names": ["age", "workclass", "fnlwgt", "education", "education_num",
                  "marital_status", "occupation", "relationship", "race", "sex",
                  "capital_gain", "capital_loss", "hours_per_week",
                  "native_country", "income"],
    },
}

PUBLIC_FRAC = 0.2


def is_local(home: Path) -> bool:
    return (Path(home) / ".lb" / "labels.npy").exists()


def prepare(home: Path, source: str, *, target: str = "", test_frac: float = 0.3,
            seed: int = 42, metric: str = "") -> str:
    """Build a competition directory from a CSV (a path, a URL, or a KNOWN name).

    The split is stratified for categorical targets so the public board and CV
    agree on the base rate — an unstratified 30% split of a 12%-positive target
    moves the base rate enough to show up as a CV/LB "gap" that is pure noise.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    home = Path(home)
    spec = KNOWN.get(source, {})
    url = spec.get("url", source)
    target = target or spec.get("target", "")
    if not target:
        raise ValueError("need --target for a dataset that is not in local.KNOWN")
    kw = {"sep": spec.get("sep", ",")}
    if "names" in spec:
        kw.update(header=None, names=spec["names"])
    df = pd.read_csv(url, **kw)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if spec.get("as_label"):
        df[target] = spec["as_label"] + df[target].astype(str)
    for c in df.columns:              # "?" is UCI's missing marker; make it real
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"?": np.nan})
    df.insert(0, "id", np.arange(len(df)))

    # Stratify labels, not numbers: a count target like abalone's Rings has
    # values seen once, which train_test_split refuses to stratify.
    numeric = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 2
    strat = None if numeric else df[target]
    try:
        tr, te = train_test_split(df, test_size=test_frac, random_state=seed,
                                  stratify=strat)
    except ValueError:                  # a label with a single row
        tr, te = train_test_split(df, test_size=test_frac, random_state=seed)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    tr["id"] = np.arange(len(tr))
    te["id"] = np.arange(len(tr), len(tr) + len(te))

    data = home / "input"
    lb = home / ".lb"
    data.mkdir(parents=True, exist_ok=True)
    lb.mkdir(parents=True, exist_ok=True)

    from . import metrics
    both = pd.concat([tr[target], te[target]])
    y_te = metrics.encode_target(both)[len(tr):]
    labels = metrics.classes(both) if metrics.is_multiclass(both) else []
    rng = np.random.default_rng(seed + 1)
    public = rng.random(len(te)) < PUBLIC_FRAC
    np.save(lb / "labels.npy", y_te)
    np.save(lb / "public.npy", public)

    tr.to_csv(data / "train.csv", index=False)
    te.drop(columns=[target]).to_csv(data / "test.csv", index=False)
    # A constant sample submission, like Kaggle's. validate() compares ids and
    # columns against it, so it must use the real target name.
    metric = metric or spec.get("metric", "")
    if labels and ("log" in metric.lower() or "entropy" in metric.lower()):
        sample = pd.DataFrame({"id": te["id"],
                               **{c: 1 / len(labels) for c in labels}})
    elif labels:
        kk = metrics._map_k(metric)
        guess = " ".join(labels[:kk]) if kk else labels[0]
        sample = pd.DataFrame({"id": te["id"], target: guess})
    elif pd.api.types.is_numeric_dtype(tr[target]) and tr[target].nunique() > 2:
        sample = pd.DataFrame({"id": te["id"], target: float(tr[target].mean())})
    else:
        sample = pd.DataFrame({"id": te["id"], target: 0.5})
    sample.to_csv(data / "sample_submission.csv", index=False)
    json.dump({"source": url, "target": target, "seed": seed, "labels": labels,
               "metric": metric, "test_frac": test_frac, "public_frac": PUBLIC_FRAC},
              open(lb / "spec.json", "w"))
    return (f"local competition from {url}: train {len(tr):,}, test {len(te):,} "
            f"({int(public.sum()):,} public / {int((~public).sum()):,} private)")


def score(home: Path, sub_path: Path, metric: str) -> tuple[float, float]:
    """(public, private) for a submission file, aligned by id."""
    import pandas as pd
    from . import metrics
    home = Path(home)
    y = np.load(home / ".lb" / "labels.npy")
    public = np.load(home / ".lb" / "public.npy")
    sample = pd.read_csv(home / "input" / "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    id_col = sample.columns[0]
    sub = sub.set_index(id_col).loc[sample[id_col]]
    labels = json.load(open(home / ".lb" / "spec.json")).get("labels") or []
    if len(sample.columns) > 2:                 # one probability column per class
        p = sub[labels].values.astype(float)
    elif labels:
        # Label submissions: turn them back into a score matrix the metric can
        # read. Top-k strings get descending pseudo-probabilities, so MAP@k
        # and accuracy computed on the matrix equal the label-based values.
        p = np.zeros((len(sub), len(labels)))
        for i, s in enumerate(sub[sample.columns[1]].astype(str)):
            for r, lab in enumerate(s.split()):
                if lab in labels:
                    p[i, labels.index(lab)] = max(p[i, labels.index(lab)], 1.0 / (r + 1))
        p += 1e-9
    else:
        p = sub[sample.columns[1]].values.astype(float)
    return (metrics.score(y[public], p[public], metric),
            metrics.score(y[~public], p[~public], metric))


def submissions_today(home: Path) -> int:
    path = Path(home) / ".lb" / "private.jsonl"
    if not path.exists():
        return 0
    today = time.strftime("%Y-%m-%d", time.gmtime())
    return sum(1 for ln in path.read_text().splitlines()
               if ln.strip() and json.loads(ln).get("ts", "").startswith(today))


def submit(conn, home: Path, exp: dict, data_dir: Path) -> tuple[bool, str]:
    """Validate exactly as for Kaggle, then score against the hidden labels.

    Only the PUBLIC score enters the ledger. The private one is appended to a
    file the prompt builders never read.
    """
    from . import kaggle, store
    sub = Path(exp["submission_path"])
    ok, detail = kaggle.validate(sub, data_dir)
    if not ok:
        return False, f"validation failed: {detail}"
    metric = store.meta(conn, "metric", "auc")
    pub, priv = score(home, sub, metric)
    store.record_submission(conn, exp["id"], exp["cv"], pub, detail)
    with open(Path(home) / ".lb" / "private.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                            "experiment_id": exp["id"], "cv": exp["cv"],
                            "public": pub, "private": priv,
                            "change": exp.get("change", "")}) + "\n")
    return True, (f"submitted exp #{exp['id']} locally ({detail})\n"
                  f"public LB {pub:.5f} (cv {exp['cv']:.5f}, gap {exp['cv'] - pub:+.5f})\n"
                  + store.cv_lb_trust(conn))


def private_report(home: Path) -> str:
    """For the operator, after the fact: did the picks hold up on private?"""
    path = Path(home) / ".lb" / "private.jsonl"
    if not path.exists():
        return "no local submissions yet"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    out = ["  exp |       cv |   public |  private | change"]
    for r in rows:
        out.append(f"{r['experiment_id']:>5} | {r['cv']:.6f} | {r['public']:.6f} | "
                   f"{r['private']:.6f} | {r.get('change', '')[:50]}")
    return "\n".join(out)


def demo() -> None:
    """Build a tiny competition from a synthetic CSV and check the labels stay
    out of $SIS_DATA, the board splits, and a perfect submission scores 1.0."""
    import tempfile

    import pandas as pd
    from . import store
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "c": rng.choice(["a", "b"], n),
                           "t": np.where(x + rng.normal(size=n) * 0.5 > 0.8,
                                         "yes", "no")})
        src = d / "src.csv"
        df.to_csv(src, index=False)
        home = d / "comp"
        msg = prepare(home, str(src), target="t")
        assert "train 1,400" in msg, msg
        assert is_local(home)
        assert not any((home / "input").glob("*.npy")), "labels leaked into input/"
        te = pd.read_csv(home / "input" / "test.csv")
        assert "t" not in te.columns

        y = np.load(home / ".lb" / "labels.npy")
        sub = pd.DataFrame({"id": te["id"], "t": 0.05 + 0.9 * y + rng.random(len(y)) * 0.05})
        sp = home / "sub.csv"
        sub.to_csv(sp, index=False)
        pub, priv = score(home, sp, "auc")
        assert pub == 1.0 and priv == 1.0, (pub, priv)

        conn = store.open_db(home)
        store.set_meta(conn, metric="auc", target="t")
        ok, out = submit(conn, home, {"id": 1, "cv": 0.99, "submission_path": str(sp),
                                      "change": "perfect"}, home / "input")
        assert ok, out
        assert store.best_lb(conn) == 1.0
        assert submissions_today(home) == 1
        # The private score is in the operator's file, never in the ledger.
        assert "private" not in " ".join(
            str(tuple(r)) for r in conn.execute("SELECT * FROM submissions"))
    print("local.py: 8 checks passed")


if __name__ == "__main__":
    demo()
