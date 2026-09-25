"""Everything that touches Kaggle: data in, submissions out, kernels for GPU.

Ported from v1 rather than rewritten, because every awkward detail in here was
learned by watching a real push fail:

* the fetched kernel log is **JSON records**, not plain text
* `SIS_DATA` is unset on a kernel, and competition data lands somewhere under
  `/kaggle/input` that is not the local path — hence the preamble that makes the
  script discover its own data directory
* `machine_shape` in kernel-metadata.json is silently ignored; the CLI's
  `--accelerator` flag is what actually selects the GPU, and asking for a P100
  is refused outright by some competitions
* `competitions leaderboard --show -v` returns a paginated table no CSV reader
  can parse; only `--download` gives a real CSV

The submission validations are the other reason this file exists. Kaggle allows
five submissions a day and each one is the only honest read you get on the test
distribution — a slot spent on a malformed CSV buys nothing at all. Every check
below has cost somebody a slot.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from . import store


def _bin() -> str:
    """The CLI lives beside this interpreter when the venv is not activated."""
    local = Path(sys.executable).parent / "kaggle"
    return str(local) if local.exists() else "kaggle"


def kaggle(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([_bin(), *args], capture_output=True, text=True,
                          timeout=timeout)


def comp_slug(slug: str) -> str:
    """The full competition name. The slug-shaped guess is right for the
    playground series and nothing else, so `init` stores the real one."""
    return f"playground-series-{slug}" if re.fullmatch(r"s\d+e\d+", slug) else slug


# ------------------------------------------------------------------ data in

def download(slug: str, data_dir: Path) -> tuple[bool, str]:
    """Fetch and unzip the competition data.

    A 403 here almost always means the rules have not been accepted — which is
    a browser click with no API equivalent, so it is worth saying so plainly
    rather than surfacing a raw HTTP error.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    out = kaggle("competitions", "download", "-c", comp_slug(slug),
                 "-p", str(data_dir), timeout=1800)
    text = (out.stdout + out.stderr)
    if out.returncode != 0:
        if "403" in text or "Forbidden" in text:
            return False, (f"403 from Kaggle. Accept the rules first: "
                           f"https://www.kaggle.com/competitions/{comp_slug(slug)}/rules")
        return False, text[-500:]
    for z in data_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data_dir)
        z.unlink()
    files = sorted(p.name for p in data_dir.glob("*.csv"))
    return True, f"downloaded {', '.join(files) or 'nothing'} to {data_dir}"


def make_folds(data_dir: Path, n_folds: int = 10, seed: int = 42,
               target: str = "") -> str:
    """Freeze one fold assignment and write it beside the data.

    **Every experiment must score on identical folds or comparing them is
    meaningless.** v1 let each script build its own split, which made a 0.001
    difference between two experiments indistinguishable from split noise —
    and 0.001 is exactly the margin that decides a playground competition.
    Writing the folds once, to disk, removes the whole class of error.

    Stratified when the target is categorical, which keeps the positive rate
    stable across folds and cuts fold-to-fold variance on an imbalanced problem.
    """
    import numpy as np
    import pandas as pd
    train = pd.read_csv(data_dir / "train.csv")
    y = train[target] if target and target in train.columns else None
    idx = np.arange(len(train))
    # Labels are stratified; a numeric target with many values (a regression,
    # or a count like abalone's Rings) is not — StratifiedKFold would treat
    # every distinct value as a class and trip over values seen once.
    if y is not None and (y.nunique() <= 2 or not pd.api.types.is_numeric_dtype(y)):
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        parts = splitter.split(idx, y)
        kind = "stratified"
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(n_folds, shuffle=True, random_state=seed)
        parts = splitter.split(idx)
        kind = "plain"
    folds = np.zeros(len(train), dtype=np.int8)
    for k, (_, va) in enumerate(parts):
        folds[va] = k
    np.save(data_dir / "folds.npy", folds)
    return (f"wrote {n_folds}-fold {kind} split (seed {seed}) to folds.npy "
            f"for {len(train):,} rows")


def profile(data_dir: Path, target: str = "") -> str:
    """A compact description of the data for the agent's prompt.

    Deliberately small: dtypes, cardinality, missing counts, and the target
    balance. The agent writes code against this, so it must be accurate and it
    must fit in a prompt beside everything else.
    """
    import pandas as pd
    tr = pd.read_csv(data_dir / "train.csv")
    te_path = data_dir / "test.csv"
    te = pd.read_csv(te_path) if te_path.exists() else None
    # Column names are NEVER truncated. The agent writes code against this
    # listing, and s6e9 has `Charging_Stations_Near_Home` and `..._Work`, which
    # a 22-char clip renders as the same string twice — an unresolvable
    # ambiguity that produces KeyErrors the debugger cannot reason its way out of.
    w = max((len(c) for c in tr.columns), default=10)
    lines = [f"train {tr.shape[0]:,} x {tr.shape[1]}"
             + (f", test {te.shape[0]:,} x {te.shape[1]}" if te is not None else ""),
             "",
             f"{'column':<{w}}  {'dtype':<9}  {'nunique':>9}  {'n_missing':>10}"]
    for c in tr.columns:
        lines.append(f"{c:<{w}}  {str(tr[c].dtype):<9}  {tr[c].nunique():>9,}  "
                     f"{tr[c].isna().sum():>10,}")
    ss = data_dir / "sample_submission.csv"
    if ss.exists():
        head = pd.read_csv(ss, nrows=2)
        lines.append(f"\nsample_submission.csv columns: {list(head.columns)}; "
                     f"first row: {head.iloc[0].to_dict()}")
    if not tr.isna().any().any():
        # Worth stating positively: it stops the agent inventing missingness
        # features (the s6e8 winner's headline trick) that cannot exist here.
        lines.append("\nNo missing values anywhere in train.")
    if target and target in tr.columns:
        t = tr[target]
        # A non-numeric target is normal (s6e9 ships "Yes"/"No") and calling
        # .mean() on it raises. The agent needs to be TOLD the encoding, not
        # left to infer it — a script that maps the labels the other way round
        # scores a mirror-image AUC and looks like a catastrophic model.
        if pd.api.types.is_numeric_dtype(t):
            lines.append(f"\ntarget `{target}`: numeric, mean {t.mean():.5f}, "
                         f"{t.nunique()} distinct values")
        else:
            vc = t.value_counts(dropna=False)
            pos = str(vc.index[-1]) if len(vc) == 2 else ""
            lines.append(f"\ntarget `{target}`: NON-NUMERIC ({t.dtype}) — you must "
                         f"encode it yourself. Value counts:")
            for val, n in vc.items():
                lines.append(f"    {str(val):<12} {n:>9,}  ({n / len(t):.4f})")
            if pos:
                lines.append(f"  The positive class is `{pos}` (the minority). Map it "
                             f"to 1 and the other to 0, and predict P({pos}).")
            elif len(vc) > 2:
                order = sorted(str(v) for v in vc.index)
                lines.append(f"  MULTICLASS ({len(vc)} labels). Encode as codes in "
                             f"sorted order {order} and save OOF/PRED as (n, "
                             f"{len(vc)}) probability matrices with columns in that "
                             f"order. Write the submission in sample_submission.csv's "
                             f"format (see its columns).")
                lines.append("  RARE CLASSES: a smoke subsample or a fold can lack a "
                             "class entirely, and XGBoost/LightGBM reject labels that "
                             "are not 0..k'-1. Fit each fold on the classes present "
                             "(remap y to contiguous codes), then write the predicted "
                             "columns back into a full (n, k) matrix with zeros for "
                             "absent classes. A wine run died three repairs deep on "
                             "exactly this.")
    return "\n".join(lines)


# ------------------------------------------------------- submissions out

def validate(sub_path: Path, data_dir: Path) -> tuple[bool, str]:
    """Six checks, each of which has cost somebody a daily slot."""
    import pandas as pd
    if not sub_path or not Path(sub_path).exists():
        return False, f"no file at {sub_path}"
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv")

    if list(sub.columns) != list(sample.columns):
        return False, f"columns {list(sub.columns)} != expected {list(sample.columns)}"
    if len(sub) != len(sample):
        return False, f"{len(sub)} rows, expected {len(sample)}"
    id_col, targets = sample.columns[0], list(sample.columns[1:])
    if set(sub[id_col]) != set(sample[id_col]):
        return False, "id set does not match sample_submission"
    for t in targets:                       # every label column, not just the first
        if sub[t].isna().any():
            return False, f"{t}: {sub[t].isna().sum()} NaN predictions"
        if not pd.api.types.is_numeric_dtype(sub[t]):
            # Class labels (argmax, or space-separated top-k for MAP@k). The
            # numeric checks below do not apply; a constant column still does.
            if sub[t].nunique() < 2 and len(sub) > 50:
                return False, f"{t}: every row has the same label — constant?"
            continue
        lo, hi = float(sub[t].min()), float(sub[t].max())
        if lo < -1e-9 or hi > 1 + 1e-9:
            # Only meaningful for probability targets; a regression sample
            # submission will have values outside [0,1] too, so check the
            # sample's own range before complaining.
            slo, shi = float(sample[t].min()), float(sample[t].max())
            if 0 <= slo and shi <= 1:
                return False, f"{t}: probabilities out of [0,1]: min {lo}, max {hi}"
        need = min(10, max(2, len(sub) // 2))
        if sub[t].nunique() < need:
            return False, (f"{t}: only {sub[t].nunique()} distinct predictions in "
                           f"{len(sub)} rows — constant?")
    t = targets[0]
    if not pd.api.types.is_numeric_dtype(sub[t]):
        return True, f"{len(sub)} rows, {t}: {sub[t].nunique()} distinct labels"
    return True, (f"{len(sub)} rows, {t} in [{sub[t].min():.4f}, {sub[t].max():.4f}], "
                  f"mean {sub[t].mean():.4f}, {sub[t].nunique()} distinct")


def submissions_today(slug: str) -> int:
    """How many of today's five slots are already gone.

    Asked before every send, because discovering the daily cap by having a
    submission rejected wastes the attempt and the wall clock around it.
    """
    out = kaggle("competitions", "submissions", "-c", comp_slug(slug), "--csv")
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines)
                  if "date" in ln.lower().split(",")), None)
    if start is None:
        return 0
    # UTC, not local. Kaggle stamps submissions in UTC and resets the daily
    # quota at UTC midnight. Comparing against the local date means that for
    # every hour the local date runs ahead of UTC (5.5 of them in IST) this
    # returns 0 while slots are actually spent — so the 5/day guard silently
    # stops guarding exactly when someone is most likely to be running
    # overnight.
    today = time.strftime("%Y-%m-%d", time.gmtime())
    n = 0
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if (row.get("date") or "").startswith(today):
            n += 1
    return n


def submit(conn, slug: str, exp: dict, data_dir: Path, *,
           poll: bool = True) -> tuple[bool, str]:
    """Validate, send, and pair the CV with the LB score.

    That pairing is the entire reason a submission is worth spending: it is the
    only evidence about whether our local CV is telling the truth.
    """
    sub = Path(exp["submission_path"])
    ok, detail = validate(sub, data_dir)
    if not ok:
        return False, f"validation failed: {detail}"

    # Last line of defence: does this experiment's OOF still reproduce the score
    # it claims? exp38 reached the point of being submitted with a ledger CV of
    # 0.945587 and an array that rescores to 0.500380 — right length, all folds
    # filled, no zeros, simply misaligned with the label order. The run-time
    # guard did not fire on whichever path produced it, and a submission slot is
    # the most expensive place to discover that. Cheap to check here, and this
    # is the only gate every submission passes through.
    try:
        import numpy as _np
        import pandas as _pd
        from . import metrics as _m
        target = store.meta(conn, "target", "")
        oof_p = exp.get("oof_path") or ""
        if oof_p and target and Path(oof_p).exists():
            _y = _m.encode_target(_pd.read_csv(Path(data_dir) / "train.csv",
                                               usecols=[target])[target])
            _o = _np.load(oof_p)
            if len(_o) == len(_y):
                actual = _m.score(_y, _o, store.meta(conn, "metric", "auc"))
                if abs(actual - (exp.get("cv") or 0)) > 0.01:
                    return False, (f"REFUSING to submit exp #{exp['id']}: its OOF "
                                   f"scores {actual:.6f} but the ledger claims "
                                   f"{exp['cv']:.6f}. The saved predictions do not "
                                   f"match the reported score — the model is not "
                                   f"what the ledger says it is.")
    except Exception:
        pass                      # never block a submission on the checker itself

    msg = f"sis2 exp#{exp['id']} cv={exp['cv']:.5f} {(exp.get('change') or '')[:60]}"
    handle = exp.get("handle") or ""
    if handle:
        # A CODE competition takes a kernel + version, not a file; `-f` then
        # names the output file to read rather than a path to upload.
        out = kaggle("competitions", "submit", "-c", comp_slug(slug), "-k", handle,
                     "-v", str(kernel_version(handle)), "-f", sub.name, "-m", msg)
    else:
        out = kaggle("competitions", "submit", "-c", comp_slug(slug),
                     "-f", str(sub), "-m", msg)
    if out.returncode != 0:
        return False, f"kaggle rejected it: {(out.stdout + out.stderr)[-300:]}"
    if not poll:
        store.record_submission(conn, exp["id"], exp["cv"], None, detail)
        return True, f"submitted exp #{exp['id']} ({detail})"
    lb, note = fetch_score(conn, slug, exp["id"], exp["cv"])
    return True, f"submitted exp #{exp['id']} ({detail})\n{note}"


def fetch_score(conn, slug: str, exp_id: int, cv: float, *, tries: int = 15,
                wait: int = 20) -> tuple[float | None, str]:
    for attempt in range(tries):
        out = kaggle("competitions", "submissions", "-c", comp_slug(slug), "--csv")
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        start = next((i for i, ln in enumerate(lines)
                      if "date" in ln.lower().split(",")), None)
        if start is not None:
            row = next(csv.DictReader(io.StringIO("\n".join(lines[start:]))), None)
            sc = (row or {}).get("publicScore", "").strip()
            if sc and sc.lower() not in ("none", "nan", ""):
                lb = float(sc)
                store.record_submission(conn, exp_id, cv, lb,
                                        (row or {}).get("description", ""))
                return lb, (f"public LB {lb:.5f} (cv {cv:.5f}, gap {cv - lb:+.5f})\n"
                            + store.cv_lb_trust(conn))
        if attempt < tries - 1:
            time.sleep(wait)
    store.record_submission(conn, exp_id, cv, None, "not scored yet")
    return None, "not scored yet — re-check with `sis2 score`"


def leaderboard(slug: str, workdir: Path) -> dict:
    """Where the board actually sits. `--show -v` returns a paginated table no
    CSV reader can parse; only `--download` gives real CSV."""
    d = workdir / "leaderboard"
    d.mkdir(parents=True, exist_ok=True)
    kaggle("competitions", "leaderboard", "-c", comp_slug(slug), "--download",
           "-p", str(d), timeout=180)
    for z in d.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    scores = []
    for f in d.glob("*.csv"):
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                col = next((k for k in row if k and "score" in k.lower()), None)
                if col and row[col]:
                    try:
                        scores.append(float(row[col]))
                    except ValueError:
                        pass
    if not scores:
        return {"known": False}
    scores.sort(reverse=True)
    n = len(scores)
    return {"known": True, "teams": n, "top": scores[0],
            "p10": scores[max(int(n * 0.10) - 1, 0)],
            "median": scores[n // 2]}


# ------------------------------------------------------------- GPU kernels

# Assembles ONE directory holding both the competition CSVs and folds.npy,
# because they arrive from two different mounts and the contract tells every
# script that $SIS_DATA contains all of them.
#
# folds.npy is ours, not Kaggle's — it is generated locally at `init` and is
# what makes every experiment comparable. A kernel that could not see it would
# invent its own split and silently produce a score that means nothing next to
# the local ones, which is the single worst failure this system can have. So it
# ships as an attached dataset and is linked in here.
KERNEL_PREAMBLE = """\
import os as _os, glob as _glob
_d = "/kaggle/working/_sisdata"
_os.makedirs(_d, exist_ok=True)
for _p in _glob.glob("/kaggle/input/**/*.csv", recursive=True) + \\
          _glob.glob("/kaggle/input/**/folds.npy", recursive=True):
    _t = _os.path.join(_d, _os.path.basename(_p))
    if not _os.path.exists(_t):
        try:
            _os.symlink(_p, _t)
        except OSError:
            import shutil as _sh; _sh.copy(_p, _t)
if not _os.path.exists(_os.path.join(_d, "folds.npy")):
    raise SystemExit("folds.npy not mounted — attach the sis2 folds dataset; "
                     "without the shared folds this run is not comparable")
_os.environ["SIS_DATA"] = _d
# Kaggle does not always give the accelerator that was asked for. A P100
# (sm_60) has no kernels in current torch builds, so `.to("cuda")` raises and
# the whole experiment is lost over hardware we did not choose. Probe once and
# publish the verdict; the script below should honour SIS_DEVICE.
try:
    import torch as _t
    _ok = _t.cuda.is_available()
    if _ok:
        _t.zeros(1).cuda()          # sm check: allocation is what actually fails
    _os.environ["SIS_DEVICE"] = "cuda" if _ok else "cpu"
except Exception as _e:
    _os.environ["SIS_DEVICE"] = "cpu"
    print(f"GPU unusable ({str(_e)[:80]}) — falling back to CPU", flush=True)
_os.environ.setdefault("SIS_OUT", "/kaggle/working")
_os.environ.setdefault("SIS_FRAC", "1.0")
_os.environ.setdefault("SIS_THREADS", "4")
# ---- sis2 preamble ends; the experiment follows ----
"""


def ensure_folds_dataset(data_dir: Path, workdir: Path) -> str:
    """Publish folds.npy as a private Kaggle dataset; return its ref.

    Created once and reused. Returns '' on failure, which callers treat as
    "GPU routing unavailable" rather than falling back to an unshared split —
    a kernel with the wrong folds is worse than no kernel at all.
    """
    user = kaggle_username()
    if not user:
        return ""
    ref = f"{user}/sis2-folds"
    d = workdir / "folds_dataset"
    d.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(data_dir / "folds.npy", d / "folds.npy")
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "sis2 folds", "id": ref,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    out = kaggle("datasets", "status", ref, timeout=60)
    exists = "ready" in (out.stdout + out.stderr).lower()
    if exists:
        r = kaggle("datasets", "version", "-p", str(d), "-m", "refresh",
                   "--dir-mode", "zip", timeout=600)
    else:
        r = kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip",
                   timeout=600)
    text = (r.stdout + r.stderr).lower()
    if r.returncode != 0 and "already exists" not in text:
        return ""
    return ref


def run_kernel(code: str, workdir: Path, name: str, slug: str, *,
               gpu: bool = True, dataset_sources: list[str] | None = None,
               poll_every: int = 30, timeout: int = 3600) -> dict:
    """Push a kernel and block until it finishes. Same result shape as run_local.

    Blocking rather than async: the loop is sequential, and a synchronous wait
    keeps the ledger honest about what is running. Kernel startup is a minute
    or two of overhead, so this only pays off for work that is genuinely slow
    locally — which is exactly what the GPU routing is for.
    """
    pushed = push_kernel(code, workdir, name, slug, gpu=gpu,
                         dataset_sources=dataset_sources)
    if pushed.get("failure"):
        return pushed
    handle = pushed["handle"]
    deadline = time.monotonic() + timeout      # sleep-safe; see run._run_watched
    while time.monotonic() < deadline:
        time.sleep(poll_every)
        res = poll_kernel(handle, workdir)
        if not res.get("pending"):
            return res
    return {"failure": "timeout",
            "error": f"kernel {handle} still running after {timeout}s"}

MACHINE_SHAPE = os.getenv("SIS_MACHINE_SHAPE", "GPU T4 x2")


def kaggle_username() -> str:
    out = kaggle("config", "view", timeout=60)
    for line in (out.stdout or "").splitlines():
        if line.strip().startswith("- username:"):
            name = line.split(":", 1)[1].strip()
            if name and name.lower() != "none":
                return name
    # Fall back to the owner of our own kernels. Skip the header row — v1
    # matched it on "/" and returned a header fragment as the username.
    out = kaggle("kernels", "list", "-m", "-v", timeout=60)
    for line in (out.stdout or "").splitlines()[1:]:
        if "/" in line:
            return line.split("/")[0].strip()
    return os.getenv("KAGGLE_USERNAME", "").strip()


def push_kernel(code: str, workdir: Path, name: str, slug: str, *,
                gpu: bool = True, dataset_sources: list[str] | None = None) -> dict:
    """Push and return a handle immediately. Async by nature — the ledger row
    goes to 'running' and the loop polls it later."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "kernel.py").write_text(KERNEL_PREAMBLE + code, encoding="utf-8")
    handle = f"{kaggle_username()}/{name}"
    (workdir / "kernel-metadata.json").write_text(json.dumps({
        "id": handle, "title": name, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": bool(gpu), "enable_internet": False,
        # BOTH this field and the --accelerator flag below. v1 set both; the
        # port kept only the flag, and a bare `enable_gpu: true` makes Kaggle
        # hand out a P100 (sm_60) — for which current torch builds ship no
        # kernels, so the run dies with "capability sm_60 is not compatible".
        # Neither alone has proven sufficient, so send both.
        **({"machine_shape": MACHINE_SHAPE} if gpu else {}),
        "competition_sources": [comp_slug(slug)],
        "dataset_sources": list(dataset_sources or []), "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    push = ["kernels", "push", "-p", str(workdir)]
    # --accelerator, not just the metadata field: `machine_shape` is read from
    # kernel-metadata.json and silently ignored, so Kaggle hands out a P100 and
    # current torch builds ship no sm_60 kernels.
    if gpu:
        push += ["--accelerator", MACHINE_SHAPE]
    out = kaggle(*push)
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0 or "successfully pushed" not in text.lower():
        return {"failure": "bug", "error": text[-2000:]}
    return {"handle": handle}


def kernel_status(handle: str) -> str:
    out = kaggle("kernels", "status", handle)
    text = (out.stdout + out.stderr).lower()
    for state in ("running", "queued", "complete", "error", "cancel"):
        if state in text:
            return {"queued": "running", "cancel": "error"}.get(state, state)
    return "unknown"


def kernel_version(handle: str) -> int:
    """Naming the wrong version submits older code, so read it rather than
    assume; 1 is the only safe guess for a kernel we just pushed."""
    out = kaggle("kernels", "status", handle)
    hits = re.findall(r"version\s*[:=]?\s*(\d+)",
                      (out.stdout or "") + (out.stderr or ""), re.I)
    return int(hits[-1]) if hits else 1


def _log_text(path: Path) -> str:
    """Kaggle returns the log as JSON records, not plain text."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        return "".join(rec.get("data", "") for rec in json.loads(raw))
    except Exception:
        return raw


def poll_kernel(handle: str, workdir: Path) -> dict:
    """{'pending': True} while it runs; otherwise the same shape as run_local."""
    from .run import _path_sentinel, _sentinel
    state = kernel_status(handle)
    if state == "running":
        return {"pending": True}
    out = kaggle("kernels", "output", handle, "-p", str(workdir), timeout=600)
    log = "\n".join(_log_text(p) for p in workdir.glob("*.log"))
    if state == "error":
        return {"failure": "bug", "error": (log or out.stderr)[-3000:]}
    cv, cv_std = _sentinel("CV_SCORE", log), _sentinel("CV_STD", log)
    if cv is None:
        return {"failure": "bug",
                "error": f"kernel finished ({state}) but printed no CV_SCORE. "
                         f"Log tail: {log[-1000:]}"}
    if cv_std is None:
        return {"failure": "bug", "error": "kernel printed CV_SCORE but no CV_STD"}
    res = {"cv": cv, "cv_std": cv_std, "runtime_s": 0.0, "stdout": log[-6000:],
           "handle": handle}
    for key, sent in (("oof", "OOF"), ("pred", "PRED"), ("submission", "SUBMISSION")):
        name = _path_sentinel(sent, log)
        p = workdir / Path(name).name if name else None
        res[f"{key}_path"] = str(p) if p and p.exists() else ""
    return res


def demo() -> None:
    """Self-check the pure logic. Anything that needs the network is exercised
    by `sis2 init` against the real competition, not here."""
    import tempfile

    import numpy as np
    import pandas as pd

    assert comp_slug("s6e9") == "playground-series-s6e9"
    assert comp_slug("titanic") == "titanic"          # only the series is guessable

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        n = 500
        pd.DataFrame({"id": range(n), "f": np.random.rand(n),
                      "y": np.random.randint(0, 2, n)}).to_csv(d / "train.csv", index=False)
        pd.DataFrame({"id": range(n, n + 100),
                      "y": 0.5}).to_csv(d / "sample_submission.csv", index=False)

        msg = make_folds(d, n_folds=5, seed=42, target="y")
        folds = np.load(d / "folds.npy")
        assert set(folds) == {0, 1, 2, 3, 4} and len(folds) == n, msg
        # Stratified: every fold carries a similar positive rate.
        rates = [folds[folds == k].size for k in range(5)]
        assert max(rates) - min(rates) <= 1, rates

        good = pd.DataFrame({"id": range(n, n + 100), "y": np.random.rand(100)})
        good.to_csv(d / "sub.csv", index=False)
        ok, why = validate(d / "sub.csv", d)
        assert ok, why

        # Each failure mode is caught with a message that names the problem.
        const = good.copy(); const["y"] = 0.5
        const.to_csv(d / "c.csv", index=False)
        ok, why = validate(d / "c.csv", d)
        assert not ok and "constant" in why, why

        short = good.iloc[:50]
        short.to_csv(d / "s.csv", index=False)
        ok, why = validate(d / "s.csv", d)
        assert not ok and "rows" in why, why

        nan = good.copy(); nan.loc[0, "y"] = np.nan
        nan.to_csv(d / "n.csv", index=False)
        ok, why = validate(d / "n.csv", d)
        assert not ok and "NaN" in why, why

        prof = profile(d, target="y")
        assert "train 500" in prof and "target `y`" in prof, prof

        # A string target must not crash the profile — s6e9 ships "Yes"/"No",
        # and .mean() on that raises. It must also name the positive class,
        # since guessing the encoding backwards mirrors the AUC.
        pd.DataFrame({"id": range(n), "f": np.random.rand(n),
                      "y": np.where(np.random.rand(n) < 0.2, "Yes", "No")}
                     ).to_csv(d / "train.csv", index=False)
        prof = profile(d, target="y")
        assert "NON-NUMERIC" in prof and "positive class is `Yes`" in prof, prof

    print("kaggle.py: 9 checks passed")


if __name__ == "__main__":
    demo()


# --------------------------------------------------------------- parallel fan-out

def push_many(jobs: list[dict], slug: str, *, gpu: bool = False,
              dataset_sources: list[str] | None = None) -> list[dict]:
    """Push several kernels at once and return handles immediately.

    `run_kernel` blocks, which is right for one experiment but wastes the whole
    point of remote compute: Kaggle will happily run several kernels at the same
    time while the laptop sits idle. Each job is {'exp_id', 'code', 'workdir'}.

    Defaults to CPU. GBDTs gain nothing from a GPU here, CPU kernels do not
    consume the weekly GPU quota, and asking for a GPU is what lands us on a
    P100 that current torch cannot use.
    """
    out = []
    for j in jobs:
        name = f"sis2-{slug}-exp{j['exp_id']}"
        res = push_kernel(j["code"], Path(j["workdir"]) / "kernel", name,
                          comp_slug(slug), gpu=gpu,
                          dataset_sources=dataset_sources)
        out.append({**j, **res})
    return out


def poll_many(pushed: list[dict], *, poll_every: int = 45,
              timeout: int = 5400) -> list[dict]:
    """Wait for a batch of kernels, returning each one's result as it lands.

    Polls round-robin rather than in order, so a slow kernel does not hide the
    results of fast ones that finished behind it.
    """
    pending = {j["handle"]: j for j in pushed if j.get("handle")}
    done = [{**j, "failure": "bug", "error": j.get("error", "push failed")}
            for j in pushed if not j.get("handle")]
    deadline = time.monotonic() + timeout      # sleep-safe
    while pending and time.monotonic() < deadline:
        time.sleep(poll_every)
        for handle in list(pending):
            j = pending[handle]
            res = poll_kernel(handle, Path(j["workdir"]) / "kernel")
            if not res.get("pending"):
                done.append({**j, **res})
                pending.pop(handle)
    for handle, j in pending.items():
        done.append({**j, "failure": "timeout",
                     "error": f"kernel {handle} still running after {timeout}s"})
    return done
