"""Run one experiment and get back numbers you can compare.

The contract between the agent and the code it writes is five stdout lines.
No JSON parsing of arbitrary output, no return values plumbed through a
subprocess — five greppable sentinels:

    CV_SCORE: 0.97042
    CV_STD:   0.00031
    OOF:      oof.npy
    PRED:     test.npy
    SUBMISSION: submission.csv

**OOF and PRED are the v2 change and the whole point.** v1 saved only a score
and a submission file, which meant every experiment was a dead end the moment
it finished: you could rank experiments but never ask *where* one was wrong, and
the final ensemble had nothing to blend but other people's published arrays.
Saving the out-of-fold and test predictions costs one `np.save` per run and
turns the experiment log into a dataset — `analyst.py` reads it to find where
the champion is losing, `blend.py` reads it to build the ensemble. Everything
downstream of here is a query over saved arrays.

CV_STD is mandatory for the same reason it was in v1: a mean with no fold
spread makes every comparison unfalsifiable, so a run that omits it is a
failure, not a result.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOCAL_TIMEOUT = 3600          # 60 min; longer than that belongs on a kernel
FRAC_ENV = "SIS_FRAC"         # scripts honour this; 0.05 for smoke runs
DATA_ENV = "SIS_DATA"         # where the competition CSVs live
THREADS_ENV = "SIS_THREADS"   # how many cores a script may use
OUT_ENV = "SIS_OUT"           # where to write oof.npy / test.npy
IDX_ENV = "SIS_IDX"           # smoke runs only: the exact rows to subsample to

# Leave headroom. A laptop pinned across every core throttles, cooks, and is
# unusable while a run is going — and overnight runs are the normal case here.
CPU_FRACTION = float(os.getenv("SIS_CPU_FRACTION", "0.7"))

# A hard ceiling on one experiment's memory. setrlimit(RLIMIT_AS) does NOT work
# on macOS (v1 proved this: a 20GB allocation sailed through a supposed 8GB
# cap), so the parent polls RSS instead. Without it a runaway script doesn't
# fail — it swaps the whole machine until the run is ruined anyway.
MEM_LIMIT_GB = float(os.getenv("SIS_MEM_LIMIT_GB", "9"))

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def thread_budget() -> int:
    return max(1, int((os.cpu_count() or 4) * CPU_FRACTION))


# Below this much free RAM the machine is already swapping and another
# experiment will make it unusable rather than faster.
MIN_FREE_GB = float(os.getenv("SIS_MIN_FREE_GB", "3.0"))
MIN_DISK_GB = float(os.getenv("SIS_MIN_DISK_GB", "5.0"))
RESERVE_GB = float(os.getenv("SIS_RESERVE_GB", "2.0"))   # left for the OS + editor


def available_gb() -> float:
    """Free + inactive + speculative pages, i.e. what a new process can take.

    `Pages free` alone badly understates macOS headroom — inactive pages are
    reclaimable and routinely dwarf free ones. On Linux (cloud containers,
    Kaggle kernels) `MemAvailable` in /proc/meminfo is the same idea, computed
    by the kernel. Returns 0.0 if neither can be read, which callers treat as
    "unknown, do not gate".
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = 0
        for kind in ("Pages free", "Pages inactive", "Pages speculative"):
            m = re.search(rf"{kind}:\s+(\d+)", out)
            if m:
                pages += int(m.group(1))
        return pages * page / 1024 ** 3
    except Exception:
        return 0.0


def free_disk_gb(path: Path) -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1024 ** 3
    except Exception:
        return 0.0


def system_ok(workdir: Path) -> tuple[bool, str, float]:
    """Is it safe to start another experiment, and what memory cap should it get?

    The fixed `MEM_LIMIT_GB` was measured against nothing: on this 16GB machine
    it was set to 9GB while only 6.4GB was actually available, so a script that
    grew to 8GB would sit under the cap and swap the whole laptop instead of
    being killed. The cap is now whatever is genuinely free, minus a reserve
    for the OS — and never more than the configured ceiling.
    """
    avail = available_gb()
    disk = free_disk_gb(workdir)
    if disk and disk < MIN_DISK_GB:
        return False, (f"only {disk:.1f}GB disk free (need {MIN_DISK_GB}GB). "
                       f"Each experiment writes ~30MB of arrays and submissions."), 0.0
    if avail and avail < MIN_FREE_GB:
        return False, (f"only {avail:.1f}GB RAM available (need {MIN_FREE_GB}GB) — "
                       f"the machine is already under pressure; starting another "
                       f"experiment would swap rather than speed anything up."), 0.0
    cap = MEM_LIMIT_GB if not avail else min(MEM_LIMIT_GB, max(2.0, avail - RESERVE_GB))
    return True, "", cap


# Stated once, embedded verbatim in the agent's prompt, so the rules the model
# is told and the rules the parser enforces cannot drift apart.
CONTRACT = f"""\
The script you write must:

1. Read competition data from the directory in ${DATA_ENV}. Never hardcode a path.
2. Honour ${FRAC_ENV}, a float in (0, 1]: subsample TRAINING rows to that
   fraction before doing anything expensive. Smoke runs use 0.05, which is what
   makes a broken script fail in 40 seconds instead of 40 minutes.
3. Use the SHARED FOLDS. Load them with:
       folds = np.load(os.path.join(os.environ["{DATA_ENV}"], "folds.npy"))
   and use `folds == k` as the validation mask for fold k. Do NOT invent your
   own split — every experiment must be scored on identical folds or the
   comparisons between them mean nothing.
   **`folds` has one entry per FULL training row.** If you subsample for
   ${FRAC_ENV}, you must subsample `folds` with the SAME indices, or the
   boolean mask will be 668k long against a 33k frame and every fold raises
   `IndexError: Boolean index has wrong length`. **When ${IDX_ENV} is set, the
   harness has chosen the rows — use exactly those, in that order** (the saved
   OOF is checked against their labels). Do it in one step:
       if os.environ.get("{IDX_ENV}"):
           idx = np.load(os.environ["{IDX_ENV}"])
       elif frac < 1:
           idx = rng.choice(len(train), int(len(train) * frac), replace=False)
       else:
           idx = np.arange(len(train))
       train, folds = train.iloc[idx].reset_index(drop=True), folds[idx]
   Keep `oof` the same length as the (possibly subsampled) train frame.
   **Iterate over EVERY fold — `for k in np.unique(folds)`.** Never hardcode a
   count like `n_folds = 5`: three earlier experiments did that against a
   10-fold assignment, ran half the folds, and left half of `oof` as zeros. The
   score they printed was real but not comparable to anything else, and the
   saved array was useless. The saved OOF is CHECKED against the score you
   print, and a mismatch fails the run.
4. Print, each on its own line:
       CV_SCORE: <mean across folds>
       CV_STD: <standard deviation across folds>
   Both are mandatory. A mean without a spread makes every comparison
   unfalsifiable and the run is rejected without it.
5. Save predictions into ${OUT_ENV} and print their names:
       OOF: oof.npy      (1-D, length == len(train), out-of-fold predictions)
       PRED: test.npy    (1-D, length == len(test), mean over fold models)
   For a MULTICLASS target save (n, k) class-probability matrices instead,
   columns in SORTED label order (sorted(str(label))). The harness scores the
   matrix itself — accuracy, MAP@k, log loss — so do not save labels.
   These are mandatory. They are what later error analysis and the final
   ensemble are built from; an experiment without them cannot be studied or
   reused, and is worth far less than the compute it cost.
6. Write a submission CSV and print:
       SUBMISSION: <filename>
7. Be one self-contained python3 file. No internet, no shell, no arguments.
   Set every random seed you use.
8. Import only packages listed as available below. An ImportError wastes a
   whole cycle and teaches nothing.
9. A **Kaggle T4 GPU is available and free**, and you should use it whenever the
   work is genuinely heavy — neural nets, large boosting sweeps, anything that
   would take more than a few minutes on 7 laptop cores. Opt in simply by
   writing GPU code; the harness detects it and runs the full job on Kaggle
   instead of locally, with the same $SIS_DATA contract and the shared folds
   mounted. Any of these route the run to a GPU kernel:
       import torch                     (and use .to("cuda"))
       XGBClassifier(device="cuda", tree_method="hist")
       CatBoostClassifier(task_type="GPU")
       LGBMClassifier(device="gpu")
   **$SIS_DEVICE is ALWAYS set, locally too** — "cuda", "mps" (Apple GPU) or
   "cpu", whatever this machine has. Use it for any torch work:
   `torch.device(os.environ["SIS_DEVICE"])`. A CPU-only neural net on 668k
   rows x 10 folds does not finish inside the timeout; one experiment already
   died that way.
   On a GPU kernel the harness probes the hardware and sets $SIS_DEVICE to
   "cuda" or "cpu". USE IT — `torch.device(os.environ.get("SIS_DEVICE","cpu"))`
   — never hardcode "cuda". Kaggle does not always hand out the accelerator
   that was requested, and a hardcoded device turns a P100 assignment into a
   lost experiment instead of a slower one.
   Startup costs 1-2 minutes, so it is NOT worth it for a job that finishes
   locally in under ~3 minutes — a plain LightGBM on this data takes about 60
   seconds and should stay local. Use the GPU for depth, not for habit.
10. MEMORY IS TIGHT — roughly 4-6GB per run on a 16GB laptop, and the ceiling
   adapts to what is free at the time. Two experiments have already been killed
   materialising a dense expansion of the full 668k-row frame (spline bases,
   interaction dummies, one-hot crosses). If your design needs an expansion,
   keep it SPARSE (scipy.sparse, and a solver that accepts sparse input), or
   chunk it, or fit on a subsample. `float32` rather than `float64` throughout.
   A design that only fits in memory at 100k rows is not a design that works here.
11. NEVER use all cores. Read `int(os.environ.get("{THREADS_ENV}", 4))` and pass
   it to every parallel knob you touch — LightGBM `num_threads`, XGBoost
   `nthread`, sklearn `n_jobs`. Do not pass -1 or os.cpu_count(). This runs on a
   laptop that has to stay usable; a run that pins every core gets throttled by
   the OS and finishes no sooner."""

_LIBS = ("pandas", "numpy", "sklearn", "scipy", "lightgbm", "xgboost",
         "catboost", "torch", "pytabkit")
_PKG_CACHE: str | None = None


def available_packages(refresh: bool = False) -> str:
    """Actually import each library — `find_spec` is not good enough.

    find_spec only proves a directory exists. In v1, lightgbm passed that check
    while every import died at dlopen for a missing libomp, so the model was
    told to use a package that could not load. A fact stated to the model has to
    be the fact the subprocess will experience.
    """
    global _PKG_CACHE
    if _PKG_CACHE is not None and not refresh:
        return _PKG_CACHE
    import importlib
    have, missing = [], []
    for lib in _LIBS:
        try:
            importlib.import_module(lib)
            have.append(lib)
        except Exception:      # ImportError, OSError from dlopen, anything
            missing.append(lib)
    _PKG_CACHE = (f"available: {', '.join(have) or 'none'}\n"
                  f"NOT installed (do not import): {', '.join(missing) or 'none'}")
    return _PKG_CACHE


def _rss_gb(pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) / 1024 ** 2 if out.isdigit() else 0.0
    except Exception:
        return 0.0


def _run_watched(cmd, cwd, env, timeout: int, mem_gb: float):
    """Run the child while watching its memory, killing it if it runs away."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, preexec_fn=(lambda: os.nice(10)) if os.name == "posix" else None)
    # monotonic, NOT time.time(): on macOS time.monotonic() does not advance
    # while the machine is asleep, whereas wall clock does. With wall clock, a
    # closed laptop lid makes every running experiment get killed as a spurious
    # "timeout" the moment it wakes — the run was fine, the clock moved. That is
    # almost certainly what killed the only transformer this campaign produced.
    deadline = time.monotonic() + timeout
    killed = None
    # 0.25s: fast enough to catch a script that balloons in a second or two,
    # slow enough that an hour-long run costs only a few thousand cheap `ps`es.
    while proc.poll() is None:
        if time.monotonic() > deadline:
            killed = "timeout"
        elif _rss_gb(proc.pid) > mem_gb:
            killed = "oom"
        if killed:
            proc.kill()
            break
        time.sleep(0.25)
    out, err = proc.communicate()
    return proc.returncode, out, err, killed


def _sentinel(name: str, text: str) -> float | None:
    hits = re.findall(name + r"\s*:\s*([-+0-9.eE]+)", text)
    try:
        return float(hits[-1]) if hits else None   # last wins
    except ValueError:
        return None


def _path_sentinel(name: str, text: str) -> str:
    hits = re.findall(name + r"\s*:\s*(\S+)", text)
    return hits[-1] if hits else ""


def _classify(stderr: str, returncode: int) -> str:
    """Why did it die? A bug is not evidence against the hypothesis, so this
    label decides whether the idea gets retried or written off."""
    low = stderr.lower()
    if ("memoryerror" in low or "out of memory" in low or returncode == -9
            or "cannot allocate" in low or "bad_alloc" in low):
        return "oom"
    if "keyboardinterrupt" in low:
        return "timeout"
    return "bug"


def _check_arrays(workdir: Path, res: dict, n_train: int, n_test: int) -> str:
    """Verify the saved arrays are the shape they claim to be.

    A wrong-length OOF is worse than a missing one: it loads fine, blends fine,
    and quietly corrupts every downstream comparison. Cheap to check here, very
    expensive to discover three days later in an ensemble.
    """
    import numpy as np
    for key, want in (("oof", n_train), ("pred", n_test)):
        p = res.get(f"{key}_path", "")
        if not p or not Path(p).exists():
            return f"printed {key.upper()} but no file at {p}"
        try:
            arr = np.load(p)
        except Exception as e:
            return f"{key}.npy will not load: {e}"
        # 1-D for binary and regression; (n, k) class probabilities for a
        # multiclass target, columns in sorted-label order.
        if arr.ndim not in (1, 2):
            return f"{key} has shape {arr.shape}; must be 1-D, or (n, k) for multiclass"
        if want and len(arr) != want:
            return f"{key} has {len(arr)} rows, expected {want}"
        if not np.isfinite(arr).all():
            return f"{key} contains NaN or inf"
    return ""


def _oof_agrees(oof_path: str, y, metric: str, claimed: float,
                tol: float = 0.01) -> str:
    """Does the saved OOF actually reproduce the score the script claimed?

    This is the check that catches everything shape-and-finiteness cannot.
    Three s6e9 experiments hardcoded `n_folds = 5`, looped `range(5)` against a
    10-fold assignment, and left HALF the OOF array as zeros. Each reported an
    honest 5-fold CV of ~0.9415 — computed only over the folds it ran — while
    the saved array scored 0.610. The length was right, every value was finite,
    and nothing downstream could tell.

    Both consumers of that array would have been silently corrupted: the
    analyst would have "diagnosed" weaknesses in a half-zero prediction vector,
    and the blend would have averaged garbage into the ensemble. Comparing the
    array against its own claim is cheap and closes the whole class.
    """
    from . import metrics as _m
    import numpy as _np
    try:
        oof = _np.load(oof_path)
    except Exception as e:
        return f"oof.npy will not load: {e}"
    if len(oof) != len(y):
        return f"oof has {len(oof)} rows, y has {len(y)}"
    actual = _m.score(y, oof, metric)
    if not _np.isfinite(actual):
        return "scoring the saved oof produced a non-finite value"
    # Generous tolerance: fold-averaged and pooled scores differ slightly even
    # when everything is correct. A real corruption is off by orders more.
    if abs(actual - claimed) > tol:
        zeros = float((oof == 0).mean())
        extra = (f" {zeros:.0%} of the array is exactly 0 — folds were probably "
                 f"skipped; iterate over np.unique(folds), never a hardcoded count."
                 if zeros > 0.02 else "")
        return (f"script printed CV_SCORE {claimed:.5f} but the saved OOF scores "
                f"{actual:.5f} on the same metric.{extra} The array must be the "
                f"predictions the score was computed from.")
    return ""


def run_local(code: str, workdir: Path, *, frac: float = 1.0,
              data_dir: Path | str = "./input", timeout: int = LOCAL_TIMEOUT,
              n_train: int = 0, n_test: int = 0,
              y=None, metric: str = "") -> dict:
    """Run one experiment here. Returns a dict the store can consume directly.

    `frac` is the smoke-test lever: run at 0.05 first, catch the traceback in 40
    seconds instead of 40 minutes, then rerun at 1.0. Most of what any agent
    generates fails for boring reasons, and paying full data cost to discover a
    typo is the biggest waste of wall clock in the whole system.
    """
    # Resolve BEFORE use. The child runs with cwd=workdir, so passing a
    # relative script path makes it resolve against itself —
    # `work/s6e9/exp1/work/s6e9/exp1/run.py`. The resulting "No such file"
    # then reaches the debugger as if the generated script were broken, and it
    # burns three repair attempts on code that was always fine.
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "run.py").write_text(code, encoding="utf-8")
    threads = thread_budget()
    device = "cpu"
    try:
        import torch as _torch
        if _torch.backends.mps.is_available():
            device = "mps"
        elif _torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    env = {**os.environ, "SIS_DEVICE": device, FRAC_ENV: str(frac),
           DATA_ENV: str(Path(data_dir).resolve()),
           OUT_ENV: str(workdir.resolve()),
           THREADS_ENV: str(threads),
           **{v: str(threads) for v in _THREAD_VARS}}

    # The HARNESS picks the smoke subsample, not the script. When the script
    # chose its own `rng.choice` rows we could not know which rows its OOF
    # covered, so every integrity check had to be skipped on smoke runs — and
    # five experiments (9, 20, 37, 38, 59) sailed through smoke, burned a full
    # 10-fold run, and only then failed on an OOF that disagreed with its own
    # printed score. Handing the indices down makes the smoke run checkable.
    idx = None
    if frac < 1.0 and n_train:
        import numpy as _np
        k = max(2, int(n_train * frac))
        idx = _np.sort(_np.random.default_rng(0).choice(n_train, k, replace=False))
        _np.save(workdir / "_sis_idx.npy", idx)
        env[IDX_ENV] = str(workdir / "_sis_idx.npy")

    # Refuse to start when the machine has no headroom, and size this run's
    # memory ceiling to what is actually free rather than to a fixed constant.
    ok, why, cap = system_ok(workdir)
    if not ok:
        return {"failure": "resources", "runtime_s": 0.0, "error": why}

    t0 = time.perf_counter()
    # sys.executable, never "python3": available_packages() inspects THIS
    # interpreter, so the subprocess must be the same one or we advertise a
    # package list the script cannot actually import.
    rc, stdout, stderr, killed = _run_watched(
        [sys.executable, str(workdir / "run.py")], workdir, env, timeout, cap)
    runtime = time.perf_counter() - t0
    out = stdout + "\n" + stderr

    if killed == "timeout":
        return {"failure": "timeout", "runtime_s": runtime,
                "error": f"exceeded {timeout}s. Subsample, use fewer folds, or "
                         f"move this to a Kaggle kernel."}
    if killed == "oom":
        return {"failure": "oom", "runtime_s": runtime,
                "error": f"killed after exceeding the {MEM_LIMIT_GB}GB ceiling. "
                         f"Use smaller dtypes, chunk the work, or avoid "
                         f"materialising a dense expansion of the full frame."}
    if rc != 0:
        # stderr tail, not head: the traceback's last lines name the real error
        return {"failure": _classify(stderr, rc), "error": stderr[-3000:],
                "runtime_s": runtime}

    cv = _sentinel("CV_SCORE", out)
    if cv is None:
        return {"failure": "bug", "runtime_s": runtime,
                "error": "script exited 0 but never printed 'CV_SCORE: <float>'. "
                         f"stdout tail: {stdout[-800:]}"}
    cv_std = _sentinel("CV_STD", out)
    if cv_std is None:
        return {"failure": "bug", "runtime_s": runtime,
                "error": "printed CV_SCORE but no 'CV_STD: <float>'. Report the "
                         "fold spread — a mean without it makes every comparison "
                         "against other experiments unfalsifiable."}

    res = {"cv": cv, "cv_std": cv_std, "runtime_s": runtime, "frac": frac,
           "stdout": stdout[-6000:]}
    for key, sent in (("oof", "OOF"), ("pred", "PRED"), ("submission", "SUBMISSION")):
        name = _path_sentinel(sent, out)
        res[f"{key}_path"] = str(workdir / name) if name else ""

    if not res["oof_path"] or not res["pred_path"]:
        return {"failure": "bug", "runtime_s": runtime,
                "error": "did not print both 'OOF: <file>' and 'PRED: <file>'. "
                         "Saving out-of-fold and test predictions is mandatory — "
                         "they are what error analysis and the final ensemble are "
                         "built from, and a run without them cannot be studied."}
    # Enforce the SAME contract on smoke runs as on full ones. A smoke run has
    # fewer rows, but it has a known number of them (len(idx)) and a known set
    # of labels (y[idx]) — so there is no reason to accept an OOF that
    # contradicts its own score just because the run was small.
    want_train = n_train if frac >= 1.0 else (len(idx) if idx is not None else 0)
    bad = _check_arrays(workdir, res, want_train, n_test)
    if bad:
        return {"failure": "bug", "runtime_s": runtime, "error": bad}
    if y is not None:
        y_ref = y if frac >= 1.0 else (y[idx] if idx is not None else None)
        if y_ref is not None:
            # On a small smoke sample the fold-mean and pooled scores can
            # legitimately differ by a few points; a misaligned array is off
            # by far more (a good model rescoring near chance).
            tol = 0.01 if frac >= 1.0 else max(0.05, 0.1 * abs(cv))
            bad = _oof_agrees(res["oof_path"], y_ref, metric, cv, tol)
            if bad:
                return {"failure": "bug", "runtime_s": runtime,
                        "error": bad + ("" if frac >= 1.0 else
                                        f"  (caught at SMOKE — fix this before the "
                                        f"full run. Smoke runs MUST subsample to "
                                        f"exactly the rows in ${IDX_ENV}: "
                                        f"idx = np.load(os.environ['{IDX_ENV}']). "
                                        f"The check compares your OOF against the "
                                        f"labels of THOSE rows, in that order.)")}
    return res


def demo() -> None:
    """Self-check: a good script passes, and each way of breaking the contract
    is caught with the specific message that tells the agent what to fix."""
    import tempfile

    import numpy as np

    good = """
import os, numpy as np
n_tr, n_te = 200, 80
rng = np.random.default_rng(0)
np.save(os.path.join(os.environ["SIS_OUT"], "oof.npy"), rng.random(n_tr))
np.save(os.path.join(os.environ["SIS_OUT"], "test.npy"), rng.random(n_te))
print("CV_SCORE: 0.9123"); print("CV_STD: 0.0021")
print("OOF: oof.npy"); print("PRED: test.npy")
"""
    with tempfile.TemporaryDirectory() as d:
        r = run_local(good, Path(d), n_train=200, n_test=80)
        assert r.get("cv") == 0.9123, r
        assert r.get("cv_std") == 0.0021, r
        assert np.load(r["oof_path"]).shape == (200,), r

    # No CV_STD -> rejected, not silently defaulted to zero.
    with tempfile.TemporaryDirectory() as d:
        r = run_local('print("CV_SCORE: 0.5")', Path(d))
        assert r.get("failure") == "bug" and "CV_STD" in r["error"], r

    # Score but no arrays -> rejected. This is the v1 gap the whole design turns on.
    with tempfile.TemporaryDirectory() as d:
        r = run_local('print("CV_SCORE: 0.5")\nprint("CV_STD: 0.01")', Path(d))
        assert r.get("failure") == "bug" and "OOF" in r["error"], r

    # Wrong-length OOF -> caught here rather than corrupting a later ensemble.
    wrong = good.replace("rng.random(n_tr)", "rng.random(n_tr - 5)")
    with tempfile.TemporaryDirectory() as d:
        r = run_local(wrong, Path(d), n_train=200, n_test=80)
        assert r.get("failure") == "bug" and "195 rows" in r["error"], r

    # A crash is classified, never turned into a score.
    with tempfile.TemporaryDirectory() as d:
        r = run_local('raise ValueError("boom")', Path(d))
        assert r.get("failure") == "bug" and "boom" in r["error"], r

    # A RELATIVE workdir must work. The child runs with cwd=workdir, so an
    # unresolved path made it look for workdir/workdir/run.py and reported a
    # missing file — which the debugger then tried to fix in the script.
    import os as _os
    cwd = _os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        _os.chdir(d)
        try:
            r = run_local(good, Path("rel/exp1"), n_train=200, n_test=80)
            assert r.get("cv") == 0.9123, r
        finally:
            _os.chdir(cwd)

    # Resource guards: the cap must track real headroom, not a constant, and
    # an impossible requirement must refuse rather than swap the machine.
    assert available_gb() > 0, "vm_stat unreadable"
    ok, why, cap = system_ok(Path("."))
    assert ok and 2.0 <= cap <= MEM_LIMIT_GB, (ok, why, cap)
    import sis2.run as _r
    old_min = _r.MIN_FREE_GB
    _r.MIN_FREE_GB = 10_000.0                    # demand more RAM than exists
    try:
        ok2, why2, _ = _r.system_ok(Path("."))
        assert not ok2 and "RAM available" in why2, (ok2, why2)
        with tempfile.TemporaryDirectory() as d:
            r = _r.run_local(good, Path(d), n_train=200, n_test=80)
            assert r.get("failure") == "resources", r
    finally:
        _r.MIN_FREE_GB = old_min

    # A half-filled OOF must be caught. This is the exact s6e9 failure: an
    # honest score over the folds that ran, and zeros everywhere else.
    half = """
import os, numpy as np
n_tr, n_te = 200, 80
rng = np.random.default_rng(0)
oof = rng.random(n_tr); oof[100:] = 0.0
np.save(os.path.join(os.environ["SIS_OUT"], "oof.npy"), oof)
np.save(os.path.join(os.environ["SIS_OUT"], "test.npy"), rng.random(n_te))
print("CV_SCORE: 0.9500"); print("CV_STD: 0.0021")
print("OOF: oof.npy"); print("PRED: test.npy")
"""
    yv = np.concatenate([np.tile([0.0, 1.0], 50), np.tile([0.0, 1.0], 50)])
    with tempfile.TemporaryDirectory() as d:
        r = run_local(half, Path(d), n_train=200, n_test=80, y=yv, metric="auc")
        assert r.get("failure") == "bug", r
        assert "saved OOF scores" in r["error"] and "exactly 0" in r["error"], r["error"]

    # An honest OOF that DOES reproduce its score must still pass.
    with tempfile.TemporaryDirectory() as d:
        r = run_local(good, Path(d), n_train=200, n_test=80)
        assert r.get("cv") == 0.9123, r

    print("run.py: 11 checks passed")


if __name__ == "__main__":
    demo()
