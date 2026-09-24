"""The loop: study the champion, change one thing, judge it, write it down.

This is the part that differs in kind from v1, so it is worth stating the shape
plainly before the code:

    while budget:
        champion  = best scored experiment in this family
        brief     = analyst.report(champion)        # WHERE it is losing
        proposal  = rival.propose(brief, notes, history)
        result    = run(proposal)                   # smoke at 5%, then full
        verdict   = paired fold-by-fold test against the parent
        if better: write a discovery note, consider submitting

v1's loop was `select a node by UCT, mutate it, score it`. That is a search over
configurations, and it converges on whatever the configuration space has a lot
of — which for a playground competition is the same blend everybody else builds.
The loop above is a search over *diagnosed weaknesses*, which is a much smaller
and much better-aimed space. The analyst brief is the input v1 never had.

Two mechanisms keep it honest:

**One change per experiment.** The proposal names a single change against a
named parent. Without that, a run that scores better tells you nothing about
which of its five differences mattered, and the tree stops meaning anything.

**Smoke before full.** Every script runs at 5% of the data first. Most generated
code fails for boring reasons, and paying the full training cost to discover a
typo is the single largest waste of wall clock in a system like this.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from . import analyst, kaggle, metrics, run, store
from . import llm as llm_mod

MAX_DEBUG_ROUNDS = 3       # a script that resists three fixes is a bad idea, not a typo
SMOKE_FRAC = 0.05
MIN_EFFECT = float(os.getenv("SIS_MIN_EFFECT", "0.05"))  # x fold spread; see _judge


SYSTEM = """\
You are a Kaggle competitor working an ongoing campaign. You are not starting \
from scratch: there is a ledger of what has already been tried, a set of \
discovery notes from previous campaigns, and an error analysis of the current \
best model. Read them before you propose anything.

Your job each turn is to propose exactly ONE change to ONE model, aimed at a \
weakness the error analysis actually shows. Not a plausible-sounding \
improvement — a targeted response to a measured failure.

What makes a good proposal here:
- It names the specific finding it addresses ("slice X carries 1.4x loss", \
"rows missing Y are harder"), and the mechanism by which the change addresses it.
- It changes ONE thing against the named parent, so the score difference is \
attributable. Bundling five ideas produces a number nobody can learn from.
- It is worth the compute. A hyperparameter nudge inside the noise floor is not.

You are NOT restricted to any set of model families, libraries, or techniques. \
Anything you can write in one Python file with the available packages is fair \
game: gradient boosting, neural networks of any architecture (including \
attention/transformers over field tokens), generalised linear models with \
splines, factorisation machines, nearest-neighbour target statistics, \
calibration layers, custom objectives, distillation, pseudo-labelling, \
constrained optimisation over out-of-fold predictions — whatever the evidence \
argues for. Optimisation method is equally open: hand-set parameters, a search, \
a custom loss, a different objective entirely.

If the incremental path has plateaued, say so and propose a DIFFERENT \
FORMULATION of the problem rather than another variation of the current model. \
A new formulation that scores slightly worse is more valuable than a tenth \
variant that scores the same, because it tells us something the ledger does not \
already contain.

What loses competitions here:
- Rebuilding the same broad ensemble everyone else has. The public notebook \
pool converges to a ceiling; matching it is not a result.
- Staying inside the approaches already in the ledger because they are familiar. \
Check what has been tried: if every experiment is the same family of idea, that \
is a reason to leave it, not to continue.

Build ONE model. Do not propose an ensemble, blend, or stack of previous \
experiments — every experiment already saves its out-of-fold predictions, and \
combining them is done separately whenever it is wanted. A blend teaches nothing \
about the problem; your job is to produce a single model that is either better \
or differently wrong than what exists.
- Chasing gains that do not survive the paired test. Every experiment is \
compared to its parent fold by fold on the shared folds (the `vs parent` \
column: mean delta and paired t). A gain that is not consistent across folds \
is noise and will not reproduce on the leaderboard.
- Ignoring the discovery notes and re-deriving something a previous campaign \
already settled.

Reply in EXACTLY this format and nothing else:

FAMILY: <a short label for the approach. ANY approach is allowed — this is a
free-text tag, not a menu. Existing labels are xgb / lgbm / catboost / nn,
but a genuinely different formulation should get its OWN new label
(transformer, glm-splines, knn-te, factorization, gp, rule-ensemble, ...)>
PARENT: <experiment id to extend, or "none" for a fresh baseline>
HYPOTHESIS: <one or two sentences: the finding you are addressing and why this \
change should move it>
CHANGE: <a short label, under 70 characters>
```python
<the complete script>
```
"""


def _parse(text: str) -> dict | None:
    """Pull the four fields and the script out of the reply.

    Returns None rather than raising on a malformed reply: a bad parse should
    cost one turn, not kill an overnight run.
    """
    code = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not code:
        return None
    def field(name, default=""):
        m = re.search(rf"^{name}\s*:\s*(.+)$", text, re.M | re.I)
        return m.group(1).strip() if m else default
    parent = field("PARENT", "none").lower()
    body = code.group(1).strip()
    if len(body) < 200:            # a stub, not a script
        return None
    return {
        "family": (field("FAMILY", "other").split()[0] if field("FAMILY") else "other")[:24],
        "parent_id": int(re.sub(r"\D", "", parent)) if re.search(r"\d", parent) else None,
        "hypothesis": field("HYPOTHESIS")[:1000],
        "change": (field("CHANGE") or "unnamed change")[:70],
        "code": body,
    }


THINKER = """\
You are a Kaggle competitor deciding what to try next. You do NOT write code \
this turn — a separate implementer will write it from your decision, so your \
decision has to carry everything they need.

Choose ONE change to ONE model, aimed at a weakness the error analysis actually \
shows. Not a plausible-sounding improvement — a targeted response to a measured \
failure. Gains that are not consistent fold by fold (the paired t in the \
ledger) are worthless; they will not reproduce on the leaderboard.

Reply in EXACTLY this format and nothing else:

FAMILY: <a short label for the approach. ANY approach is allowed — this is a
free-text tag, not a menu. Existing labels are xgb / lgbm / catboost / nn,
but a genuinely different formulation should get its OWN new label
(transformer, glm-splines, knn-te, factorization, gp, rule-ensemble, ...)>
PARENT: <experiment id to extend, or "none" for a fresh baseline>
HYPOTHESIS: <the finding you are addressing and why this change should move it>
CHANGE: <a short label, under 70 characters>
SPEC:
<Precise implementation instructions for the implementer. Name exact columns, \
exact transforms, exact model and parameters. They cannot see the error \
analysis or the ledger — only what you write here. Be specific enough that two \
implementers would produce the same script.>
"""

EXECUTOR = """\
You are implementing a decision someone else made. Do not second-guess it, do \
not add extra ideas, do not "improve" the plan — the experiment is only \
interpretable if the script tests exactly the one change described.

Reply with ONLY a ```python block containing the complete script. No prose.
"""


def _parse_spec(text: str) -> dict | None:
    """Parse the thinker's decision. Same fields as a full proposal, minus code."""
    def field(name, default=""):
        m = re.search(rf"^{name}\s*:\s*(.+)$", text, re.M | re.I)
        return m.group(1).strip() if m else default
    spec = re.search(r"^SPEC\s*:\s*\n(.+)", text, re.M | re.I | re.S)
    if not spec or len(spec.group(1).strip()) < 80:
        return None                      # no usable instructions for the implementer
    parent = field("PARENT", "none").lower()
    return {
        "family": (field("FAMILY", "other").split()[0] if field("FAMILY") else "other")[:24],
        "parent_id": int(re.sub(r"\D", "", parent)) if re.search(r"\d", parent) else None,
        "hypothesis": field("HYPOTHESIS")[:1000],
        "change": (field("CHANGE") or "unnamed change")[:70],
        "spec": spec.group(1).strip(),
    }


class Campaign:
    """One competition, one working directory, one ledger."""

    def __init__(self, home: Path, llm, *, slug: str, notes_dir: Path):
        self.home = Path(home)
        self.slug = slug
        self.llm = llm
        self.notes_dir = Path(notes_dir)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.conn = store.open_db(self.home)
        self.data = self.home / "input"
        self.metric = store.meta(self.conn, "metric", "auc")
        self.target = store.meta(self.conn, "target", "")
        # Recipes (and any script that wants them) read these rather than
        # hardcoding a column name.
        os.environ["SIS_TARGET"], os.environ["SIS_METRIC"] = self.target, self.metric

    # ------------------------------------------------------------- context

    def _brief(self, family: str) -> str:
        """The error analysis of this family's champion, plus the rival's.

        Loading the training frame each turn is a few seconds and worth it —
        stale analysis of a superseded champion is how an agent spends a day
        fixing a weakness it already fixed.
        """
        # Global champion, not a per-label one. Scoping by family meant the
        # analyst studied "the best lgbm" rather than "the best model", which
        # made the diagnosis depend on a word the rival had chosen.
        champ = store.champion(self.conn)
        if champ is None or not champ["oof_path"]:
            return ("No scored model in this family yet. Propose a solid, honest "
                    "baseline: correct use of the shared folds, no leakage, and "
                    "saved OOF/PRED arrays so the next turn can study it.")
        try:
            import numpy as np
            import pandas as pd
            tr = pd.read_csv(self.data / "train.csv")
            # `tr[self.target].values` here silently broke the ENTIRE premise of
            # this system. s6e9's target is "Yes"/"No", so every analyst call
            # raised "could not convert string to float" and the except-branch
            # below turned it into a one-line "error analysis unavailable" that
            # looked like a minor degradation. It is not minor: the error
            # analysis IS the loop. Rivals were proposing from the data profile
            # and history alone — i.e. doing exactly what v1 did.
            y = self._y()
            if y is None:
                raise ValueError(f"cannot encode target `{self.target}`")
            X = tr.drop(columns=[c for c in (self.target, "id") if c in tr.columns])
            oof = np.load(champ["oof_path"])
            # Compare against the most DECORRELATED scored model, found by
            # measuring error correlation rather than by picking a different
            # label. Disagreement is what an ensemble can use; a category is not.
            rival, worst = None, 2.0
            champ_loss = analyst.row_loss(y, oof, self.metric)
            for r in self.conn.execute(
                    "SELECT id, change, oof_path FROM experiments WHERE status='done' "
                    "AND oof_path != '' AND family != 'blend' AND id != ?",
                    (champ["id"],)):
                try:
                    ro = np.load(r["oof_path"])
                except Exception:
                    continue
                if len(ro) != len(oof):
                    continue
                ec = float(np.corrcoef(champ_loss,
                                       analyst.row_loss(y, ro, self.metric))[0, 1])
                if ec < worst:
                    worst, rival = ec, (f"most decorrelated model, exp #{r['id']} "
                                        f"({r['change'][:40]})", ro)
            head = (f"The current champion is experiment #{champ['id']} "
                    f"({champ['change']}), CV {champ['cv']:.5f} "
                    f"+/- {champ['cv_std']:.5f}.\n\n")
            return head + analyst.report(X, y, oof, metric=self.metric, rival=rival)
        except Exception as e:
            return (f"(champion is exp #{champ['id']}, CV {champ['cv']:.5f}; "
                    f"error analysis unavailable: {e})")

    def _history(self, family: str) -> str:
        rows = store.history(self.conn, family, limit=30)
        if not rows:
            return "(nothing tried yet)"
        out = ["  id | parent | verdict       |      CV |  +/-    | vs parent (paired) | change"]
        for r in rows:
            if r["failure"]:
                out.append(f"{r['id']:>4} | {str(r['parent_id'] or '-'):>6} | "
                           f"FAILED ({r['failure']})".ljust(40) + f"| {r['change'][:50]}")
            else:
                vs = (f"{r['delta']:+.5f} t={r['delta_t']:+.1f}"
                      if r["delta"] is not None else "")
                out.append(f"{r['id']:>4} | {str(r['parent_id'] or '-'):>6} | "
                           f"{(r['verdict'] or '?'):<13} | {r['cv']:.5f} | "
                           f"{(r['cv_std'] or 0):.5f} | {vs:<18} | {r['change'][:50]}")
        return "\n".join(out)

    ADVISOR_ASK = (
        "You are advising a Kaggle team. Do NOT write code. List the 3 most "
        "promising modelling directions for THIS dataset specifically, ranked. "
        "For each: one line on the mechanism, one line on why it suits this data "
        "in particular. Be concrete about actual column names — generic advice "
        "('try feature engineering', 'tune hyperparameters') is worthless.")

    def _advice(self, family: str, turn: int = 0) -> str:
        """Ask an outside model for directions. Empty string on any failure.

        A different architecture brings different priors, which is the entire
        value — and the reason the text is framed below as something to weigh
        rather than obey. An advisor that could dictate the turn would just be
        a worse rival; one that can be disagreed with is a source of ideas.

        **Periodic, not per-turn.** Measured on s6e9: an advisor call takes
        ~150s while the full 10-fold experiment it advises takes ~66s. Asking
        every turn would spend most of a campaign waiting for opinions instead
        of running experiments. So it fires only where the advice can still
        change something — opening a family that has no champion yet, or every
        Nth turn after that to break a plateau.
        """
        every = int(os.getenv("SIS_ADVISOR_EVERY", "5"))
        if store.champion(self.conn, family) is not None:
            if every <= 0 or turn % every != 0:
                return ""
        try:
            return self.llm.advise(self.ADVISOR_ASK,
                                   f"Family under work: {family}\n\n"
                                   f"{self._brief(family)}")
        except Exception:
            return ""

    def _prompt(self, family: str, hint: str = "", advice: str = "") -> str:
        # A hint appears in exactly one prompt and is then consumed. It is
        # emphasised because it is a guaranteed slot, not because it outranks
        # the evidence — the resulting experiment is judged on CV like any other.
        hint_txt = (f"\n# A hint from the human running this campaign\n"
                    f"> {hint}\n\nAddress this in the proposal you make now. "
                    f"You are seeing it once; it will not be repeated.\n"
                    if hint else "")
        # Framed as a second opinion on purpose. It comes from a model that
        # cannot see the ledger and has no stake in the result; treating it as
        # instruction would let one bad idea derail a turn.
        advice_txt = (f"\n# A second opinion from an outside model\n"
                      f"It has not seen the experiment history and may be wrong. "
                      f"Weigh it; disagree freely if the error analysis says "
                      f"otherwise.\n\n{advice}\n" if advice else "")
        flight = store.in_flight(self.conn)
        flight_txt = "\n".join(
            f"  #{r['id']} [{r['family']}] {r['change']} (by {r['author'] or '?'})"
            for r in flight) or "  (nothing running)"
        lb = store.best_lb(self.conn)
        board = store.meta(self.conn, "board", "")
        return f"""\
# Competition
{self.slug} — metric **{self.metric}**, \
{'higher' if metrics.direction(self.metric) else 'lower'} is better.
{board}

Our best public LB score so far: {f'{lb:.5f}' if lb is not None else 'nothing submitted yet'}
{store.cv_lb_trust(self.conn)}

# Data
{kaggle.profile(self.data, self.target)}

# The contract your script must satisfy
{run.CONTRACT}

{run.available_packages()}

# How to approach this kind of competition (standing playbook)
{store.read_skills(Path(__file__).resolve().parent.parent / "skills")}

# Discovery notes from this and previous campaigns
{store.read_notes(self.notes_dir)}

# Transferable lessons from OTHER competitions (weigh them; data differs)
{store.read_notes(self.notes_dir.parent / "_global", limit=30)}

# Everything tried, most recent first
{self._history("")}

# Where prediction-space is already crowded (measured, not labelled)
{self._novelty()}

# The portfolio — what we would submit right now
{self._portfolio()}

# Running right now — do NOT duplicate these
{flight_txt}

# Error analysis
{self._brief(family)}
{advice_txt}{hint_txt}
Now propose one change, in the required format."""

    # ------------------------------------------------------------ execution

    def _execute(self, exp_id: int, code: str) -> dict:
        """Smoke at 5%, repair on failure, then run for real.

        The debugger seat is a cheap model and it handles most of what goes
        wrong, which is why it is a separate seat: it runs on the majority of
        turns, and a large model there is how an agent's cost quietly triples.
        """
        wd = self.home / f"exp{exp_id}"
        n_tr, n_te = self._row_counts()
        for attempt in range(MAX_DEBUG_ROUNDS + 1):
            # Smoke exists to catch typos in seconds, not to judge whether an
            # architecture is fast enough. A 2-layer transformer over 33k rows x
            # 10 folds legitimately needs longer than a GBDT, and a flat 900s
            # budget killed the only transformer this campaign has produced.
            heavy = bool(re.search(r"^\s*(import|from)\s+torch", code, re.M))
            res = run.run_local(code, wd,
                                frac=SMOKE_FRAC / 5 if heavy else SMOKE_FRAC,
                                data_dir=self.data, timeout=2400 if heavy else 900)
            # A machine under pressure is not a broken script. Sending this to
            # the debugger would burn three repair rounds "fixing" correct code
            # and then record a bug that never existed — the same way the
            # doubled-path error did. Wait for the machine instead.
            if res.get("failure") == "resources":
                if not self._wait_for_resources(wd):
                    return res           # machine never recovered; keep the code
                continue                 # room now — retry the same script
            if not res.get("failure"):
                break
            if attempt == MAX_DEBUG_ROUNDS:
                res["error"] = (f"still failing after {MAX_DEBUG_ROUNDS} repair "
                                f"attempts: {res.get('error', '')}")
                return res
            fixed = self._repair(code, res)
            if not fixed:
                return res
            code = fixed
            self.conn.execute("UPDATE experiments SET code=? WHERE id=?",
                              (code, exp_id))
            self.conn.commit()

        # Smoke passed. The full run goes to a Kaggle GPU kernel when the work
        # wants one — always smoke LOCALLY first, because a kernel costs a
        # minute of startup plus quota and finding a typo there is the most
        # expensive way to find a typo.
        if self._needs_gpu(code):
            ds = self._folds_dataset()
            if ds:
                res = kaggle.run_kernel(
                    code, wd / "kernel", f"sis2-{self.slug}-exp{exp_id}",
                    kaggle.comp_slug(self.slug), gpu=True, dataset_sources=[ds],
                    timeout=int(os.getenv("SIS_KERNEL_TIMEOUT", "5400")))
                if not res.get("failure"):
                    return res
                # A kernel failure is not a reason to lose the experiment —
                # fall through and run it here, slowly but reliably.
                print(f"  [kernel] {str(res.get('error'))[:120]} — running locally",
                      flush=True)
            else:
                print("  [kernel] folds dataset unavailable — running locally",
                      flush=True)

        res = run.run_local(code, wd, frac=1.0, data_dir=self.data,
                            n_train=n_tr, n_test=n_te,
                            y=self._y(), metric=self.metric)

        # An OOM is a statement about THIS LAPTOP, not about the idea. Three
        # novel formulations (spline GLMs, a tensor-product GLM, a transformer)
        # have died against a ~6GB ceiling that a Kaggle kernel does not have —
        # kernels get roughly 30GB. Losing a good hypothesis to the wrong
        # hardware is the most wasteful failure available, so retry it there.
        if res.get("failure") == "oom":
            ds = self._folds_dataset()
            if ds:
                print(f"  [oom] {res.get('error','')[:70]} — retrying on a Kaggle "
                      f"kernel (~30GB)", flush=True)
                k = kaggle.run_kernel(
                    code, wd / "kernel", f"sis2-{self.slug}-exp{exp_id}",
                    kaggle.comp_slug(self.slug), gpu=False, dataset_sources=[ds],
                    timeout=int(os.getenv("SIS_KERNEL_TIMEOUT", "5400")))
                if not k.get("failure") and k.get("oof_path"):
                    bad = run._oof_agrees(k["oof_path"], self._y(), self.metric,
                                          k["cv"])
                    if bad:
                        return {"failure": "bug", "error": bad}
                return k
        return res

    def _needs_gpu(self, code: str) -> bool:
        """Should this run on a Kaggle GPU rather than here?

        Triggered by the script importing torch — that is the honest signal,
        far better than guessing from the family label, because a `nn` family
        experiment implemented with sklearn's MLPClassifier gains nothing from
        a GPU while an `xgb` one using `device="cuda"` does.
        """
        if os.getenv("SIS_GPU", "auto") == "off":
            return False
        # A local competition has no Kaggle to push to — its data exists only
        # here, so a kernel would run against nothing.
        from . import local
        if local.is_local(self.home):
            return False
        # Detect the opt-in the contract describes. Matching the ACTUAL GPU
        # switches rather than the family label matters: an `nn` experiment
        # written with sklearn's MLPClassifier gains nothing from a T4, while
        # an `xgb` one with device="cuda" gains a lot. exp11 was exactly that
        # case — family `nn`, no torch, correctly kept local.
        patterns = (
            r"^\s*(import|from)\s+torch",
            r"device\s*=\s*[\"']cuda[\"']",
            r"task_type\s*=\s*[\"']GPU[\"']",
            r"device\s*=\s*[\"']gpu[\"']",
            r"tree_method\s*=\s*[\"']gpu_hist[\"']",
        )
        return any(re.search(p, code, re.M) for p in patterns)

    def _folds_dataset(self) -> str:
        """The shared folds, published so a kernel can see them. Cached."""
        if getattr(self, "_folds_ds", None) is None:
            try:
                self._folds_ds = kaggle.ensure_folds_dataset(self.data, self.home)
            except Exception:
                self._folds_ds = ""
        return self._folds_ds

    def _wait_for_resources(self, wd: Path, tries: int = 10,
                            wait: int = 60) -> bool:
        """Block until the machine has room again. True if it recovered.

        Memory pressure on a laptop is usually transient — another process
        exits, a cache drains. Waiting ten minutes beats abandoning an
        experiment, and beats pushing the machine into swap where every run
        gets slower and the laptop becomes unusable to its owner.
        """
        for i in range(tries):
            ok, why, _ = run.system_ok(wd)
            if ok:
                return True
            print(f"  [resources] {why[:90]} — waiting {wait}s ({i + 1}/{tries})",
                  flush=True)
            time.sleep(wait)
        return run.system_ok(wd)[0]

    def _repair(self, code: str, res: dict) -> str:
        prompt = (f"This script failed with `{res['failure']}`. Fix it and return "
                  f"ONLY the corrected script in one ```python block — no "
                  f"explanation.\n\nDo not change what the experiment is testing; "
                  f"fix only what broke.\n\nError:\n{res.get('error', '')[:3000]}"
                  f"\n\nScript:\n```python\n{code}\n```")
        try:
            out = self.llm.complete("debugger", "You fix Python scripts.", prompt)
        except Exception:
            return ""
        m = re.search(r"```(?:python)?\s*\n(.*?)```", out, re.S)
        return m.group(1).strip() if m else ""

    def _y(self):
        """Labels as floats, cached. Non-numeric targets (s6e9 ships Yes/No)
        are encoded with the MINORITY class as 1 — the same convention the data
        profile tells the agent to use, so the verification and the script
        agree on which way round the positive class is."""
        if getattr(self, "_y_cache", None) is None:
            try:
                import pandas as pd
                col = pd.read_csv(self.data / "train.csv", usecols=[self.target])[self.target]
                self._y_cache = metrics.encode_target(col)
            except Exception:
                self._y_cache = False        # unavailable; skip verification
        # `self._y_cache or None` looks equivalent and is not: `or` calls
        # bool() on the array, which raises "truth value of an array with more
        # than one element is ambiguous" and killed a 40-turn run on turn 1.
        # Compare against the sentinel explicitly.
        return None if self._y_cache is False else self._y_cache

    def _judge(self, parent_oof: str, oof: str):
        """Paired fold-by-fold verdict: (verdict, mean delta, t) or None.

        None means "cannot judge this way" (arrays missing or misaligned),
        and the store falls back to the old spread rule rather than guessing.
        """
        import numpy as np
        y = self._y()
        if y is None:
            return None
        folds = np.load(self.data / "folds.npy")
        a, b = np.load(oof), np.load(parent_oof)
        if not (len(a) == len(b) == len(y) == len(folds)):
            return None
        m, se, t = metrics.paired(y, a, b, folds, self.metric)
        tc = metrics.t_crit(len(np.unique(folds)))
        # Significance is not enough on its own: two near-identical blends
        # differ by ~1e-7 on EVERY fold, which is a huge t and a worthless
        # gain (it spent a real submission slot on the bank proxy). Require
        # the effect to be a visible fraction of the fold spread as well.
        floor = MIN_EFFECT * float(np.std(metrics.fold_scores(y, a, folds, self.metric)))
        v = ("better" if t > tc and m > floor else
             "worse" if t < -tc and -m > floor else "inconclusive")
        return v, m, t

    def _row_counts(self) -> tuple[int, int]:
        try:
            import pandas as pd
            n_tr = len(pd.read_csv(self.data / "train.csv", usecols=[0]))
            n_te = len(pd.read_csv(self.data / "test.csv", usecols=[0]))
            return n_tr, n_te
        except Exception:
            return 0, 0

    # --------------------------------------------------------------- notes

    def _write_note(self, exp: dict, verdict: str, brief: str) -> None:
        """Record a finding as markdown, for every future campaign to read.

        The writeup calls these files "gold" and it is the cheapest durable
        thing in the system: a paragraph written now stops a model three weeks
        from now re-running an experiment that already answered the question.
        """
        prompt = (
            f"Write a short discovery note (under 200 words) about this result, "
            f"for a future agent who has never seen this competition.\n\n"
            f"State what was changed, what happened to CV, and — most importantly "
            f"— WHY you think it worked or did not, tied to the error analysis. "
            f"A note that only restates the score is worthless; the transferable "
            f"part is the mechanism.\n\n"
            f"Change: {exp['change']}\nHypothesis: {exp['hypothesis']}\n"
            f"Verdict: {verdict} (CV {exp['cv']:.5f} +/- {exp['cv_std']:.5f})\n\n"
            f"Error analysis that motivated it:\n{brief[:4000]}\n\n"
            f"Reply with markdown only, starting with a `# ` title line.")
        try:
            body = self.llm.complete("reporter", "You write concise technical notes.",
                                     prompt, experiment_id=exp["id"])
        except Exception:
            return
        slug = re.sub(r"[^a-z0-9]+", "-", exp["change"].lower()).strip("-")[:40]
        path = self.notes_dir / f"{exp['id']:04d}-{slug or 'note'}.md"
        path.write_text(body.strip() + "\n", encoding="utf-8")
        title = body.strip().splitlines()[0].lstrip("# ").strip() if body.strip() else exp["change"]
        store.add_note(self.conn, path, title[:120], exp["family"], exp["id"])

    # ---------------------------------------------------------------- turns

    def propose(self, seat: str, family: str, hint: str, advice: str,
                turn: int = 0) -> dict | None:
        """Get one proposal. Either one call, or a thinker/executor pair.

        The split is worth it ONLY on the API backend. Through `claude -p`,
        96% of a prompt is Claude Code's own ~36K-token harness prefix, so two
        calls pay it twice and the split costs ~32% MORE than a single call.
        On the API there is no prefix, so moving the ~3.4K output tokens of
        boilerplate script-writing from Opus ($25/Mtok) to Sonnet ($15/Mtok)
        is a real saving on the expensive half of the call.

        Third-party rivals are never split: they are free, so there is nothing
        to save, and a handoff can only lose information.
        """
        # Every Nth turn, or whenever six experiments have passed with no gain,
        # ask for a different FORMULATION instead of another variant.
        every = int(os.getenv("SIS_REFORMULATE_EVERY", "7"))
        extra = (self.REFORMULATE
                 if (self._plateaued() or (every > 0 and turn and turn % every == 0))
                 else "")
        split = (os.getenv("SIS_SPLIT", "0") == "1"
                 and "/" not in llm_mod.SEATS.get(seat, ("", ""))[0])
        if not split:
            reply = self.llm.complete(seat, SYSTEM + extra,
                                      self._prompt(family, hint, advice))
            return _parse(reply)

        decision = self.llm.complete(seat, THINKER + extra,
                                     self._prompt(family, hint, advice))
        spec = _parse_spec(decision)
        if spec is None:
            return None
        # The implementer sees the contract, the data, and the decision — and
        # deliberately NOT the ledger or the error analysis. Its job is
        # transcription, not judgement; giving it the reasoning would invite it
        # to substitute its own.
        code_reply = self.llm.complete("executor", EXECUTOR, f"""\
# The decision to implement
Family: {spec['family']}
Change: {spec['change']}
Why: {spec['hypothesis']}

## Specification
{spec['spec']}

# Contract
{run.CONTRACT}

{run.available_packages()}

# Data
{kaggle.profile(self.data, self.target)}

Write the complete script now.""")
        code = re.search(r"```(?:python)?\s*\n(.*?)```", code_reply, re.S)
        if not code or len(code.group(1).strip()) < 200:
            return None
        return {**spec, "code": code.group(1).strip()}

    REFORMULATE = """
# The incremental path looks exhausted

Look at the ledger above. If most recent experiments are variations on one idea \
and the verdicts are mostly `inconclusive` or `worse`, then more variations will \
not help — that is what a plateau looks like from the inside.

This turn, propose a DIFFERENT FORMULATION rather than another variant. Not a \
tweak to the leading model: a different way of posing the problem. Attention over \
field tokens, a generalised linear model with splines on the numerics, \
nearest-neighbour target statistics, a factorisation of the categorical \
interactions, a two-stage model, a custom objective, distillation from the \
ensemble — or something not on this list. Give it a NEW family label.

A different formulation that scores somewhat worse is a better outcome than a \
variant that ties, because it adds information the ledger does not have.
"""

    def _all_families(self) -> str:
        rows = self.conn.execute(
            "SELECT family, COUNT(*) n, SUM(verdict='better') wins, "
            "MAX(cv) best FROM experiments WHERE status='done' "
            "GROUP BY family ORDER BY n DESC").fetchall()
        if not rows:
            return "(nothing yet)"
        out = ["  approach   | tried | wins | best CV"]
        for r in rows:
            out.append(f"  {r['family'][:10]:<10} | {r['n']:>5} | {r['wins'] or 0:>4} | "
                       f"{r['best']:.6f}" if r["best"] else
                       f"  {r['family'][:10]:<10} | {r['n']:>5} | {r['wins'] or 0:>4} |    -")
        return "\n".join(out)

    def _plateaued(self) -> bool:
        """No `better` verdict in the last 6 completed experiments."""
        rows = self.conn.execute(
            "SELECT verdict FROM experiments WHERE status='done' "
            "ORDER BY id DESC LIMIT 6").fetchall()
        return len(rows) >= 6 and not any(r["verdict"] == "better" for r in rows)

    def _novelty(self) -> str:
        """How crowded prediction-space already is, measured not labelled.

        `family` was a proxy for "is this experiment different from that one",
        and a bad one: two lgbm runs can differ more than an lgbm and an xgb,
        and asking a model to declare a category makes it think in categories.
        We already store every experiment's out-of-fold predictions, so
        difference is measurable — this reports the actual error-correlation
        structure and lets the rival aim at a genuinely empty region instead of
        an unused word.
        """
        import numpy as np
        rows = self.conn.execute(
            "SELECT id, change, cv, oof_path FROM experiments WHERE status='done' "
            "AND oof_path != '' AND family != 'blend' ORDER BY cv "
            f"{'DESC' if store.higher_is_better(self.conn) else 'ASC'} LIMIT 12").fetchall()
        y = self._y()
        if y is None or len(rows) < 2:
            return "(too few scored models to measure novelty yet)"
        keep = []
        for r in rows:
            try:
                o = np.load(r["oof_path"])
            except Exception:
                continue
            if len(o) == len(y) and np.isfinite(o).all():
                keep.append((r["id"], r["cv"], r["change"], analyst.row_loss(y, o, self.metric)))
        if len(keep) < 2:
            return "(too few loadable OOF arrays yet)"
        best_id, best_cv, best_change, best_loss = keep[0]
        out = [f"The leading model is exp #{best_id} (CV {best_cv:.6f}): {best_change[:60]}",
               "",
               "How much every other scored model DUPLICATES it (error correlation):",
               "  exp | err-corr |      CV | change"]
        mind, minid = 1.0, None
        for eid, cv, change, loss in keep[1:]:
            ec = float(np.corrcoef(best_loss, loss)[0, 1])
            if ec < mind:
                mind, minid = ec, eid
            out.append(f"  {eid:>3} | {ec:>8.3f} | {cv:.6f} | {change[:44]}")
        out.append("")
        if mind > 0.95:
            out.append(f"EVERY model here duplicates the leader (lowest correlation "
                       f"{mind:.3f}, exp #{minid}). Prediction-space is crowded in one "
                       f"spot: nothing tried so far makes genuinely different mistakes. "
                       f"A proposal that lands below the leader on CV but decorrelates "
                       f"is worth MORE than one that ties it, because an ensemble can "
                       f"only gain from disagreement.")
        else:
            out.append(f"exp #{minid} is the most decorrelated at {mind:.3f} — that "
                       f"direction has room in it.")
        return "\n".join(out)

    def _portfolio(self) -> str:
        """The current best blend and who is in it.

        The board is won by the ensemble, not by the best single model, and a
        rival judged only on single-model CV has no reason to build the
        decorrelated model the ensemble actually needs. Showing the blend and
        its weights makes "enters the portfolio" a visible way to win a turn.
        """
        b = store.champion(self.conn, "blend")
        single = store.champion(self.conn)
        if b is None or single is None:
            return ("(no blend beats the best single model yet — a model that is "
                    "differently wrong is what would create one)")
        return (f"blend exp #{b['id']}: nested-CV {b['cv']:.6f} +/- {b['cv_std']:.6f} "
                f"vs best single #{single['id']} {single['cv']:.6f}\n"
                f"members (weight): {b['hypothesis'].split('weights', 1)[-1].strip()}\n"
                f"A model that earns weight here is a win even if it loses to the "
                f"champion on its own.")

    def pick_family(self) -> str:
        """Which family to work on this turn.

        Round-robin over families that have a champion, with a bias toward the
        one with fewest attempts. Explicitly NOT a bandit on score: the leading
        family would monopolise every turn, and the whole value of the second
        family is that its errors are decorrelated from the first's.
        """
        # Whatever has actually been tried, plus any configured seeds. Derived
        # from the ledger rather than a fixed list, so a rival that invents a
        # new approach gets it rotated like any other — the old hardcoded
        # "xgb,lgbm,catboost,nn" meant 33 experiments never left four families.
        counts = {f.strip(): 0 for f in
                  store.meta(self.conn, "families", "xgb,lgbm,catboost").split(",")
                  if f.strip()}
        for r in self.conn.execute(
                "SELECT family, COUNT(*) n FROM experiments GROUP BY family"):
            counts[r["family"]] = r["n"]
        return min(counts, key=lambda f: counts[f])

    def propose_and_queue(self, seat: str, turn_no: int = 0) -> tuple[int | None, str]:
        """Ask one rival for a proposal and queue it. (exp_id, message).

        Split from execution so the next proposal can be written WHILE the
        current experiment trains. A rival call takes 30-150s and a full run
        on a mid-sized table about as long; doing them in series leaves the
        CPU idle for half of every turn.
        """
        family = self.pick_family()
        hint = store.pending_hint(self.conn)
        advice = self._advice(family, turn_no)
        try:
            prop = self.propose(seat, family, hint["text"] if hint else "", advice,
                                turn_no)
        except Exception as e:
            return None, f"[{seat}] could not get a proposal: {e}"
        if prop is None:
            return None, f"[{seat}] proposal did not parse; skipping this turn"
        # A hallucinated parent would compute the verdict against the wrong
        # baseline and corrupt every later attribution — fall back to the
        # family champion instead of trusting the id.
        pid = prop["parent_id"]
        if pid is not None and not self.conn.execute(
                "SELECT 1 FROM experiments WHERE id=?", (pid,)).fetchone():
            champ = store.champion(self.conn, prop["family"])
            pid = champ["id"] if champ is not None else None
        exp_id = store.queue(self.conn, family=prop["family"],
                             hypothesis=prop["hypothesis"], change=prop["change"],
                             code=prop["code"], parent_id=pid, author=seat,
                             source="user" if hint else "agent")
        if hint:
            store.consume_hint(self.conn, hint["id"], exp_id)
        return exp_id, ""

    def run_one(self, exp_id: int, seat: str = "") -> str:
        """Execute a queued experiment, record its verdict, maybe write a note."""
        row = self.conn.execute("SELECT * FROM experiments WHERE id=?",
                                (exp_id,)).fetchone()
        family, code = row["family"], row["code"]
        seat = seat or row["author"] or "?"
        store.mark_running(self.conn, exp_id)
        res = self._execute(exp_id, code)
        verdict = store.finish(self.conn, exp_id, res, judge=self._judge)
        if res.get("failure"):
            return (f"[{seat}] exp #{exp_id} [{family}] FAILED ({res['failure']}): "
                    f"{res.get('error', '')[:160]}")
        exp = dict(self.conn.execute("SELECT * FROM experiments WHERE id=?",
                                     (exp_id,)).fetchone())
        vs = (f" ({exp['delta']:+.5f}, t {exp['delta_t']:+.1f})"
              if exp.get("delta") is not None else "")
        line = (f"[{seat}] exp #{exp_id} [{family}] {verdict}{vs} — CV {exp['cv']:.5f} "
                f"+/- {exp['cv_std']:.5f} — {exp['change']}")
        if verdict in ("better", "baseline") and exp.get("source") != "recipe":
            self._write_note(exp, verdict, self._brief(family))
        return line

    def turn(self, seat: str = "rival_a", turn_no: int = 0) -> str:
        """One full cycle, in series. Returns a one-line summary for the log."""
        # A queued row (a user hint, or a retry) always goes first.
        pending = store.next_queued(self.conn)
        if pending is not None:
            return self.run_one(pending["id"], seat)
        exp_id, msg = self.propose_and_queue(seat, turn_no)
        if exp_id is None:
            return msg
        return self.run_one(exp_id, seat)

    def clone(self) -> "Campaign":
        """Same campaign, own SQLite connection — for use from another thread."""
        return Campaign(self.home, self.llm, slug=self.slug, notes_dir=self.notes_dir)

    def seed(self) -> list[str]:
        """Queue and run the recipe baselines that are not in the ledger yet."""
        from . import recipes
        have = {r["change"] for r in self.conn.execute(
            "SELECT change FROM experiments WHERE source='recipe'")}
        out = []
        for name in recipes.names():
            change = recipes.DESCRIPTIONS[name]
            if change in have:
                continue
            eid = store.queue(self.conn, family=name, hypothesis="standard baseline",
                              change=change, code=recipes.script(name),
                              source="recipe", author="(recipe)")
            out.append(self.run_one(eid, "recipe"))
        return out


def demo() -> None:
    """Self-check the parsing and the parent-validation guard — the two places
    a malformed model reply could quietly corrupt the ledger."""
    good = """
FAMILY: xgb
PARENT: 12
HYPOTHESIS: Rows missing `income` carry 1.4x loss, so an explicit
missing-count feature should help the model separate them.
CHANGE: add per-row missing-count feature
```python
import os, numpy as np, pandas as pd
# ... a script long enough to be real, padded here for the check ...
""" + "# padding\n" * 30 + """
print("CV_SCORE: 0.97")
```
"""
    p = _parse(good)
    assert p["family"] == "xgb" and p["parent_id"] == 12, p
    assert p["change"] == "add per-row missing-count feature", p
    assert "import os" in p["code"], p

    # No code block at all -> None, not an exception.
    assert _parse("FAMILY: xgb\nno code here") is None
    # A stub script is rejected rather than queued and run.
    assert _parse("FAMILY: xgb\n```python\npass\n```") is None
    # "none" parent parses as no parent, not as a crash.
    stub = good.replace("PARENT: 12", "PARENT: none")
    assert _parse(stub)["parent_id"] is None
    # Missing fields fall back rather than raising.
    bare = "```python\n" + "x = 1\n" * 60 + "```"
    assert _parse(bare)["family"] == "other"

    # `array or None` raises on a numpy array and killed a 40-turn run on
    # turn 1. Guard the sentinel comparison explicitly.
    import numpy as _np
    class _C:
        _y_cache = _np.zeros(5)
        _y = Campaign._y
    assert _C()._y().shape == (5,)
    class _D:
        _y_cache = False
        _y = Campaign._y
    assert _D()._y() is None

    print("agent.py: 8 checks passed")


if __name__ == "__main__":
    demo()
