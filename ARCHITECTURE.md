# Architecture

How the agents talk, which model sits where, and why — with the measurements
that decided it. For the per-module map see README.md; for history, CHANGES.md.

## 1. Agents do not talk to each other. They share a blackboard.

There is no agent-to-agent chat anywhere in sis2. Every seat reads from and
writes to one measured record, and the harness — not a model — is the only
thing that writes numbers into it:

```
                 ┌──────────────── blackboard (work/<slug>/) ────────────────┐
                 │ ledger.db   experiment tree: code, parent, CV, paired Δ,  │
                 │             verdict, failure class, author                │
                 │ exp*/       OOF + test arrays of EVERY run                │
                 │ analyst     where the champion loses (slices, pairs, cal.)│
                 │ novelty     error-correlation of every model vs leader    │
                 │ portfolio   nested-CV blend + weights (what we'd submit)  │
                 │ notes/      per-competition findings + _global lessons    │
                 │ hints       one-shot human steering rows                  │
                 └───────────────────────────────────────────────────────────┘
   reads ▲  │ writes                                               ▲
         │  ▼                                                      │ measures
  rival A, rival B ──proposal──▶ runner ──arrays──▶ judge / portfolio / analyst
  (alternate turns)   (FAMILY, PARENT,   (smoke 5%,   (paired t-test, nested
                       HYPOTHESIS,        repair,      blend, error analysis —
                       CHANGE, code)      full run)    no LLM anywhere here)
```

Why a blackboard rather than a conversation:

* **Ground truth beats persuasion.** Two models debating converge on whichever
  argues better. Two models reading the same paired-t column converge on what
  the data says. Every claim a rival can make ("this helped") is checked by
  code before any other rival sees it.
* **It scales in rivals for free.** Adding a third proposer costs one seat
  definition; nobody has to learn to talk to it.
* **It survives restarts and model swaps.** The whole state is files; a
  campaign resumed next week with different models picks up exactly where it
  stopped. A chat transcript would not.
* **Hints stay one-shot.** Human steering is a row consumed by one proposal,
  never a standing instruction (the anti-obsession rule from v1).

The only structured "messages" are artifacts: a proposal (five fields + code),
a verdict row, a discovery note, a hint.

## 2. Seats

| seat | job | default model | why this model |
|---|---|---|---|
| `rival_a` | propose one change + write the code | Opus 5.5 | judgement is the product |
| `rival_b` | same, alternating turns | Fable 5.1 | a *different* prior; two copies of one model is one expensive agent |
| `executor` | write code from a spec (only with `SIS_SPLIT=1`) | Sonnet 5 | see §4 |
| `debugger` | repair a failing script (≤3x per experiment) | Haiku 4.5 | reads a traceback; runs most turns |
| `reporter` | write the discovery note | Haiku 4.5 | low stakes, high volume |
| `researcher` | distil public notebooks into notes | Sonnet 5 | reading comprehension |
| advisors | outside ideas, never code (`SIS_ADVISOR_MODELS`) | any OpenAI-compatible | different lab = different priors |

Non-LLM seats do everything that has a right answer: the runner (contract,
memory, timeouts, OOF integrity at smoke and full scale), the judge (paired
fold test + minimum effect), the portfolio (nested-CV hill-climb), the analyst,
the recipe bank (LGBM/XGB/CatBoost baselines at turn 0), and the submit rule.

## 3. A turn

```
seed (once)  recipes → 3 scored, blendable baselines, zero LLM calls
loop:
  rival(i+1) writes its proposal  ─┐  concurrently (pipelined; it sees
  experiment i trains             ─┘  exp i as "running", not its result)
  smoke 5% on harness-chosen rows → OOF checked against those labels
    fail → debugger (≤3) ; pass → full run ; full-run bug → 1 repair + rerun
  judge: paired Δ vs parent on shared folds → better / inconclusive / worse
  portfolio: re-blend with nested CV; new row only if paired-better
  submit: best candidate (single OR blend) if paired-better than best sent
  note: reporter writes why it worked (better / baseline only)
```

## 4. Competitors, or a thinker and an implementer? — measured

The question: do two frontier models each writing their own code (A) beat the
same two models deciding while a mid-tier model implements their spec (B)?

Both ran 12 turns in parallel on identical data (local bank proxy, same split
and folds, same recipe seed, 2 threads each). Results are recorded in
`work/bankA.log` / `work/bankB.log`; the table is filled from those logs:

| | A: competitors | B: think + implement |
|---|---|---|
| see CHANGES.md 2026-09-24 | | |

(The measured table and verdict live in CHANGES.md, next to the date they were
taken, so this file does not go stale when the numbers are re-measured.)

## 5. Judging and submitting

* **Paired verdicts.** Every model is scored on the same frozen folds, so the
  judge compares fold-by-fold deltas (mean / SE, one-sided t at 95%) and
  additionally requires the gain to exceed 5% of the fold spread. The v2.0 rule
  (gain vs raw fold spread) called a +0.0016, t=6.3 gain "noise"; a pure t-test
  called a 4e-7 re-blend "better" and spent a slot. Both are fixed.
* **The board wants the ensemble.** `best_candidate` includes portfolio blends;
  the research loop still studies the best *single* model (a blend cannot be
  extended by one change).
* **Honest blend CV.** Weights for fold k are chosen without fold k, so the
  blend competes under the same rule as the models it is made of.

## 6. Generality

* Targets: binary (1-D probabilities), regression (1-D values), multiclass
  ((n, k) matrix, sorted-label columns). Metrics: AUC, log loss, RMSE/RMSLE/MAE,
  accuracy, F1, MAP@k, multiclass log loss.
* No Kaggle? `sis2 init <slug> --local <csv|url|bank|abalone|wine|adult>` builds
  a hidden-label competition with a public/private board. Private scores never
  enter the ledger, so they cannot leak into a prompt.
* Hardware: macOS (vm_stat, MPS) and Linux (/proc/meminfo, CUDA/CPU); Kaggle
  GPU kernels when credentials exist.
