# Changes

What we did and why. Newest first. Not a spec — the code is the spec.

## 2026-09-24 — cloud container, local boards, paired verdicts, role A/B

**No route to Kaggle from the cloud container** (network policy 403s
kaggle.com; no credentials). Rather than stop, built `local.py`: a
hidden-label competition from a real dataset with a public/private board.
`bank` is UCI bank marketing — the source s5e8 was synthesised from. Private
scores go to a file the prompts never read. s6e9 work resumes the moment the
environment allows kaggle.com.

**The verdict rule was throwing away real gains.** It compared a mean gain to
the raw fold spread, but every model shares the folds, so most of that spread
is fold difficulty and cancels in a paired comparison. First bank turn: +0.0016
against a 0.0027 spread → "inconclusive"; all five folds improved, paired t 6.3.
On s6e9 the fold spread (0.0008) exceeded the top-10% band (0.0005), so the old
rule could almost never say `better`. Now: paired t (one-sided 95%) AND an
effect >= 5% of the fold spread. The second clause exists because a re-blend
4e-7 higher on every fold passed a pure t-test and spent a slot.

**Everything else built today, each from a measured failure:**

* Linux support (`/proc/meminfo`), and `claude -p` children stripped of the
  parent session's env — without it they resumed the operator's session.
* Recipes: LGBM/XGB/CatBoost baselines at turn 0 with no LLM call.
* Pipelining: proposal i+1 is written while experiment i trains.
* Portfolio: nested-CV blend refreshed every turn, shown to rivals, and
  eligible for submission; blends excluded from champion/analyst/novelty.
* Multiclass end to end ((n, k) OOF; accuracy / MAP@k / logloss); regression
  proxy `abalone` (RMSLE, s4e4's source) and multiclass `wine` (MAP@3).
* Smoke runs check the OOF on harness-chosen rows (`$SIS_IDX`): bankB exp7's
  misaligned NN (printed 0.943, array 0.502) was only caught after its full run.
* Full-run bugs get one repair; the OOF check distinguishes per-fold scale
  mismatch (ranking objectives) from misalignment.
* Researcher seat: public notebooks → notes (ran on the five s6e9 notebooks;
  surfaced the generator's income breakpoints and orig-data target-mean use).
* Porter seat: a win is carried onto the other GBDT families automatically.
* `sis2 finalize`: deterministic seed-bagging (2 of 3 bags paired-better on a
  bankB copy); a bag supersedes its original in the blend pool.
* Notes scoped per competition (`notes/<slug>/` + `notes/_global/`).

**Role A/B on the bank proxy** (identical split/folds/recipe seed, 2 threads):

| | turns | wins | fail | best single | blend | private of best-public pick | notional $ |
|---|---|---|---|---|---|---|---|
| A Opus + Fable, each writes code | 9 | 1 | 1 | 0.95135 | 0.95219 | **0.95462** | 30.39 |
| B Opus + Fable decide, Sonnet codes | 12 | 3 | 1 | **0.95187** | 0.95219 | 0.95417 | 23.14 |
| C Opus + Sonnet, each writes code | 12 | 1 | 1 | 0.95028 | 0.95100 | 0.95413 | **8.53** |
| baseline (one LGBM, no agent) | – | – | – | 0.94947 | – | 0.95291 | 0 |

Read with care — one campaign per arm, private differences of ~0.0005:

* Every arm beat the no-agent baseline on the hidden private split by
  +0.0012 to +0.0017. The loop adds held-out value.
* A had the fewest single-model wins and the best private score: Fable's
  neural nets (transformer 0.9462, PLR-MLP 0.9510) lost alone but decorrelated,
  and the blend used them. Cost: one transformer ran 55 minutes on CPU.
* B turned the most ideas into wins fastest; Fable's spec-only calls cost ~half
  its code-writing calls.
* C was cheapest and weakest on ideas.
* Two of B's three wins were a rival hand-porting one feature set to other
  families — the porter now does that.

Then D = hybrid (Opus codes; Fable thinks, Sonnet implements; porter on) —
killed by a container restart after 8 agent turns: 3 agent wins (5 more from
ports/bags), blend 0.95178, private 0.95367. A paired bootstrap on the private
rows found only one significant gap between arms: A over D (+0.00094, CI
+0.00024..+0.00165); all others straddle zero. So: A (competitors, each
writes code) is the default; the porter is now opt-in (`SIS_PORT=1`) because
its "wins" added no blend diversity; `compare` no longer counts ports/bags as
wins; `sis2 run --total N` resumes to N agent turns after a restart. Final
table and recommendation: ARCHITECTURE.md §4.

## 2026-09-02 — s6e9 live, advisor swarm

**Rules accepted; campaign initialised.** 668,665 train / 286,571 test, 14
features, 10-fold stratified split frozen to `folds.npy`.

**Two recon findings that change the plan:**

- **Zero missing values in all 15 columns.** The s6e8 winner's headline
  discovery — missingness features — has nothing to work with here. Copying that
  playbook would have wasted days. `profile()` now says so explicitly so the
  agent does not invent features that cannot exist.
- **The board is brutally tight**: 489 teams, top `0.94631`, top-10% `0.94577`
  — a 0.00054 spread. Fold-spread discipline is the whole game.

**Two bugs found by running against real data:**

- `profile()` crashed on a non-numeric target (`Will_Buy_EV` is "Yes"/"No").
  It now reports value counts and names the positive class — guessing the
  encoding backwards mirrors the AUC and looks like a catastrophic model.
- Column names were truncated to 22 chars, rendering
  `Charging_Stations_Near_Home` and `..._Work` as the *same string*. The agent
  writes code against that listing; names are never truncated now.

**Advisor swarm added** (`LLM.advise`, `SIS_ADVISOR_MODELS`). Outside models
give ideas, never code, one rotating call per turn, non-blocking on failure.
Kimi K3 verified live producing column-specific, genuinely useful direction.

**Benchmarked `nemotron-3.5-lightning-30b-a3b`** for the debugger: 5.0s vs the
550B Ultra's 1.5s. Rejected — provisioning dominates parameter count on this
endpoint.

## 2026-09-02 — NVIDIA integration, measured

**Benchmarked three NVIDIA-hosted models on the debugger's real job** (repair a
broken training script) instead of picking by reputation. All three produced the
same correct fix:

| model | latency | verdict |
| --- | ---: | --- |
| `nvidia/nemotron-3-ultra-550b-a55b` (thinking off) | 2.9s | **in use** |
| `deepseek-ai/deepseek-v4-pro-0813` (thinking off) | 39.7s | swarm-tier |
| `moonshotai/kimi-k3` | 203.4s | swarm-tier |

The debugger runs up to 3x per experiment, so a 70x latency gap decides it.
`enable_thinking: false` on Nemotron cut output 190 -> 100 tokens for an
identical fix.

**Two bugs the benchmark exposed, both fixed:**

- *Throttle contamination.* Any backend raising `Throttled` set
  `_throttled_until`, which means "the SUBSCRIPTION is throttled". An NVIDIA 503
  would have pushed ten minutes of work onto the billed Anthropic API for no
  reason. Now only the CLI backend can set that window.
- *Silent empty replies.* A reasoning model that spends its whole budget on
  `reasoning_content` returns `content: ""`. That would have flowed onward as a
  successful no-op repair. Now it raises with the finish_reason and reasoning
  length, and falls through to the next backend.

**Free tier 503s under concurrency** (`Service temporarily overloaded`,
observed on 1 of 4 parallel calls). 429 and 5xx are both treated as "ask someone
else" rather than as a config error.

**Added `sis2 hint`** — terminal steering, same one-shot consumed-row mechanism
as Telegram. Telegram deferred; the bot was never created and the campaign can
be steered from a second terminal for now.

**Deleted the `implementer` seat** — defined, configurable, documented, and
never once called. Dead config that misrepresented where the money goes.

## 2026-09-02 — v2 from scratch

**Diagnosed v1 before writing anything.** v1's s6e8 run: best CV `0.970177`,
LB `0.97125`, 252nd. The winner's ensemble was CV `0.97098`, LB `0.97207` — v1's
CV was *higher* and its LB *lower*. A commenter independently measured that the
whole public-notebook pool, stacked, tops out around `0.9701`. v1 landed exactly
there because `harvest.py` (900 LOC) scraped and stacked other people's
published OOF arrays. It converged on the consensus, then optimised noise on top.

**So the loop changed in kind, not degree.**

| | v1 | v2 |
| --- | --- | --- |
| loop | UCT tree search over configs | error analysis → one targeted fix → verdict |
| a run saves | CV + submission.csv | **+ OOF array + test array** |
| memory | sqlite lessons | append-only `notes/*.md`, re-read every campaign |
| agents | one strategist | two rivals on different models, alternating |
| ensemble | harvest public OOFs | hill-climb over *our own* pool, at the end |
| size | 9,425 LOC / 27 modules | ~1,900 LOC / 10 modules |

**Built `analyst.py`** — the module v1 had no equivalent of. Given
`(y, oof, X)` it reports: calibration by decile; slices carrying excess loss
(ranked `lift * sqrt(n)`, so big-and-bad beats tiny-and-spectacular);
missingness (loss *and* target-rate shift per column, plus count-of-missing
per row); pairwise cells worse than either parent slice; error-correlation vs a
rival family with the 50/50 blend score. Self-check plants a blind spot and
asserts the report recovers it.

**Made OOF/PRED mandatory in the runner contract**, with shape and finiteness
checks on full runs. A wrong-length OOF is worse than a missing one: it loads,
blends, and silently corrupts every later comparison.

**Froze the folds.** `folds.npy` written once at init, stratified when the
target is categorical. v1 let each script build its own split, so a 0.001
difference between two experiments was indistinguishable from split noise —
and 0.001 is the margin that decides a playground competition.

**Two billing backends behind one interface.** `claude -p` (subscription,
throttles) first, Anthropic API (bills) on throttle. Discovered by measurement
that a `claude -p` call carries ~36K tokens of Claude Code's own system prompt —
free on subscription, ~$0.12/call on the API. `usage.jsonl` keeps *billed* and
*notional* dollars separate so a subscription run doesn't appear to have spent
hundreds it never did, and so the daily cap can't fire on spend that never
happened.

**Ported rather than rewrote the Kaggle plumbing** — kernel log is JSON not
text, `--accelerator` not `machine_shape`, leaderboard needs `--download` not
`--show`, six submission validations. Every one of those was learned from a real
failure in v1 and none of it was worth re-deriving.

**Auto-submit rule** (`cli.maybe_submit`): send when the champion beats the best
already-submitted CV by more than the fold spread, and slots remain. Unused
Kaggle slots expire — up to 140 over a month — but the noise-floor clause stops
it burning five a day on runs it can't tell apart.

**Hints are consumed once.** Telegram reply → `hints` row → appears in exactly
one prompt → marked spent. Carried from v1's design note; the failure it avoids
is a hint in the system prompt turning into a month-long obsession.

**Dropped from v1:** harvest, dashboard, hpo, fusion, publish, chat, researcher
seat. ~5,500 LOC that either chased the public ceiling or served the operator
rather than the score.

**59 self-check assertions**, one `demo()` per module, no test framework.
`sis2 check` runs them all.

**Found:** v1's `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are empty strings —
the steering channel never worked in v1 either.
