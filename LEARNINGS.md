# Learnings

Concepts, not history. Each transfers to any agent — Kaggle is just the excuse.
(For what we did and when, see CHANGES.md. v1's learnings still hold; these are
what v2 added or corrected.)

---

## Search vs. diagnosis

**An agent that can only score cannot improve — it can only search.** v1 knew
model A scored 0.9702 and B scored 0.9698 and had nothing to say about *where*
either lost. The only move left is "try more configurations", and that
converges on whatever the configuration space has a lot of. Give it a
diagnostic and the space it searches becomes "measured weaknesses", which is
smaller and vastly better aimed.

**Save the intermediate artifact, not just the score.** The single highest-
leverage line in v2's contract is "print OOF and PRED". One `np.save` per run
turns a finished experiment from a dead end into a queryable object — error
analysis and ensembling both fall out of it for free. Ask of any agent: what
does a completed task leave behind that a later task can *use*?

**Consensus ingredients make a consensus result.** v1 stacked other people's
published predictions and landed precisely on the score everyone else's stack
reaches. Borrowing the field's inputs guarantees the field's output. If the goal
is to beat a population, at least some ingredient has to be yours.

**Rank findings by `effect × sqrt(n)`, never effect alone.** A slice with 3x
the loss and 30 rows is a coin flip; chasing it burns a day. The weighting keeps
big-and-moderately-bad above small-and-spectacular — which is where the real
work is.

---

## Verdicts and noise

**Three verdicts beat two, and the third is the useful one.** `better` /
`inconclusive` / `worse`, where the noise floor is the fold spread. Without
`inconclusive`, an agent learns superstitions from flukes and builds a month of
strategy on 0.001 of nothing.

**A crash is not evidence against the hypothesis.** Classify failures
(`bug`/`timeout`/`oom`) and never let one produce a `worse` verdict. Agents that
conflate these abandon good ideas over typos and never revisit them.

**Freeze the comparison substrate before you compare anything.** Shared folds,
written to disk once. Any per-run randomness in the evaluation makes small real
differences indistinguishable from noise — and small real differences are
usually the whole game.

**Validate the shape of what you saved, not just that you saved it.** A
wrong-length prediction array loads cleanly, blends cleanly, and silently
corrupts everything downstream. Cheap to check at write time; very expensive to
find three days later.

---

## Human input

**A hint must be consumed, not installed.** Append "try target encoding" to a
system prompt and the agent does target encoding forever. File it as a row that
gets one guaranteed turn and is then spent, and you get the redirect without the
permanent bias. Same mechanism, opposite half-life.

**The steering channel has to reach where the human actually is.** v1 had
Telegram code and empty credentials, so the highest-leverage input in the whole
design — a human noticing something and saying so — was never once delivered. A
feature that degrades silently to a no-op will do exactly that, indefinitely.

---

## Paying for agents

**Subscription and API are different resources, not different prices.** One
throttles, the other bills. A month-long autonomous run wants both: the
subscription for the grind, the API for what the throttle blocks. Putting them
behind one interface makes the billing decision reversible instead of
architectural.

**Measure the harness overhead before assuming it's free.** A `claude -p` call
carries ~36K tokens of Claude Code's own system prompt. Irrelevant on a
subscription; ~$0.12 per call on the API. The same prompt has two very different
costs depending on how it's delivered, and only measurement reveals it.

**Log notional spend separately from real spend.** A subscription run that
records "would have cost $340" is useful information and a catastrophic input to
a budget cap. Same number, two meanings — keep the flag.

**The debugger seat runs on most turns.** It fixes the boring failures that make
up the majority of generated-code errors, so it is where model choice compounds
hardest. v1's own ledger: debugger $0.69 of $22.66 total, because it was a small
model. That is the whole argument for seats.

---

## Sizing

**Most of a system built to impress is subtractable.** v1: 9,425 LOC, 27
modules. v2 does strictly more, better, in ~1,900 across 10 — because the
dashboard, the chat seat, the notebook publisher, and the HPO module all served
the operator's sense of completeness rather than the score. Ask what each module
would cost to *not* have.

**A `demo()` per module beats a test directory.** 59 assertions, no framework,
runnable individually or all at once. The bar is "would this fail if the logic
broke", and for a system nobody else maintains that is the entire requirement.

## Campaign 2 — what the hand EDA changed (2026-09-03)

Ran a full EDA by hand rather than through the agent. It closed three lines of
attack and opened one, and it explains results we already had:

* **`Concern x Subsidy` is a 10-cell crosstab scoring 0.91546 alone**, vs 0.8428
  for concern by itself. No other feature pair clears 0.852. Income is *stable*
  across concern strata (0.669-0.687), so it is additive in logit space —
  crossing it with concern (exps 23, 31, 33) was effort spent against the data.
* **Round-number population**: `income % 1000 == 0` is 61,674 rows at a 0.0443
  target rate vs 0.1879 outside. Explicit modular flags are worth **+0.0015**,
  matching the best gains of the campaign.
* **Zero duplicates, zero train/test signature overlap.** Record linkage cannot
  work here — which is precisely why exp48 landed at 0.9943 error correlation
  with the champion while looking novel.
* **Adversarial AUC 0.5001.** No shift; CV is trustworthy (7 submissions, mean
  gap +0.00056), and shift-based tricks are inert.
* **Ceiling ladder**: concern 0.8428 -> +income 0.8810 -> all raw 0.9415 ->
  best single 0.945476. Feature engineering has bought +0.004 over raw.

Also fixed this session: `pytabkit` installed and advertised, with RealMLP
verified working (43s/fold on 30k rows) and the pandas-4 arrow-`str` dtype trap
documented — the same trap that had killed rival_b's exp22.

**Still untapped: the original (non-synthetic) dataset has never been merged.**
