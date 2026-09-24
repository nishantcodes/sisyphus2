# sisyphus 2

An autonomous Kaggle agent that improves one model by studying where it is
wrong, rather than searching a space of configurations.

```
sis2 init s6e9 --target Will_Buy_EV --metric auc
sis2 run  s6e9 --turns 40 --hours 8
sis2 status s6e9
sis2 hint   s6e9 have you studied the missing values     # steer the next proposal
sis2 blend  s6e9
sis2 finals s6e9
```

`sis2 hint` is the steering channel. The text becomes a row that appears in
exactly **one** proposal prompt and is then marked spent — safe to run from a
second terminal while a campaign is mid-flight. Telegram uses the same mechanism
when configured.

## The loop

```
champion  = best scored experiment in this family
brief     = analyst.report(champion)     <- WHERE it is losing, measured
proposal  = rival.propose(brief, notes, history)
result    = run(proposal)                <- smoke at 5%, then full
verdict   = compare to parent, fold spread as the noise floor
if better: write a discovery note, consider submitting
```

v1 searched configurations by UCT and landed on `CV 0.970177 / LB 0.97125` in
s6e8 — 252nd, and almost exactly the score the entire public notebook pool
converges to when stacked. Its CV was *higher* than the winner's and its LB was
lower. It had no way to ask where a model was losing, so it did the only thing
available and searched wider.

The change here is `analyst.py`: given true labels, out-of-fold predictions and
the feature frame, it reports calibration drift, loss-carrying slices,
missingness effects, missing interactions, and error-correlation against a rival
family. That report is the input to every proposal.

## Layout

| file | what it owns |
|---|---|
| `run.py` | the stdout contract; local execution, memory watchdog, smoke runs |
| `store.py` | ledger — experiment tree, three verdicts, hints, notes index |
| `analyst.py` | **error analysis** — the thing v1 had no machinery for |
| `agent.py` | the turn: brief → proposal → execute → verdict → note |
| `blend.py` | greedy hill-climb ensemble over *our own* OOF pool |
| `kaggle.py` | data, shared folds, six submission validations, GPU kernels |
| `llm.py` | subscription (`claude -p`) first, API on throttle, usage ledger |
| `metrics.py` | the only file that knows what the metric means |
| `notify.py` | Telegram out; replies become one-shot hint rows |
| `cli.py` | commands, and the auto-submit rule |

`sis2 check` runs every module's self-check (59 assertions, no framework).

## Design rules worth not breaking

**Every experiment saves OOF and test predictions.** This is mandatory in the
contract and enforced by the runner, including array shape and finiteness. It is
what makes error analysis and ensembling possible at all; v1 saved only a score
and a submission, so every finished experiment was a dead end.

**All experiments share one frozen fold assignment**, written to `folds.npy` at
init. v1 let each script build its own split, which made a 0.001 difference
indistinguishable from split noise — and 0.001 decides a playground competition.

**Three verdicts, not two.** A gain inside the fold spread is `inconclusive`.
This is the rule that stops the agent building a strategy on a fluke.

**A crash is not a refutation.** Failures are classified `bug` / `timeout` /
`oom` and never produce a `worse` verdict. A KeyError says nothing about whether
the hypothesis was good.

**Human hints are consumed once.** A Telegram reply becomes a row that gets one
guaranteed turn and is then never seen again. Putting it in the system prompt
instead makes the agent chase that one idea for the rest of the month.

**The blend pool is ours.** v1 mounted other people's published OOF arrays and
stacked them, which is precisely why it landed on the public ceiling. If your
ingredients are the consensus, so is your ensemble.

## Billing

Two backends behind `LLM.complete()`, subscription-first:

- **`claude -p`** — runs on the Claude subscription. No marginal cost; throttles
  rather than bills. Carries Claude Code's own ~36K-token system prompt per call,
  which is free here and would be ~$0.12/call on the API.
- **Anthropic API** — bills per token, never throttles. Used automatically when
  the subscription is throttled, subject to `SIS_DAILY_CAP` (default $20).

- **Any OpenAI-compatible endpoint** — configured and live on **NVIDIA
  Nemotron 3 Ultra**. Set `SIS_COMPAT_BASE_URL`, `SIS_COMPAT_API_KEY`,
  `SIS_COMPAT_MODEL`, and optionally `SIS_COMPAT_EXTRA_BODY` for
  provider-specific fields.

Measured on the debugger's real job (repair a broken training script), same
correct fix from all three:

| model | latency | notes |
|---|---:|---|
| `nvidia/nemotron-3-ultra-550b-a55b`, thinking **off** | **2.9s** | in use |
| `deepseek-ai/deepseek-v4-pro-0813`, thinking off | 39.7s | swarm-tier |
| `moonshotai/kimi-k3` | 203.4s | swarm-tier |

The debugger runs up to 3× per experiment, so 70× latency is the whole
decision. Kimi and DeepSeek are fine where one call per turn is the budget —
they are not fine in the repair loop. `enable_thinking: false` on Nemotron
halves output tokens (190 → 100) for an identical fix; the debugger reads a
traceback, it does not solve a hard problem.

### Advisor swarm (the writeup's phase 4)

`SIS_ADVISOR_MODELS` is a rotating list of outside models asked for **ideas, not
code**, once per turn, before the rival proposes. Different architectures carry
different priors, and rotating them is the point — the diversity comes from
asking *different* models on different turns.

Measured on the real s6e9 brief, Kimi K3 named specific column crosses
(`Near_Work / (Near_Home + 1)`, `Range_Anxiety x Daily_Commute`) and reasoned
that with only 14 raw features the ceiling is set by inputs rather than
algorithm choice. Latency 144-203s, which is free at one call per turn.

**Never blocking.** The free tier answered 1 of 3 concurrent calls in testing
(one 504, one 503), so every failure path returns empty and the turn proceeds
without advice. The advice is framed in the prompt as a second opinion to weigh,
not an instruction — an advisor that could dictate a turn would just be a worse
rival.

Rejected after measurement: `nemotron-3.5-lightning-30b-a3b` for the debugger
seat. It is 3x *slower* than the 550B Ultra on this endpoint (5.0s vs 1.5s) —
provisioning dominates parameter count here.

The free tier **503s under concurrency** (`Service temporarily overloaded`,
observed). 429 and 5xx are both treated as "ask someone else" and fall through
to the next backend.

`usage.jsonl` records all of them and keeps the dollars distinct: subscription
spend is logged as *notional* (what it would have cost) and never counted
against the cap; third-party tokens are recorded but no dollar figure is
claimed, since rates vary by provider and tier.

**The compat backend is for throttle relief, not savings.** It serves only
`debugger` and `reporter` (`SIS_COMPAT_SEATS`), and only while the subscription
is throttled — when the subscription is healthy nothing is sent to a third party,
because it is already free and already better. It is **never** given a rival
seat: the proposals are the product, and the entire premise here is that a
frontier model's data-science judgement is what makes the loop work.

The arithmetic, from v1's real ledger: the debugger was **$0.69 of $22.66 —
3%**. Reasoning seats were ~63%. Cheap models cannot save meaningful money in a
system whose cost is concentrated where quality decides the outcome.

Override the primary with `SIS_BACKEND=api`, seats with `SIS_RIVAL_A` /
`SIS_RIVAL_B` / `SIS_DEBUGGER` / `SIS_REPORTER` (aliases: `opus`, `sonnet`,
`haiku`, `fable`). The two rivals default to different models on purpose — a
rivalry between two instances of one model is just a more expensive single agent.

## Auto-submit

Sends when a new champion beats the best already-submitted CV **by more than the
fold spread**, and slots remain today. Unused Kaggle slots expire, and over a
month that is up to 140 discarded reads on the test distribution; the noise-floor
clause is what stops it spending them on runs it cannot tell apart.

Marking the two final submissions is a browser click — Kaggle exposes no API for
it. `sis2 finals` prints the picks and the reasoning.

## Setup

`.env` (copied from v1) needs:

- `ANTHROPIC_API_KEY` — set
- Kaggle credentials in `~/.kaggle/` — set
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — **empty**; get a token from
  [@BotFather](https://t.me/botfather) and your chat id from
  [@userinfobot](https://t.me/userinfobot). Without these, steering degrades to
  a silent no-op and `sis2 ping` says so.

You must also accept the competition rules in a browser before `init` can
download anything.
