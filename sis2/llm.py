"""One interface, two ways to pay for it.

`claude -p` runs on the Claude subscription: no marginal cost per call, but a
quota that throttles rather than bills. The Anthropic API bills per token with
no throttle. Neither is strictly better for a month-long autonomous run, so
both live behind `LLM.complete()` and the default is subscription-first with
API fallback on throttle.

**The two backends want different prompts, and that is not cosmetic.** A
`claude -p` call carries Claude Code's own system prompt — measured at ~36K
tokens on this machine — on top of whatever we send. On the subscription that
is free and irrelevant. Through the API it is about $0.12 of overhead *per
call* before our prompt is even considered. So the CLI backend gets our full
context, and the API backend gets the same context with nothing wasted around
it.

Seats, kept from v1 because the economics were the lesson: a debugger runs on
most turns and putting a large model in that seat is how an agent quietly costs
$200/day. Every seat is env-overridable and no model is named anywhere else in
the codebase.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

# seat -> (model, effort). The two rivals are deliberately DIFFERENT models:
# the writeup's central finding was that two architectures compete productively
# because they fail differently — one creative, one careful — and a rivalry
# between two instances of the same model is just a more expensive single agent.
SEATS: dict[str, tuple[str, str]] = {
    "rival_a":  (os.getenv("SIS_RIVAL_A", "opus"), "high"),
    # FULL model ids, not short aliases. `claude -p --model haiku` silently
    # resolves to claude-sonnet-5 — it does not error, it just serves a bigger
    # model. 24 debugger calls ran that way. `opus` and `sonnet` do resolve
    # correctly, but a full id is honoured on BOTH backends, so use it.
    "debugger": (os.getenv("SIS_DEBUGGER", "claude-haiku-4-5"), "low"),
    "reporter": (os.getenv("SIS_REPORTER", "claude-haiku-4-5"), "low"),
    # Writes the script from the thinker's spec when SIS_SPLIT=1.
    "executor": (os.getenv("SIS_EXECUTOR", "sonnet"), "medium"),
}

# An optional third rival, off unless SIS_RIVAL_C names a model. Measured on
# s6e9 with Kimi K3: a proposal takes ~294s against Nemotron's 38s, and its
# baseline scored 0.94179 vs Nemotron's 0.94184 — indistinguishable inside a
# 0.0009 fold spread, and it needed a debugger repair the others did not. So
# it buys a third genuinely different prior at roughly 4x the turn cost, and
# whether that trade is worth it is a judgement call rather than a fact.
# Every optional rival is CONDITIONAL, so removing its env var removes the
# seat. rival_b used to be unconditional with a "sonnet" default, so
# commenting out SIS_RIVAL_B silently swapped Nemotron for Sonnet instead of
# retiring the seat — the same word meaning two different things.
if os.getenv("SIS_RIVAL_B"):
    SEATS["rival_b"] = (os.environ["SIS_RIVAL_B"], "high")

if os.getenv("SIS_RIVAL_C"):
    SEATS["rival_c"] = (os.environ["SIS_RIVAL_C"], "high")

# A fourth rival, off unless SIS_RIVAL_D names a model. Intended for a
# thinker-only seat: with SIS_SPLIT=1 an Anthropic rival decides and the
# `executor` seat writes the script, so an expensive model spends its tokens
# on judgement (~600 output) rather than on boilerplate (~3,400).
if os.getenv("SIS_RIVAL_D"):
    SEATS["rival_d"] = (os.environ["SIS_RIVAL_D"], os.getenv("SIS_RIVAL_D_EFFORT", "high"))


def rivals() -> list[str]:
    """The proposer rotation, in order. Turns alternate across whatever exists."""
    return [s for s in ("rival_a", "rival_b", "rival_c", "rival_d") if s in SEATS]

# Seats that may be served by a third-party OpenAI-compatible endpoint.
# Deliberately NOT the rivals: their proposals are the product, and the whole
# premise of this design is that a frontier model's data-science judgement is
# what makes the loop work. The debugger and reporter are high-volume and
# low-stakes — a repair either fixes the traceback or it doesn't, and a note
# that reads slightly worse costs nothing.
COMPAT_SEATS = {s.strip() for s in
                os.getenv("SIS_COMPAT_SEATS", "debugger,reporter").split(",")
                if s.strip()}

# CLI aliases -> API model ids. The CLI takes short names; the API needs the
# full id, and keeping the mapping here means a seat definition works on both.
API_MODEL = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    # Fable 5.1 is API-ONLY: `claude -p --model claude-fable-5-1` returns
    # "Claude Code 2.1.231 does not support" it, and the bare `fable` alias
    # resolves to Fable 5. A fable seat must therefore run with SIS_BACKEND=api.
    "fable": "claude-fable-5-1",
    "fable-5": "claude-fable-5",
}

DAILY_CAP_USD = float(os.getenv("SIS_DAILY_CAP", "20"))
CLI_TIMEOUT = int(os.getenv("SIS_CLI_TIMEOUT", "1200"))
_LOCK = threading.Lock()

# Substrings that mean "throttled, try the other backend" rather than "broken".
_THROTTLE = ("rate limit", "rate_limit", "usage limit", "quota",
             "too many requests", "429", "overloaded")


class Throttled(RuntimeError):
    """The subscription is out of quota for now — fall back, do not retry."""


# ------------------------------------------------------------------ backends

class CLIBackend:
    """Claude Code in headless mode, on the user's subscription.

    Deliberately shells out to the `claude` binary rather than importing the
    Agent SDK: the binary is already installed, already authenticated, and its
    `--output-format json` contract is stable. One less dependency to drift.
    """

    name = "cli"

    def __init__(self, home: Path | None = None):
        # One pinned session per seat. `claude -p` carries Claude Code's own
        # ~36K-token system prompt on every call, and a FRESH session pays it
        # at the 1.25x cache-WRITE rate every time while `cache_read` stays 0.
        # Resuming a session turns that into a 0.1x cache READ — measured at
        # $0.1638 -> $0.0288 for an identical call, a 5.7x reduction with no
        # change to model, prompt, or output.
        #
        # This is also why splitting a seat into "thinker + executor" is a
        # false economy on this backend: two calls pay the 36K prefix twice.
        # The prefix, not our content, is 96% of the prompt.
        self._sessions: dict[str, str] = {}
        self._store = (Path(home) / ".sessions.json") if home else None
        if self._store and self._store.exists():
            try:
                self._sessions = json.loads(self._store.read_text())
            except Exception:
                self._sessions = {}

    def _remember(self, seat: str, sid: str) -> None:
        self._sessions[seat] = sid
        if self._store:
            try:
                self._store.write_text(json.dumps(self._sessions))
            except OSError:
                pass

    def complete(self, system: str, user: str, seat: str) -> tuple[str, dict]:
        model, effort = SEATS.get(seat, ("sonnet", "medium"))
        base = ["claude", "-p", user, "--output-format", "json", "--model", model,
                "--append-system-prompt", system,
                # This process only ever asks for text back. Denying tools stops
                # a stray file write from the model touching the working tree.
                "--permission-mode", "plan", "--allowed-tools", ""]
        # Session resume ignores --model, so a pinned session silently serves
        # whatever model created it — 24 debugger calls configured as `haiku`
        # were served by claude-sonnet-5 this way. The big seats keep the 5.7x
        # cache saving; the cheap ones keep their model.
        sid = None if seat in ("debugger", "reporter") else self._sessions.get(seat)
        cmd = base + (["--resume", sid] if sid else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT)
        # A stored session can vanish (cleared history, another machine). That
        # is not an error worth failing a turn over — start a fresh one.
        if sid and proc.returncode != 0:
            self._sessions.pop(seat, None)
            proc = subprocess.run(base, capture_output=True, text=True,
                                  timeout=CLI_TIMEOUT)
        raw = proc.stdout.strip()
        if proc.returncode != 0 or not raw:
            err = (proc.stderr or raw or "")[-500:]
            if any(s in err.lower() for s in _THROTTLE):
                raise Throttled(err)
            raise RuntimeError(f"claude -p failed: {err}")
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p returned non-JSON: {raw[:300]}")
        if d.get("is_error"):
            msg = str(d.get("result") or d.get("api_error_status") or "")
            if any(s in msg.lower() for s in _THROTTLE):
                raise Throttled(msg)
            raise RuntimeError(f"claude -p error: {msg[:400]}")
        if d.get("session_id"):
            self._remember(seat, d["session_id"])
        u = d.get("usage", {}) or {}
        return d.get("result", ""), {
            "input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
            "cache_read": u.get("cache_read_input_tokens", 0),
            "cache_write": u.get("cache_creation_input_tokens", 0),
            # Notional on a subscription — what these tokens WOULD have cost on
            # the API. Recorded anyway so the two backends are comparable and a
            # subscription run still has a spend curve to look at.
            "usd": float(d.get("total_cost_usd") or 0.0), "billed": False,
            "model": next(iter(d.get("modelUsage", {})), model),
        }


class APIBackend:
    """The Anthropic API. Bills per token, never throttles us out of a run."""

    name = "api"

    # $ per million tokens: (input, output), cached 2026-06-24.
    PRICES = {"claude-opus-5": (5.0, 25.0), "claude-fable-5": (10.0, 50.0),
              # 5.1 assumed at Fable-5 rates until published; tokens are
              # stored raw so the whole history reprices if that is wrong.
              "claude-fable-5-1": (10.0, 50.0),
              "claude-sonnet-5": (3.0, 15.0), "claude-haiku-4-5": (1.0, 5.0)}
    CACHE_WRITE, CACHE_READ = 1.25, 0.10

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, seat: str) -> tuple[str, dict]:
        import anthropic
        alias, effort = SEATS.get(seat, ("sonnet", "medium"))
        model = API_MODEL.get(alias, alias)
        for attempt in range(4):
            # Fable-family models run thinking ALWAYS ON — sending a `thinking`
            # param at all is a 400 — and their safety classifiers can decline a
            # request outright (HTTP 200, stop_reason "refusal"). Opt into
            # server-side fallbacks so a decline is re-served rather than
            # costing the turn. Sent only for fable; harmless to omit elsewhere.
            kw = {}
            if "fable" in model:
                kw = {"betas": ["server-side-fallback-2026-07-01"],
                      "fallbacks": "default"}
            try:
                # Streaming: max_tokens this large would otherwise risk an HTTP
                # timeout on a long reasoning turn.
                maker = (self._client.beta.messages if kw
                         else self._client.messages)
                with maker.stream(
                    model=model, max_tokens=32000,
                    system=[{"type": "text", "text": system,
                             "cache_control": {"type": "ephemeral"}}],
                    output_config={"effort": effort},
                    messages=[{"role": "user", "content": user}],
                    **kw,
                ) as stream:
                    msg = stream.get_final_message()
                break
            except anthropic.BadRequestError:
                # Beta flag or fallback shape not accepted — retry plainly
                # rather than lose the turn to a parameter we merely preferred.
                if kw:
                    kw = {}
                    continue
                raise
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError) as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 5)
            except anthropic.APIStatusError:
                raise                      # 4xx: never retry, the request is wrong
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"model declined: {getattr(msg, 'stop_details', None)}")
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                 "model": model, "billed": True}
        usage["usd"] = self._cost(model, usage)
        return text, usage

    def _cost(self, model: str, u: dict) -> float:
        rates = self.PRICES.get(model)
        if not rates:
            # An unknown model returning 0.0 is worse than a wrong estimate: the
            # daily cap is computed from these numbers, so a model missing from
            # the table spends without limit and reports nothing. Fable 5.1 hit
            # exactly this. Charge the most expensive known rate instead, so the
            # cap still protects and the figure errs toward caution.
            rates = max(self.PRICES.values(), key=lambda r: r[1])
        inp, out = rates
        return (u["input"] * inp + u["cache_write"] * inp * self.CACHE_WRITE
                + u["cache_read"] * inp * self.CACHE_READ
                + u["output"] * out) * 1e-6


class CompatBackend:
    """Any OpenAI-compatible endpoint — NVIDIA, Groq, Cerebras, Together,
    OpenRouter, Gemini's compat route, and most others.

    One class instead of one per provider, because they all speak the same wire
    format and the only thing that actually differs is a base URL. Configure
    with three env vars:

        SIS_COMPAT_BASE_URL=https://integrate.api.nvidia.com/v1
        SIS_COMPAT_API_KEY=nvapi-...
        SIS_COMPAT_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1

    This exists for **throttle relief**, not for savings. When the subscription
    runs out of quota the fallback is the billed Anthropic API; routing the two
    high-volume seats here keeps that budget for the seats whose output decides
    the competition. It is never used for the rivals — see COMPAT_SEATS.

    Uses httpx directly rather than the `openai` package: it is already a
    dependency, and the request is a single POST.
    """

    name = "compat"

    def __init__(self) -> None:
        self.base = os.getenv("SIS_COMPAT_BASE_URL", "").rstrip("/")
        self.key = os.getenv("SIS_COMPAT_API_KEY", "")
        self.model = os.getenv("SIS_COMPAT_MODEL", "")
        # Provider-specific request fields, as raw JSON. This is where you turn
        # a reasoning model's thinking OFF for the repair seat — measured on
        # Nemotron: same correct fix, 100 output tokens instead of 190. The
        # debugger is not solving a hard problem, it is reading a traceback.
        try:
            self.extra = json.loads(os.getenv("SIS_COMPAT_EXTRA_BODY", "") or "{}")
        except json.JSONDecodeError:
            self.extra = {}
        if not (self.base and self.key and self.model):
            raise RuntimeError(
                "compat backend needs SIS_COMPAT_BASE_URL, SIS_COMPAT_API_KEY "
                "and SIS_COMPAT_MODEL")

    def complete(self, system: str, user: str, seat: str,
                 model: str = "") -> tuple[str, dict]:
        import httpx
        r = httpx.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            # 16k, not 8k: reasoning models on this route spend part of the
            # budget on `reasoning_content` before writing any answer, so a
            # tight ceiling silently truncates them into an empty reply.
            json={"model": model or self.model, "max_tokens": int(
                      os.getenv("SIS_COMPAT_MAX_TOKENS", "16384")),
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  **self.extra},
            timeout=float(os.getenv('SIS_COMPAT_TIMEOUT', '120')))
        # 429 is quota, 5xx is the free tier being busy — measured: NVIDIA
        # returns 503 "Service temporarily overloaded" under concurrency. Both
        # mean "ask someone else", neither means our config is wrong.
        if r.status_code == 429 or r.status_code >= 500:
            raise Throttled(f"{self.model}: {r.status_code} {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"{self.base} returned {r.status_code}: {r.text[:300]}")
        d = r.json()
        u = d.get("usage", {}) or {}
        msg = d["choices"][0]["message"]
        # Verified against NVIDIA's Nemotron route: thinking models return the
        # chain of thought in a SEPARATE `reasoning_content` field and the
        # answer in `content`. We want the answer. But if the model burned its
        # budget thinking and never wrote one, `content` is an empty string —
        # which would otherwise flow onward as a "successful" empty repair and
        # be counted as the model declining to fix anything. Fail loudly so the
        # caller falls through to the next backend instead.
        text = (msg.get("content") or "").strip()
        if not text:
            raise RuntimeError(
                f"{model or self.model} returned no content "
                f"(finish_reason={d['choices'][0].get('finish_reason')}, "
                f"{len(msg.get('reasoning_content') or '')} chars of reasoning). "
                f"Raise SIS_COMPAT_MAX_TOKENS if this repeats.")
        return text, {
            "input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0),
            "cache_read": 0, "cache_write": 0,
            # Third-party rates vary by provider and tier and are not worth
            # hardcoding to go stale. Tokens are recorded; dollars are not
            # claimed. `billed: False` keeps this out of the Anthropic cap,
            # which is the only budget this process can actually enforce.
            "usd": 0.0, "billed": False,
            "model": f"{self.name}:{model or self.model}",
        }


# ----------------------------------------------------------------- the front

class LLM:
    """Subscription first, API on throttle, every call recorded.

    `usage.jsonl` distinguishes *billed* dollars from *notional* ones, because
    conflating them would make a subscription run look like it cost hundreds of
    dollars it never spent — and would make the daily cap fire on spend that
    never happened.
    """

    def __init__(self, home: Path, prefer: str = "cli"):
        self.path = Path(home) / "usage.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prefer = prefer
        self._cli = CLIBackend(self.path.parent)
        self._api: APIBackend | None = None
        self._api_broken = ""
        self._throttled_until = 0.0
        self._advisor_turn = 0
        # Optional and entirely absent unless configured — an unset compat
        # backend must behave exactly like one that was never written.
        try:
            self._compat: CompatBackend | None = CompatBackend()
        except Exception:
            self._compat = None

    def _api_backend(self) -> APIBackend | None:
        if self._api is None and not self._api_broken:
            try:
                self._api = APIBackend()
            except Exception as e:      # no key configured, SDK missing, etc.
                self._api_broken = str(e)
        return self._api

    def spend_today(self) -> float:
        """Real dollars only. Notional subscription cost is recorded but never
        counted against the cap — it was never charged."""
        if not self.path.exists():
            return 0.0
        today = time.strftime("%Y-%m-%d")
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                # a torn final line must not break reporting
            if r.get("billed") and r.get("ts", "").startswith(today):
                total += r.get("usd", 0.0)
        return total

    def complete(self, seat: str, system: str, user: str, *,
                 experiment_id: int | None = None) -> str:
        # A seat whose configured model contains "/" is a third-party model id
        # (e.g. nvidia/nemotron-3-ultra-550b-a55b) and is served by compat
        # regardless of which seat it is. This is how a rival can be a non-
        # Anthropic model: measured on s6e9, Nemotron Ultra wrote a valid,
        # contract-compliant baseline scoring CV 0.94184 — above the board
        # median — so the earlier blanket ban on third-party rivals was wrong.
        # Opus vs Sonnet is one lab's priors twice; Opus vs Nemotron is the
        # genuinely different rivalry the writeup's phase 2 depended on.
        alias = SEATS.get(seat, ("", ""))[0]
        if "/" in alias and self._compat is not None:
            last = None
            # 503 "Service temporarily overloaded" is common on the free tier
            # and transient — 1 of 2 attempts failed in testing. A rival seat
            # losing its turn to a blip is worth two retries to avoid.
            for attempt in range(3):
                try:
                    t0 = time.perf_counter()
                    text, usage = self._compat.complete(system, user, seat,
                                                        model=alias)
                    self._record(seat, "compat", usage,
                                 time.perf_counter() - t0, experiment_id)
                    return text
                except Exception as e:
                    last = e
                    if attempt < 2:
                        time.sleep(15 * (attempt + 1))
            # A third-party seat that cannot answer must NOT kill the turn.
            # Measured: one debugger call took 2071s (34 min) on a degraded
            # free endpoint, against 2.9s when healthy. Rivals have no Claude
            # equivalent so they still raise; the cheap seats fall through to
            # the subscription, which is free too and never this slow.
            if seat in ("debugger", "reporter"):
                print(f"  [compat] {alias} unavailable ({str(last)[:60]}) — "
                      f"using Claude for {seat}", flush=True)
            else:
                raise RuntimeError(f"{alias} unavailable after 3 tries: {last}")

        order = [self._cli, self._api_backend()] if self.prefer == "cli" \
            else [self._api_backend(), self._cli]
        # A recent throttle means skip the CLI entirely rather than paying the
        # latency of finding out again on every call for the next 10 minutes.
        throttled = time.time() < self._throttled_until
        if throttled and self.prefer == "cli":
            order = [self._api_backend(), self._cli]
        # Only while throttled does a low-stakes seat get pushed to a third
        # party. When the subscription is healthy there is no reason to send
        # work anywhere else — it is already free and it is already better.
        if throttled and self._compat is not None and seat in COMPAT_SEATS:
            order = [self._compat] + order

        last = None
        for backend in [b for b in order if b is not None]:
            if backend.name == "api" and self.spend_today() >= DAILY_CAP_USD:
                last = RuntimeError(
                    f"API daily cap reached (${self.spend_today():.2f} of "
                    f"${DAILY_CAP_USD:.2f}) and the subscription is throttled. "
                    f"Raise SIS_DAILY_CAP or wait.")
                continue
            t0 = time.perf_counter()
            try:
                text, usage = backend.complete(system, user, seat)
            except Throttled as e:
                # Only the SUBSCRIPTION has a throttle window worth remembering.
                # A third-party 429/503 says nothing about our Claude quota, and
                # letting it set this flag would push ten minutes of work onto
                # the billed API for no reason.
                if backend.name == "cli":
                    self._throttled_until = time.time() + 600
                last = e
                continue
            except Exception as e:
                last = e
                continue
            self._record(seat, backend.name, usage, time.perf_counter() - t0,
                         experiment_id)
            return text
        raise RuntimeError(f"no backend could answer: {last}")

    def advise(self, system: str, user: str) -> str:
        """Ask an outside model for ideas. Returns '' if none answers.

        This is the writeup's phase 4 — a swarm of different architectures
        contributing *hypotheses*, not code. Measured on the real s6e9 brief,
        Kimi K3 named specific column crosses and correctly reasoned that with
        only 14 raw features the ceiling is set by inputs rather than algorithm
        choice. That is worth 169 seconds once per turn in an eight-hour run.

        **Never blocking, by construction.** The free tier answered 1 of 3
        concurrent calls in testing (one 504, one 503), so every failure path
        returns the empty string and the turn proceeds without advice. An
        advisor that can stall a campaign is worse than no advisor.

        Models rotate per call, which is the point: the diversity comes from
        asking *different* priors on different turns, not from asking one model
        repeatedly.
        """
        models = [m.strip() for m in os.getenv("SIS_ADVISOR_MODELS", "").split(",")
                  if m.strip()]
        if not models or self._compat is None:
            return ""
        start = self._advisor_turn % len(models)
        self._advisor_turn += 1
        for i in range(len(models)):
            model = models[(start + i) % len(models)]
            t0 = time.perf_counter()
            try:
                text, usage = self._compat.complete(system, user, "advisor",
                                                    model=model)
            except Exception:
                continue                  # 503/504/timeout: just try the next one
            self._record("advisor", "compat", usage, time.perf_counter() - t0, None)
            return text
        return ""

    def _record(self, seat: str, backend: str, usage: dict, secs: float,
                experiment_id: int | None) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "seat": seat,
               "backend": backend, "latency_s": round(secs, 1),
               "experiment_id": experiment_id, **usage}
        with _LOCK, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def summary(self) -> str:
        if not self.path.exists():
            return "no model calls yet"
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            return "no model calls yet"
        billed = sum(r.get("usd", 0) for r in rows if r.get("billed"))
        notional = sum(r.get("usd", 0) for r in rows if not r.get("billed"))
        out = [f"{len(rows)} calls — ${billed:.2f} actually billed (API), "
               f"${notional:.2f} notional (subscription, not charged)",
               f"today: ${self.spend_today():.2f} of ${DAILY_CAP_USD:.2f} cap", "",
               f"{'seat':<12} {'backend':<8} {'calls':>5} {'usd':>9}  model"]
        agg: dict[tuple, dict] = {}
        for r in rows:
            k = (r.get("seat", "?"), r.get("backend", "?"))
            a = agg.setdefault(k, {"n": 0, "usd": 0.0, "model": r.get("model", "?")})
            a["n"] += 1
            a["usd"] += r.get("usd", 0.0)
        for (seat, backend), a in sorted(agg.items(), key=lambda kv: -kv[1]["usd"]):
            out.append(f"{seat:<12} {backend:<8} {a['n']:>5} {a['usd']:>9.4f}  {a['model']}")
        return "\n".join(out)


def demo() -> None:
    """Self-check the routing and accounting logic with fake backends — the
    real ones cost money or quota, and neither belongs in a unit check."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        llm = LLM(Path(d))

        class FakeCLI:
            name = "cli"
            calls = 0

            def complete(self, s, u, seat):
                FakeCLI.calls += 1
                return "from-cli", {"input": 1, "output": 1, "cache_read": 0,
                                    "cache_write": 0, "usd": 0.5, "billed": False,
                                    "model": "sonnet"}

        class FakeThrottled:
            name = "cli"

            def complete(self, s, u, seat):
                raise Throttled("usage limit reached")

        class FakeAPI:
            name = "api"

            def complete(self, s, u, seat):
                return "from-api", {"input": 1, "output": 1, "cache_read": 0,
                                    "cache_write": 0, "usd": 0.25, "billed": True,
                                    "model": "claude-sonnet-5"}

        llm._cli, llm._api = FakeCLI(), FakeAPI()
        assert llm.complete("rival_a", "sys", "hi") == "from-cli"

        # Subscription spend is notional and must NOT count against the cap.
        assert llm.spend_today() == 0.0, llm.spend_today()

        # A throttled subscription falls through to the API rather than failing.
        llm._cli = FakeThrottled()
        assert llm.complete("rival_a", "sys", "hi") == "from-api"
        assert abs(llm.spend_today() - 0.25) < 1e-9, llm.spend_today()

        # Both dollar figures are reported, and kept distinct.
        s = llm.summary()
        assert "actually billed" in s and "notional" in s, s

        # Once throttled, the CLI is skipped for a while instead of being
        # retried on every single call.
        assert llm._throttled_until > time.time()

        # A third-party endpoint takes over a low-stakes seat WHILE THROTTLED...
        class FakeCompat:
            name = "compat"

            def complete(self, s, u, seat):
                return "from-compat", {"input": 1, "output": 1, "cache_read": 0,
                                       "cache_write": 0, "usd": 0.0,
                                       "billed": False, "model": "compat:x"}

        llm._compat = FakeCompat()
        assert llm.complete("debugger", "sys", "hi") == "from-compat"
        # ...but NEVER takes a rival seat. The proposals are the product.
        assert llm.complete("rival_a", "sys", "hi") == "from-api"

        # And when the subscription is healthy, nothing is sent to a third
        # party at all — it is already free and already better.
        llm._throttled_until = 0.0
        llm._cli = FakeCLI()
        assert llm.complete("debugger", "sys", "hi") == "from-cli"

    print("llm.py: 8 checks passed")


if __name__ == "__main__":
    demo()
