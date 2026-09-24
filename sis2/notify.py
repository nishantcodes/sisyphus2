"""Telegram: push what the agent is doing, pull back what you think about it.

The 1st-place writeup's most reproducible claim is that a one-line human hint —
"have you studied the missing values?" — redirected hours of agent work
productively. That input has no substitute inside the architecture, so it needs
a channel that reaches a phone rather than a terminal the operator is not
sitting at during a 4am campaign.

The receiving half matters as much as the sending half, and the design rule is
in `store.add_hint`: a reply becomes a **row that is consumed once**, never a
line appended to the system prompt. Put "try target encoding" in a system prompt
and the agent does target encoding for the rest of the month. Put it in a queue
and it gets one guaranteed turn, then has to earn its place on CV like every
other idea.

Everything here degrades to a no-op when the token is unset, so a run with no
Telegram configured behaves exactly like a run with it, minus the messages.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from . import store

API = "https://api.telegram.org/bot{token}/{method}"


def _cfg() -> tuple[str, str]:
    return os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")


def enabled() -> bool:
    return all(_cfg())


def _call(method: str, **params) -> dict:
    token, _ = _cfg()
    if not token:
        return {}
    url = API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception:
        # Notification is never load-bearing. A network blip must not take down
        # a campaign that is otherwise running fine.
        return {}


def send(text: str) -> bool:
    """Push a message. Telegram caps messages at 4096 chars, so this trims."""
    _, chat = _cfg()
    if not enabled():
        return False
    body = text if len(text) <= 3900 else text[:3900] + "\n… (truncated)"
    return bool(_call("sendMessage", chat_id=chat, text=body,
                      disable_web_page_preview="true").get("ok"))


def poll_hints(conn, home: Path) -> list[str]:
    """Fetch new replies and file each as a hint row. Returns what it took in.

    The update offset is persisted so a restart does not replay every message
    ever sent — which would re-queue hints the campaign already acted on.
    """
    if not enabled():
        return []
    _, chat = _cfg()
    offset_file = Path(home) / ".tg_offset"
    offset = 0
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip() or 0)
        except ValueError:
            offset = 0

    res = _call("getUpdates", offset=offset + 1, timeout=0)
    if not res.get("ok"):
        return []
    taken, last = [], offset
    for upd in res.get("result", []):
        last = max(last, upd.get("update_id", 0))
        msg = upd.get("message") or upd.get("channel_post") or {}
        text = (msg.get("text") or "").strip()
        # Only accept messages from the configured chat: an open bot can be
        # messaged by anyone, and anyone should not be able to steer the run.
        if not text or str(msg.get("chat", {}).get("id", "")) != str(chat):
            continue
        if text.startswith("/"):
            continue                       # commands are handled by the CLI, not here
        store.add_hint(conn, text)
        taken.append(text)
    if last != offset:
        offset_file.write_text(str(last))
    return taken


def campaign_summary(conn, llm=None) -> str:
    """The message worth waking someone up for: where we are, what is next."""
    best = store.champion(conn)
    lb = store.best_lb(conn)
    n_done = conn.execute("SELECT COUNT(*) c FROM experiments "
                          "WHERE status='done'").fetchone()["c"]
    n_fail = conn.execute("SELECT COUNT(*) c FROM experiments "
                          "WHERE status='failed'").fetchone()["c"]
    lines = [f"sis2 — {n_done} done, {n_fail} failed"]
    if best is not None:
        lines.append(f"best CV {best['cv']:.5f} +/- {best['cv_std']:.5f} "
                     f"(#{best['id']} {best['family']}: {best['change']})")
    lines.append(f"best LB {lb:.5f}" if lb is not None else "nothing submitted yet")
    lines.append(store.cv_lb_trust(conn))
    recent = store.history(conn, limit=5)
    if recent:
        lines.append("\nlast 5:")
        for r in recent:
            mark = r["failure"] or r["verdict"] or "?"
            cv = f"{r['cv']:.5f}" if r["cv"] is not None else "  —  "
            lines.append(f"  #{r['id']:<4} {mark:<13} {cv}  {r['change'][:44]}")
    if llm is not None:
        lines.append("\n" + llm.summary().splitlines()[0])
    lines.append("\nReply with a hint and it goes into the queue for the next turn.")
    return "\n".join(lines)


def demo() -> None:
    """Self-check what does not need the network: hint intake and the summary.

    Sending is exercised by `sis2 ping` against the real bot, not here.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        conn = store.open_db(Path(d))
        store.set_meta(conn, higher_is_better="1", metric="auc")

        # With no token configured everything must be a quiet no-op, not a crash.
        old = os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        try:
            assert enabled() is False
            assert send("hello") is False
            assert poll_hints(conn, Path(d)) == []
        finally:
            if old:
                os.environ["TELEGRAM_BOT_TOKEN"] = old

        eid = store.queue(conn, family="xgb", hypothesis="h", change="baseline", code="x")
        store.finish(conn, eid, {"cv": 0.97, "cv_std": 0.001,
                                 "oof_path": "o", "pred_path": "p"})
        s = campaign_summary(conn)
        assert "best CV 0.97000" in s and "nothing submitted yet" in s, s

        store.add_hint(conn, "study the missing values")
        assert store.pending_hint(conn)["text"] == "study the missing values"

    print("notify.py: 6 checks passed")


if __name__ == "__main__":
    demo()
