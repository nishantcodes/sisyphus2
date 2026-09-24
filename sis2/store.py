"""The ledger: what was tried, what it scored, and what we concluded.

One SQLite file per competition. Four tables and a directory of markdown notes.

The design decisions that matter, all carried from v1 because they were the
parts that worked:

* `parent_id` on every experiment makes the history a **tree**, not a list.
  Only a tree lets you say "I changed X and the score moved Y" — without the
  parent link you have a pile of configs and no causality.
* **Three verdicts**, not two: `better` / `inconclusive` / `worse`. A gain
  inside the fold spread is not evidence. Skipping this is how an agent learns
  superstitions and builds a strategy on a 0.001 fluke.
* **A crash is not a refutation.** Failures get a taxonomy (`bug` / `timeout` /
  `oom`) and never produce a `worse` verdict — a KeyError says nothing about
  whether the hypothesis was good.
* **Human hints enter as rows**, never as prompt text. Appending "try target
  encoding" to a system prompt makes the agent do target encoding *forever*.
  A queued row gets a guaranteed slot and then competes on its score like
  anything else. This is the whole anti-over-correction mechanism.

The one structural change from v1 is `family` and the champion query. v1 ran a
UCT search over a single tree and converged on the public consensus. v2 keeps
one champion **per model family** and spends its time deepening each champion
rather than breadth-searching configs — which is what the 1st-place writeup
describes doing, and what a diverse-ensemble search does not do.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER,
  family TEXT NOT NULL DEFAULT 'other',
  hypothesis TEXT NOT NULL DEFAULT '',
  change TEXT NOT NULL DEFAULT '',
  code TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'queued',   -- queued|running|done|failed
  source TEXT NOT NULL DEFAULT 'agent',    -- agent|user|analyst
  author TEXT NOT NULL DEFAULT '',         -- which rival proposed it
  cv REAL, cv_std REAL, verdict TEXT,
  oof_path TEXT DEFAULT '', pred_path TEXT DEFAULT '',
  submission_path TEXT DEFAULT '', handle TEXT DEFAULT '',
  failure TEXT DEFAULT '', error TEXT DEFAULT '',
  runtime_s REAL DEFAULT 0, created_at REAL, finished_at REAL
);
CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id INTEGER, cv REAL, lb REAL, note TEXT DEFAULT '', ts REAL
);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE, title TEXT, family TEXT DEFAULT '',
  experiment_id INTEGER, ts REAL
);
-- Human hints. Stored as rows and CONSUMED once, never appended to the system
-- prompt: a hint in the system prompt makes the agent chase that idea forever,
-- which is the failure this whole design exists to avoid. A row gets exactly
-- one guaranteed turn, then competes on CV like anything else.
CREATE TABLE IF NOT EXISTS hints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL, consumed_at REAL, experiment_id INTEGER, ts REAL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS ix_exp_family ON experiments(family, status);
"""


def open_db(home: Path) -> sqlite3.Connection:
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "ledger.db", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Columns added after ledgers already existed. ALTER is idempotent here
    # only because we check first; an old ledger must keep opening.
    have = {r["name"] for r in conn.execute("PRAGMA table_info(experiments)")}
    for col, decl in (("delta", "REAL"), ("delta_t", "REAL")):
        if col not in have:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {decl}")
    conn.commit()
    return conn


def set_meta(conn, **kv) -> None:
    conn.executemany("INSERT INTO meta(k,v) VALUES(?,?) "
                     "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                     [(k, str(v)) for k, v in kv.items()])
    conn.commit()


def meta(conn, key: str, default: str = "") -> str:
    r = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return r["v"] if r else default


def higher_is_better(conn) -> bool:
    return meta(conn, "higher_is_better", "1") == "1"


# ------------------------------------------------------------- experiments

def queue(conn, *, family: str, hypothesis: str, change: str, code: str,
          parent_id: int | None = None, source: str = "agent",
          author: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO experiments(parent_id,family,hypothesis,change,code,source,"
        "author,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (parent_id, family, hypothesis, change, code, source, author, time.time()))
    conn.commit()
    return cur.lastrowid


def next_queued(conn) -> sqlite3.Row | None:
    """User hints jump the queue. That is their entire privilege — they get a
    guaranteed slot, then they are judged on CV like every other row."""
    return conn.execute(
        "SELECT * FROM experiments WHERE status='queued' "
        "ORDER BY (source='user') DESC, id ASC LIMIT 1").fetchone()


def mark_running(conn, exp_id: int) -> None:
    conn.execute("UPDATE experiments SET status='running' WHERE id=?", (exp_id,))
    conn.commit()


def verdict_for(conn, exp_id: int, cv: float, cv_std: float,
                judge=None, oof_path: str = "") -> str:
    """Compare against the parent, with the fold spread as the noise floor.

    A run with no parent is a `baseline`. A gain smaller than the spread is
    `inconclusive` — not `better`. This single rule is the difference between
    an agent that learns and one that accumulates superstitions.
    """
    row = conn.execute("SELECT cv, cv_std FROM experiments WHERE id=?",
                       (exp_id,)).fetchone()
    parent = conn.execute(
        "SELECT cv, cv_std, oof_path FROM experiments e WHERE e.id="
        "(SELECT parent_id FROM experiments WHERE id=?)", (exp_id,)).fetchone()
    if parent is None or parent["cv"] is None:
        return "baseline"
    # The paired test, when the caller can supply one. It compares the two
    # OOF arrays fold by fold on the shared folds, which cancels fold
    # difficulty — see metrics.paired for the measurement that motivated it.
    if judge is not None and oof_path and parent["oof_path"]:
        try:
            out = judge(parent["oof_path"], oof_path)
        except Exception:
            out = None
        if out is not None:
            v, delta, t = out
            conn.execute("UPDATE experiments SET delta=?, delta_t=? WHERE id=?",
                         (delta, t, exp_id))
            return v
    # Noise floor is the larger of the two spreads: comparing a tight run to a
    # noisy parent is only as trustworthy as the noisier of the pair.
    floor = max(cv_std or 0.0, parent["cv_std"] or 0.0)
    delta = cv - parent["cv"]
    if not higher_is_better(conn):
        delta = -delta
    if delta > floor:
        return "better"
    if delta < -floor:
        return "worse"
    return "inconclusive"


def finish(conn, exp_id: int, result: dict, judge=None) -> str:
    """Record a completed run and return its verdict ('' when it failed)."""
    if result.get("failure"):
        conn.execute(
            "UPDATE experiments SET status='failed', failure=?, error=?, "
            "runtime_s=?, finished_at=? WHERE id=?",
            (result["failure"], (result.get("error") or "")[:4000],
             result.get("runtime_s", 0), time.time(), exp_id))
        conn.commit()
        return ""
    conn.execute(
        "UPDATE experiments SET status='done', cv=?, cv_std=?, oof_path=?, "
        "pred_path=?, submission_path=?, runtime_s=?, finished_at=? WHERE id=?",
        (result["cv"], result["cv_std"], result.get("oof_path", ""),
         result.get("pred_path", ""), result.get("submission_path", ""),
         result.get("runtime_s", 0), time.time(), exp_id))
    v = verdict_for(conn, exp_id, result["cv"], result["cv_std"], judge,
                    result.get("oof_path", ""))
    conn.execute("UPDATE experiments SET verdict=? WHERE id=?", (v, exp_id))
    conn.commit()
    return v


def champion(conn, family: str = "") -> sqlite3.Row | None:
    """The best scored experiment, overall or within one family.

    This is what the next experiment extends and what the analyst studies. The
    ordering respects the metric direction so the same code works for AUC and
    RMSE alike.
    """
    # Blends are excluded unless asked for by name. A blend is not a model
    # anyone can extend, and the analyst studying one would diagnose "where
    # the average of everything is wrong" — which no single proposal can fix.
    order = "DESC" if higher_is_better(conn) else "ASC"
    where = "AND family=?" if family else "AND family != 'blend'"
    args = (family,) if family else ()
    return conn.execute(
        f"SELECT * FROM experiments WHERE status='done' AND cv IS NOT NULL "
        f"AND oof_path != '' {where} ORDER BY cv {order} LIMIT 1", args).fetchone()


def best_candidate(conn) -> sqlite3.Row | None:
    """The best thing we could SUBMIT: a single model or a portfolio blend.

    Separate from champion() on purpose. The champion is what the next
    proposal builds on; the candidate is what goes to the board. For a medal
    they are usually different objects — the board wants the ensemble, the
    research loop wants the strongest single model to study.
    """
    order = "DESC" if higher_is_better(conn) else "ASC"
    return conn.execute(
        f"SELECT * FROM experiments WHERE status='done' AND cv IS NOT NULL "
        f"AND submission_path != '' ORDER BY cv {order} LIMIT 1").fetchone()


def families(conn) -> list[str]:
    return [r["family"] for r in conn.execute(
        "SELECT DISTINCT family FROM experiments WHERE status='done'")]


def in_flight(conn) -> list[sqlite3.Row]:
    """What is running right now. Goes in the prompt so parallel rivals don't
    independently propose the same idea three times because none have landed."""
    return conn.execute(
        "SELECT id,family,change,author FROM experiments "
        "WHERE status IN ('queued','running') ORDER BY id").fetchall()


def history(conn, family: str = "", limit: int = 40) -> list[sqlite3.Row]:
    where = "AND family=?" if family else ""
    args = ((family, limit) if family else (limit,))
    return conn.execute(
        f"SELECT id,parent_id,family,change,cv,cv_std,verdict,author,failure,"
        f"delta,delta_t,runtime_s "
        f"FROM experiments WHERE status IN ('done','failed') {where} "
        f"ORDER BY id DESC LIMIT ?", args).fetchall()


# ------------------------------------------------------------- submissions

def record_submission(conn, exp_id: int, cv: float, lb: float | None,
                      note: str = "") -> None:
    conn.execute("INSERT INTO submissions(experiment_id,cv,lb,note,ts) "
                 "VALUES(?,?,?,?,?)", (exp_id, cv, lb, note, time.time()))
    conn.commit()


def best_lb(conn) -> float | None:
    order = "DESC" if higher_is_better(conn) else "ASC"
    r = conn.execute(f"SELECT lb FROM submissions WHERE lb IS NOT NULL "
                     f"ORDER BY lb {order} LIMIT 1").fetchone()
    return r["lb"] if r else None


def cv_lb_trust(conn) -> str:
    """Whether our CV predicts the board — the only thing a spent submission
    actually buys, and the number that decides which of the two final slots to
    trust when they disagree."""
    rows = conn.execute("SELECT cv, lb FROM submissions WHERE lb IS NOT NULL "
                        "ORDER BY ts").fetchall()
    if len(rows) < 2:
        return f"{len(rows)} scored submission(s) — not enough to judge CV/LB trust yet"
    gaps = [r["cv"] - r["lb"] for r in rows]
    mean = sum(gaps) / len(gaps)
    spread = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    # Rank agreement matters more than the absolute gap: a constant offset is
    # harmless, an inconsistent one means CV cannot rank candidates.
    pairs = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:]]
    agree = sum(1 for a, b in pairs
                if (a["cv"] > b["cv"]) == (a["lb"] > b["lb"]))
    return (f"CV/LB over {len(rows)} submissions: mean gap {mean:+.5f} "
            f"(sd {spread:.5f}), rank agreement {agree}/{len(pairs)}")


# ------------------------------------------------------------------- hints

def add_hint(conn, text: str) -> int:
    cur = conn.execute("INSERT INTO hints(text, ts) VALUES(?,?)",
                       (text.strip(), time.time()))
    conn.commit()
    return cur.lastrowid


def pending_hint(conn) -> sqlite3.Row | None:
    """The oldest unconsumed hint, or None. Read once per turn."""
    return conn.execute("SELECT * FROM hints WHERE consumed_at IS NULL "
                        "ORDER BY id ASC LIMIT 1").fetchone()


def consume_hint(conn, hint_id: int, experiment_id: int | None = None) -> None:
    """Mark a hint spent. This is the anti-over-correction mechanism: the hint
    influenced exactly one proposal and will never be seen again."""
    conn.execute("UPDATE hints SET consumed_at=?, experiment_id=? WHERE id=?",
                 (time.time(), experiment_id, hint_id))
    conn.commit()


# ------------------------------------------------------------------- notes

def add_note(conn, path: Path, title: str, family: str = "",
             experiment_id: int | None = None) -> None:
    conn.execute("INSERT OR REPLACE INTO notes(path,title,family,experiment_id,ts)"
                 " VALUES(?,?,?,?,?)",
                 (str(path), title, family, experiment_id, time.time()))
    conn.commit()


def read_notes(notes_dir: Path, limit: int = 60) -> str:
    """Every discovery note, newest last, concatenated.

    This is the memory that the 1st-place writeup calls "gold" — 50 discovery
    files a fresh model can be shown so it neither repeats itself nor re-derives
    what a previous campaign already established. Cheap to assemble, and it is
    the reason a 4am campaign can build on a 4pm one.
    """
    if not notes_dir.exists():
        return "(no discovery notes yet)"
    files = sorted(notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)[-limit:]
    if not files:
        return "(no discovery notes yet)"
    return "\n\n".join(f"--- {f.name} ---\n{f.read_text(encoding='utf-8').strip()}"
                       for f in files)


def demo() -> None:
    """Self-check on the rules that make this a learning system rather than a
    spreadsheet: three verdicts, noise-floor comparison, crash-is-not-refutation,
    and user hints jumping the queue."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        conn = open_db(Path(d))
        set_meta(conn, higher_is_better="1", metric="auc")

        base = queue(conn, family="xgb", hypothesis="baseline", change="baseline", code="x")
        assert finish(conn, base, {"cv": 0.960, "cv_std": 0.001,
                                   "oof_path": "o", "pred_path": "p"}) == "baseline"

        # A gain larger than the fold spread is real.
        big = queue(conn, family="xgb", hypothesis="h", change="target encoding",
                    code="x", parent_id=base)
        assert finish(conn, big, {"cv": 0.970, "cv_std": 0.001,
                                  "oof_path": "o", "pred_path": "p"}) == "better"

        # A gain INSIDE the spread is not. This is the rule that stops an agent
        # building a strategy on a fluke.
        tiny = queue(conn, family="xgb", hypothesis="h", change="+1 seed",
                     code="x", parent_id=base)
        assert finish(conn, tiny, {"cv": 0.9605, "cv_std": 0.002,
                                   "oof_path": "o", "pred_path": "p"}) == "inconclusive"

        worse = queue(conn, family="xgb", hypothesis="h", change="dropped feats",
                      code="x", parent_id=base)
        assert finish(conn, worse, {"cv": 0.940, "cv_std": 0.001,
                                    "oof_path": "o", "pred_path": "p"}) == "worse"

        # A crash yields no verdict at all — the idea stays open for a retry.
        crash = queue(conn, family="xgb", hypothesis="h", change="c", code="x",
                      parent_id=base)
        assert finish(conn, crash, {"failure": "bug", "error": "KeyError"}) == ""
        assert conn.execute("SELECT verdict FROM experiments WHERE id=?",
                            (crash,)).fetchone()["verdict"] is None

        assert champion(conn, "xgb")["id"] == big
        # Only rows with saved OOF can be champion — you cannot study what you
        # did not keep.
        assert champion(conn)["oof_path"] == "o"

        # A user hint queued last still comes out first.
        queue(conn, family="xgb", hypothesis="a", change="agent idea", code="x")
        hint = queue(conn, family="xgb", hypothesis="u", change="study the NaNs",
                     code="x", source="user")
        assert next_queued(conn)["id"] == hint

        # A steering hint is delivered exactly once, then never again — that is
        # what stops one suggestion dominating the rest of the campaign.
        h = add_hint(conn, "have you studied the missing values?")
        assert pending_hint(conn)["id"] == h
        consume_hint(conn, h, experiment_id=big)
        assert pending_hint(conn) is None

        print("store.py: 10 checks passed")


if __name__ == "__main__":
    demo()


def read_skills(skills_dir) -> str:
    """Procedural memory: how to act, as markdown playbooks.

    v1 had six of these and v2 shipped with none — the rebuild kept the
    experiment ledger and the discovery notes but silently dropped the
    strategy. Twelve s6e9 experiments were spent on feature engineering and
    tuning while `playground.md` sat unread in the old repo saying, in the
    first bullet, that neither is the lever in a synthetic competition.
    """
    from pathlib import Path as _P
    d = _P(skills_dir)
    if not d.exists():
        return ""
    parts = [f.read_text(encoding="utf-8").strip() for f in sorted(d.glob("*.md"))]
    return "\n\n".join(p for p in parts if p)
