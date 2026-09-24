"""Command line: init, run, status, submit, blend, ping.

The auto-submit rule lives here, and it is the one policy decision worth
stating out loud. Kaggle gives five submissions a day and **unused slots expire
forever** — over a month that is up to 140 discarded reads on the true test
distribution, which is the only external check on whether our CV means
anything. So the agent sends on its own, under a rule narrow enough to be
trustworthy:

    send when a new champion beats the best ALREADY-SUBMITTED CV by more than
    the fold spread, and slots remain today.

The "more than the fold spread" clause is what stops it burning five slots a
day on noise. It will not submit a run it cannot distinguish from the last one.
Every one of the six validations still runs before anything is sent, and
marking the two final submissions is still a human click — Kaggle exposes no
API for that, and saying otherwise would be wrong.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Read .env without a hard dependency on python-dotenv being importable."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _home(slug: str) -> Path:
    return ROOT / "work" / slug


def _campaign(slug: str):
    from . import agent, llm
    home = _home(slug)
    if not (home / "ledger.db").exists():
        sys.exit(f"no campaign at {home}. Run `sis2 init {slug} --target <col>` first.")
    prefer = os.getenv("SIS_BACKEND", "cli")
    return agent.Campaign(home, llm.LLM(home, prefer=prefer), slug=slug,
                          notes_dir=ROOT / "notes" / slug)


# ------------------------------------------------------------------ commands

def cmd_init(a) -> None:
    from . import kaggle, store
    home = _home(a.slug)
    home.mkdir(parents=True, exist_ok=True)
    data = home / "input"

    if a.local and not (data / "train.csv").exists():
        from . import local
        print(local.prepare(home, a.local, target=a.target, seed=a.seed))
        a.target = a.target or local.KNOWN.get(a.local, {}).get("target", "")
    elif not (data / "train.csv").exists():
        ok, msg = kaggle.download(a.slug, data)
        print(msg)
        if not ok:
            sys.exit(1)
    else:
        print(f"data already present in {data}")

    if not a.target:
        sys.exit("--target is required (it can be omitted only for a known --local dataset)")
    conn = store.open_db(home)
    store.set_meta(conn, slug=a.slug, metric=a.metric, target=a.target,
                   higher_is_better="1" if __import__(
                       "sis2.metrics", fromlist=["direction"]).direction(a.metric) else "0",
                   families=a.families)
    print(kaggle.make_folds(data, n_folds=a.folds, seed=a.seed, target=a.target))

    board = {} if a.local else kaggle.leaderboard(a.slug, home)
    if board.get("known"):
        line = (f"Leaderboard: {board['teams']} teams, top {board['top']:.5f}, "
                f"top-10% {board['p10']:.5f}, median {board['median']:.5f}")
        store.set_meta(conn, board=line)
        print(line)

    print("\n" + kaggle.profile(data, a.target))
    print(f"\nready. `sis2 run {a.slug}` to start.")


def cmd_run(a) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from . import blend, notify, store
    from . import llm as llm_mod
    camp = _campaign(a.slug)
    seats = llm_mod.rivals()   # rival_c joins automatically when configured
    started = time.time()
    if notify.enabled() and not a.quiet:
        notify.send(f"sis2 campaign starting on {a.slug} — {a.turns} turns")

    # A run killed mid-experiment leaves a row 'running' forever, and it then
    # sits in every prompt's "running right now — do not duplicate" list.
    camp.conn.execute("UPDATE experiments SET status='queued' WHERE status='running'")
    camp.conn.commit()

    def after(line: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)
        try:
            pl = blend.refresh_portfolio(camp.conn, camp.home, camp.data)
            if pl:
                print(f"  {pl}", flush=True)
        except Exception as e:           # the portfolio must never stop a campaign
            print(f"  portfolio failed: {str(e)[:120]}", flush=True)
        if a.submit:
            sent, why = maybe_submit(camp, a.slug)
            if sent:
                print(f"  -> {why}", flush=True)
                if notify.enabled():
                    notify.send(f"submitted on {a.slug}\n{why}")

    if a.seed and not camp.conn.execute(
            "SELECT 1 FROM experiments WHERE source='recipe'").fetchone():
        for line in camp.seed():
            after(line)

    # Anything left queued by an earlier run goes first.
    while (row := store.next_queued(camp.conn)) is not None:
        after(camp.run_one(row["id"]))

    def propose(seat: str, i: int):
        c = camp.clone()                 # SQLite connections are per-thread
        try:
            return c.propose_and_queue(seat, i)
        finally:
            c.conn.close()

    # Pipelined: proposal i+1 is written while experiment i trains. It sees
    # experiment i in its "running right now" list rather than its result —
    # a one-turn lag bought for roughly halving wall clock per turn.
    pool = ThreadPoolExecutor(max_workers=1)
    ahead = None
    for i in range(a.turns):
        if a.hours and (time.time() - started) / 3600 >= a.hours:
            print(f"[wall clock] {a.hours}h budget reached")
            break
        # Pull steering replies before proposing, so a hint sent two minutes ago
        # lands on this turn rather than the next one.
        for h in notify.poll_hints(camp.conn, camp.home):
            print(f"[hint] {h}")
        seat = seats[i % len(seats)]        # the rivalry: alternate the proposer
        if not a.pipeline:
            after(camp.turn(seat, turn_no=i))
            continue
        if ahead is None:
            ahead = pool.submit(propose, seat, i)
        eid, msg = ahead.result()
        nxt = i + 1
        ahead = (pool.submit(propose, seats[nxt % len(seats)], nxt)
                 if nxt < a.turns else None)
        if eid is None:
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
            continue
        after(camp.run_one(eid, seat))
    if ahead is not None:                   # budget hit with a proposal in hand
        eid, _ = ahead.result()
        print(f"(exp #{eid} left queued for the next run)" if eid else "")
    pool.shutdown()

    if a.blend:
        ok, msg = blend.build(camp.conn, camp.home, camp.data, record=False)
        print("\n" + msg)

    summary = notify.campaign_summary(camp.conn, camp.llm)
    print("\n" + summary)
    if notify.enabled() and not a.quiet:
        notify.send(summary)


def maybe_submit(camp, slug: str) -> tuple[bool, str]:
    """The auto-submit rule. Returns (sent, explanation)."""
    from . import kaggle, store
    # The best SUBMITTABLE thing, blends included — the board wants the
    # ensemble even though the research loop studies single models.
    champ = store.best_candidate(camp.conn)
    if champ is None or not champ["submission_path"]:
        return False, "no candidate with a submission file"
    already = camp.conn.execute(
        "SELECT 1 FROM submissions WHERE experiment_id=?", (champ["id"],)).fetchone()
    if already:
        return False, "champion already submitted"

    higher = store.higher_is_better(camp.conn)
    prev = camp.conn.execute(
        "SELECT e.* FROM submissions s JOIN experiments e ON e.id=s.experiment_id "
        f"ORDER BY e.cv {'DESC' if higher else 'ASC'} LIMIT 1").fetchone()
    if prev is not None:
        gain = (champ["cv"] - prev["cv"]) if higher else (prev["cv"] - champ["cv"])
        # Paired test against the best thing already on the board: a slot is
        # spent only on a candidate that beats it consistently fold by fold.
        judged = None
        if prev["oof_path"] and champ["oof_path"]:
            try:
                judged = camp._judge(prev["oof_path"], champ["oof_path"])
            except Exception:
                judged = None
        if judged is not None:
            v, delta, t = judged
            if v != "better":
                return False, (f"candidate #{champ['id']} vs submitted #{prev['id']}: "
                               f"{delta:+.5f} (paired t {t:+.1f}) — not a clear "
                               f"enough gain to spend a slot")
        else:
            floor = champ["cv_std"] or 0.0
            if gain <= floor:
                return False, (f"candidate beats the best submitted CV by {gain:+.5f}, "
                               f"inside the {floor:.5f} fold spread — not worth a slot")

    from . import local
    if local.is_local(camp.home):
        cap = int(os.getenv("SIS_LOCAL_DAILY", "5"))
        used = local.submissions_today(camp.home)
        if used >= cap:
            return False, f"{used}/{cap} daily local submissions already used"
        return local.submit(camp.conn, camp.home, dict(champ), camp.data)

    used = kaggle.submissions_today(slug)
    if used >= 5:
        return False, f"{used}/5 daily submissions already used"

    return kaggle.submit(camp.conn, slug, dict(champ), camp.data)


def cmd_fanout(a) -> None:
    """Every rival proposes at once, and all of them run on Kaggle in parallel.

    The normal loop is strictly sequential: one rival, one experiment, one local
    training run, repeat. That leaves Kaggle idle and caps throughput at whatever
    the laptop does in an hour. Here each rival proposes simultaneously, every
    script is smoke-tested locally at 5% (cheap, catches the boring failures
    before they cost a kernel slot), and the survivors are pushed to Kaggle
    together and polled as a batch.

    CPU kernels by default: GBDTs gain nothing from a GPU, CPU does not consume
    the weekly GPU quota, and requesting a GPU is what lands us on a P100 that
    current torch cannot use.
    """
    from . import kaggle, llm as llm_mod, run, store
    camp = _campaign(a.slug)
    seats = llm_mod.rivals()[:a.n] if a.n else llm_mod.rivals()
    ds = camp._folds_dataset()
    if not ds:
        sys.exit("cannot publish the shared folds dataset — a kernel without "
                 "folds.npy would invent its own split and its score would not "
                 "be comparable to anything")
    print(f"folds dataset: {ds}\nasking {len(seats)} rivals in parallel: {seats}")

    jobs = []
    for i, seat in enumerate(seats):
        family = camp.pick_family()
        hint = store.pending_hint(camp.conn)
        try:
            prop = camp.propose(seat, family, hint["text"] if hint else "", "", i)
        except Exception as e:
            print(f"  [{seat}] no proposal: {str(e)[:90]}")
            continue
        if prop is None:
            print(f"  [{seat}] proposal did not parse")
            continue
        pid = prop["parent_id"]
        if pid is not None and not camp.conn.execute(
                "SELECT 1 FROM experiments WHERE id=?", (pid,)).fetchone():
            ch = store.champion(camp.conn)
            pid = ch["id"] if ch is not None else None
        eid = store.queue(camp.conn, family=prop["family"],
                          hypothesis=prop["hypothesis"], change=prop["change"],
                          code=prop["code"], parent_id=pid, author=seat,
                          source="user" if hint else "agent")
        if hint:
            store.consume_hint(camp.conn, hint["id"], eid)
        print(f"  [{seat}] exp #{eid} [{prop['family']}] {prop['change'][:52]}")

        wd = camp.home / f"exp{eid}"
        smoke = run.run_local(prop["code"], wd, frac=0.05, data_dir=camp.data,
                              timeout=900)
        if smoke.get("failure"):
            fixed = camp._repair(prop["code"], smoke)
            if fixed:
                smoke = run.run_local(fixed, wd, frac=0.05, data_dir=camp.data,
                                      timeout=900)
                if not smoke.get("failure"):
                    prop["code"] = fixed
                    camp.conn.execute("UPDATE experiments SET code=? WHERE id=?",
                                      (fixed, eid))
                    camp.conn.commit()
        if smoke.get("failure"):
            print(f"      smoke failed ({smoke['failure']}) — not spending a kernel")
            store.finish(camp.conn, eid, smoke)
            continue
        store.mark_running(camp.conn, eid)
        jobs.append({"exp_id": eid, "code": prop["code"], "workdir": str(wd)})

    if not jobs:
        sys.exit("nothing survived the smoke test")
    print(f"\npushing {len(jobs)} kernels...")
    pushed = kaggle.push_many(jobs, a.slug, gpu=a.gpu, dataset_sources=[ds])
    for j in pushed:
        print(f"  exp #{j['exp_id']}: {j.get('handle') or 'PUSH FAILED: ' + str(j.get('error'))[:70]}")
    print(f"\npolling every {a.poll}s (timeout {a.timeout}s)...")
    n_tr, n_te = camp._row_counts()
    for res in kaggle.poll_many(pushed, poll_every=a.poll, timeout=a.timeout):
        eid = res["exp_id"]
        if not res.get("failure") and res.get("oof_path"):
            bad = run._oof_agrees(res["oof_path"], camp._y(), camp.metric, res["cv"])
            if bad:
                res = {"failure": "bug", "error": bad}
        verdict = store.finish(camp.conn, eid, res, judge=camp._judge)
        if res.get("failure"):
            print(f"  exp #{eid} FAILED ({res['failure']}): {str(res.get('error'))[:80]}")
        else:
            print(f"  exp #{eid} {verdict} — CV {res['cv']:.6f} +/- {res['cv_std']:.6f}")
    print("\n" + __import__("sis2.notify", fromlist=["campaign_summary"]).campaign_summary(camp.conn))


def cmd_status(a) -> None:
    from . import llm, notify, store
    home = _home(a.slug)
    conn = store.open_db(home)
    print(notify.campaign_summary(conn, llm.LLM(home)))
    print("\n" + llm.LLM(home).summary())


def cmd_submit(a) -> None:
    from . import kaggle, store
    camp = _campaign(a.slug)
    exp = camp.conn.execute("SELECT * FROM experiments WHERE id=?",
                            (a.experiment,)).fetchone() if a.experiment \
        else store.champion(camp.conn)
    if exp is None:
        sys.exit("no such experiment")
    from . import local
    if local.is_local(camp.home):
        ok, msg = local.submit(camp.conn, camp.home, dict(exp), camp.data)
    else:
        ok, msg = kaggle.submit(camp.conn, a.slug, dict(exp), camp.data)
    print(msg)


def cmd_private(a) -> None:
    """Local competitions only: every submission's hidden private score."""
    from . import local
    print(local.private_report(_home(a.slug)))


def cmd_blend(a) -> None:
    from . import blend
    camp = _campaign(a.slug)
    ok, msg = blend.build(camp.conn, camp.home, camp.data, record=False)
    print(msg)
    print("\n" + (blend.refresh_portfolio(camp.conn, camp.home, camp.data)
                  or "portfolio: nothing to blend yet"))


def cmd_finals(a) -> None:
    """Recommend the two submissions to mark final.

    Kaggle scores two chosen submissions on the private board and everyone's
    instinct is to pick the top two public scores — which hedges nothing, since
    both lose to the same failure. The two slots exist to cover two different
    ways of being wrong: public LB is a small sample you have been selecting
    against all month, and local CV can be measuring the wrong thing entirely.
    There is no API for marking them; this prints the picks and you click.
    """
    from . import store
    camp = _campaign(a.slug)
    rows = camp.conn.execute(
        "SELECT s.experiment_id id, s.cv, s.lb, e.cv_std, e.change "
        "FROM submissions s JOIN experiments e ON e.id=s.experiment_id "
        "WHERE s.lb IS NOT NULL").fetchall()
    if not rows:
        sys.exit("no scored submissions yet")
    higher = store.higher_is_better(camp.conn)
    pick = max if higher else min
    best_of = {}
    for r in rows:
        p = best_of.get(r["id"])
        if p is None or ((r["lb"] > p["lb"]) if higher else (r["lb"] < p["lb"])):
            best_of[r["id"]] = dict(r)
    pool = list(best_of.values())

    def lcb(d):                      # penalise variance when trusting CV
        s = d["cv_std"] or 0.0
        return d["cv"] - s if higher else d["cv"] + s

    trust_cv = pick(pool, key=lcb)
    trust_lb = pick(pool, key=lambda d: d["lb"])
    if trust_cv["id"] == trust_lb["id"] and len(pool) > 1:
        rest = [d for d in pool if d["id"] != trust_cv["id"]]
        trust_lb = pick(rest, key=lambda d: d["lb"])

    print(f"Final-slot picks for {a.slug} ({len(pool)} scored submissions):")
    print(f"  slot 1  exp #{trust_cv['id']}  cv {trust_cv['cv']:.5f} "
          f"lb {trust_cv['lb']:.5f}   <- trusts local CV (best lower bound)")
    print(f"  slot 2  exp #{trust_lb['id']}  cv {trust_lb['cv']:.5f} "
          f"lb {trust_lb['lb']:.5f}   <- trusts the public board")
    print("  " + store.cv_lb_trust(camp.conn))
    if trust_cv["id"] != trust_lb["id"]:
        print("  The two disagree — that is exactly the case the second slot "
              "exists for. Marking both is strictly safer than either alone.")
    print(f"\nMark them at https://www.kaggle.com/competitions/"
          f"{__import__('sis2.kaggle', fromlist=['comp_slug']).comp_slug(a.slug)}"
          f"/submissions (browser only — Kaggle has no API for this).")


def cmd_hint(a) -> None:
    """Steer the campaign from the terminal.

    Same mechanism Telegram replies use — the text becomes a `hints` row that
    appears in exactly ONE proposal prompt and is then marked spent. That is
    the whole anti-over-correction design: a hint gets a guaranteed turn, then
    has to earn its place on CV like every other idea. Put it in the system
    prompt instead and the agent chases it for the rest of the month.

    Safe to run against a campaign that is mid-flight in another terminal —
    SQLite handles the concurrent write, and the loop picks the hint up on its
    next turn.
    """
    from . import store
    conn = store.open_db(_home(a.slug))
    text = " ".join(a.text).strip()
    if not text:
        pending = store.pending_hint(conn)
        print(f"queued hint: {pending['text']}" if pending else "no hint queued")
        return
    store.add_hint(conn, text)
    print(f"queued — the next proposal will address it, once:\n  {text}")


def cmd_ping(a) -> None:
    from . import notify
    if not notify.enabled():
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")
    print("sent" if notify.send("sis2 ping — steering channel is live") else "failed")


def cmd_check(a) -> None:
    """Run every module's self-check. One command to prove the install works."""
    import subprocess
    mods = ["metrics", "run", "store", "analyst", "kaggle", "llm", "agent",
            "blend", "notify", "local"]
    bad = 0
    for m in mods:
        r = subprocess.run([sys.executable, "-m", f"sis2.{m}"],
                           capture_output=True, text=True, cwd=ROOT)
        line = (r.stdout or r.stderr).strip().splitlines()
        print(f"  {'ok  ' if r.returncode == 0 else 'FAIL'} {line[-1] if line else m}")
        bad += r.returncode != 0
    sys.exit(bad)


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser(prog="sis2", description="autonomous Kaggle agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="download data, freeze folds, read the board")
    i.add_argument("slug")
    i.add_argument("--target", default="", help="name of the label column")
    i.add_argument("--local", default="",
                   help="build a LOCAL competition from a CSV path/URL or a "
                        "known name (bank, adult) instead of downloading from "
                        "Kaggle; test labels stay hidden, submissions are "
                        "scored on a public/private split")
    i.add_argument("--metric", default="auc")
    i.add_argument("--folds", type=int, default=10)
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--families", default="xgb,lgbm,catboost")
    i.set_defaults(fn=cmd_init)

    r = sub.add_parser("run", help="run the improvement loop")
    r.add_argument("slug")
    r.add_argument("--turns", type=int, default=20)
    r.add_argument("--hours", type=float, default=0, help="wall-clock budget")
    r.add_argument("--no-submit", dest="submit", action="store_false")
    # Blending is NOT part of a campaign. Rivals build individual models;
    # combining them is an operator decision taken on demand, because a
    # blend is not a hypothesis anyone can learn from and it pollutes the
    # rival ledger with rows nobody proposed. Use `sis2 blend` instead.
    r.add_argument("--blend", action="store_true",
                   help="also build an ensemble at the end (off by default)")
    r.add_argument("--quiet", action="store_true", help="no Telegram start/end message")
    r.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                   help="propose and execute strictly in series")
    r.add_argument("--no-seed", dest="seed", action="store_false",
                   help="skip the recipe baselines on a fresh campaign")
    r.set_defaults(fn=cmd_run, submit=True, blend=False, pipeline=True, seed=True)

    for name, fn, helptext in (("status", cmd_status, "where the campaign is"),
                               ("blend", cmd_blend, "build the ensemble"),
                               ("finals", cmd_finals, "which two to mark final"),
                               ("ping", cmd_ping, "test the Telegram channel"),
                               ("private", cmd_private, "local only: private scores")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("slug")
        s.set_defaults(fn=fn)

    s = sub.add_parser("submit", help="send a submission by hand")
    s.add_argument("slug")
    s.add_argument("--experiment", type=int, default=0)
    s.set_defaults(fn=cmd_submit)

    h = sub.add_parser("hint", help="steer the next proposal (one-shot)")
    h.add_argument("slug")
    h.add_argument("text", nargs="*", help="the hint; omit to show what is queued")
    h.set_defaults(fn=cmd_hint)

    f = sub.add_parser("fanout", help="all rivals propose at once, run on Kaggle in parallel")
    f.add_argument("slug")
    f.add_argument("--n", type=int, default=0, help="how many rivals (default: all)")
    f.add_argument("--gpu", action="store_true", help="request a GPU kernel (default CPU)")
    f.add_argument("--poll", type=int, default=45)
    f.add_argument("--timeout", type=int, default=5400)
    f.set_defaults(fn=cmd_fanout)

    c = sub.add_parser("check", help="run every self-check")
    c.set_defaults(fn=cmd_check)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
