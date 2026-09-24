"""Read what the field is doing — for ideas, never for predictions.

v1 mounted other people's published OOF arrays and stacked them, and landed
exactly on the public ceiling. The lesson was "do not borrow the field's
*outputs*", not "do not read the field". Public notebooks and discussions are
where the playground-specific facts surface — which original dataset the
generator used, which column is secretly discrete, what the leaderboard
probing has shown — and a rival that cannot see any of it re-derives in ten
turns what a notebook states in one cell.

So this seat reads notebooks and writes a discovery note per notebook:
techniques, features, reported CV/LB, and what is actually new. The note lands
in `notes/<slug>/`, which every rival prompt already includes. No array from a
notebook ever enters the blend pool.

Sources, in order of preference:
  * `kaggle kernels list --competition <c> --sort-by voteCount` + `pull`, when
    Kaggle is reachable;
  * a local directory of .ipynb / .py files (the repo's `research/` folder
    holds five s6e9 notebooks pulled by hand).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_CHARS = 24000          # per notebook, after stripping outputs

ASK = """\
You are reading a public Kaggle notebook for a competition we are working. \
Extract what a competitor could USE. Do not summarise the notebook's story.

Write a markdown discovery note (under 250 words) starting with a `# ` title \
that names the core idea. Cover, only where the notebook actually shows it:
- the reported CV and/or public LB score, and the validation scheme behind it;
- every feature-engineering step, by exact column name;
- the model(s) and any non-default settings that look deliberate;
- data facts it relies on (original dataset used, discrete/rounded columns,
  leaks, duplicates, distribution quirks);
- the ONE thing in it most worth testing that a standard GBDT baseline lacks.
Be concrete and skeptical: a claim without a score is a claim, say so."""


def _notebook_text(path: Path) -> str:
    """Code and markdown cells, outputs stripped. Plain .py passes through."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return raw[:MAX_CHARS]
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:MAX_CHARS]
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        tag = "# [markdown]\n" if cell.get("cell_type") == "markdown" else ""
        parts.append(tag + src)
    return "\n\n".join(parts)[:MAX_CHARS]


def pull_top(slug: str, dest: Path, top: int = 5) -> list[Path]:
    """Download the top-voted notebooks for a competition. [] when unreachable."""
    from . import kaggle
    dest.mkdir(parents=True, exist_ok=True)
    out = kaggle.kaggle("kernels", "list", "--competition", kaggle.comp_slug(slug),
                        "--sort-by", "voteCount", "--page-size", str(top), "--csv",
                        timeout=120)
    if out.returncode != 0:
        return []
    refs = [ln.split(",")[0] for ln in out.stdout.splitlines()[1:] if "/" in ln]
    got = []
    for ref in refs[:top]:
        d = dest / ref.replace("/", "__")
        if not d.exists():
            kaggle.kaggle("kernels", "pull", ref, "-p", str(d), timeout=180)
        got += sorted(d.glob("*.ipynb")) + sorted(d.glob("*.py"))
    return got


def study(llm, notes_dir: Path, files: list[Path], slug: str) -> list[str]:
    """Write one note per notebook not already studied. Returns note paths."""
    notes_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in files:
        name = re.sub(r"[^a-z0-9]+", "-", f.stem.lower()).strip("-")[:50]
        out = notes_dir / f"research-{name}.md"
        if out.exists():
            continue                    # studied on an earlier call
        text = _notebook_text(f)
        if len(text) < 200:
            continue
        try:
            body = llm.complete("researcher", ASK,
                                f"Competition: {slug}\nNotebook: {f.name}\n\n{text}")
        except Exception as e:
            print(f"  [research] {f.name}: {str(e)[:100]}")
            continue
        body = body.strip()
        if not body.startswith("#"):
            body = f"# {f.stem}\n\n{body}"
        out.write_text(body + f"\n\n_source: public notebook `{f.name}`_\n",
                       encoding="utf-8")
        written.append(str(out))
    return written


def demo() -> None:
    """Notebook text extraction: code and markdown kept, outputs dropped."""
    import tempfile
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# Title\n", "We use digits."]},
        {"cell_type": "code", "source": ["x = 1\n"],
         "outputs": [{"text": "SECRET OUTPUT " * 50}]},
    ]}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "n.ipynb"
        p.write_text(json.dumps(nb))
        t = _notebook_text(p)
        assert "We use digits" in t and "x = 1" in t, t
        assert "SECRET OUTPUT" not in t, t

        class FakeLLM:
            def complete(self, seat, system, user):
                assert seat == "researcher"
                return "# Digits\nIncome digit features, CV 0.9459."
        notes = Path(d) / "notes"
        big = Path(d) / "big.ipynb"
        nb["cells"][1]["source"] = ["y = 2\n" * 80]
        big.write_text(json.dumps(nb))
        w = study(FakeLLM(), notes, [big], "s6e9")
        assert len(w) == 1 and "public notebook" in Path(w[0]).read_text()
        assert study(FakeLLM(), notes, [big], "s6e9") == []   # not re-studied
    print("research.py: 5 checks passed")


if __name__ == "__main__":
    demo()
