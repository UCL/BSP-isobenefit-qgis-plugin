"""Mechanical gate against AI-flavoured prose in user-facing text.

Checks the high-precision banned patterns (the ones a regex can catch without
false alarms) in the files readers actually see. The judgement-only patterns
(mirrored restatement, meta-commentary) stay with docs/prose-review.md.

Usage:
    prose_lint.py <file> [...]    # lint the given files (non-scoped files pass)
    prose_lint.py --all           # lint every scoped file in the repo

Exit 0 when clean, exit 2 with file:line findings on stderr when a banned
pattern is present. Wired in twice: as a PostToolUse hook on Edit/Write, and
into scripts/verify.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# user-facing scope: prefixes (dirs) and exact files, relative to the repo root
SCOPE_DIRS = ("website/src", "docs", "scenarios")
SCOPE_FILES = ("README.md", "core/README.md", "website/README.md", "isobenefit_qgis/metadata.txt")
SUFFIXES = (".md", ".astro", ".txt")
# names the banned patterns in order to ban them
EXEMPT = ("docs/prose-review.md",)

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("em-dash aside", re.compile(r"—")),
    ("not-X-but-Y", re.compile(r"\bnot only\b[^.\n]{0,60}\bbut\b|\bnot\s+\w[^.,;:\n]{0,40},\s*but\b", re.I)),
    (
        "escalating coda",
        re.compile(r"\b(further|furthermore|alike|as well|even more|all the more)\.(\s|$)", re.I),
    ),
    (
        "grand framing",
        re.compile(
            r"\b(elegant\w*|beautiful\w*|powerful\w*|seamless\w*|crucial\w*|importantly|delv\w+|"
            r"effortless\w*|game.chang\w*|revolutioni\w+|supercharg\w+|master(?:ful|fully)|"
            r"the decisive\b|the whole point\b|full stop\b|the journey\b)",
            re.I,
        ),
    ),
    (
        "stock phrase",
        re.compile(
            r"\b(it'?s worth noting|worth noting|needless to say|in essence|at its core|"
            r"in brief|quietly|silently|sharpens it|takes it further|to put it simply|"
            r"simply put|when it comes to)\b",
            re.I,
        ),
    ),
]


def in_scope(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    if rel in EXEMPT or path.suffix not in SUFFIXES:
        return False
    return rel in SCOPE_FILES or any(rel.startswith(d + "/") for d in SCOPE_DIRS)


def lint(path: Path) -> list[str]:
    findings = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for name, pattern in PATTERNS:
            m = pattern.search(line)
            if m:
                rel = path.resolve().relative_to(REPO).as_posix()
                findings.append(f"{rel}:{lineno}: [{name}] …{m.group(0)}…")
    return findings


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--all":
        targets = [p for d in SCOPE_DIRS for p in (REPO / d).rglob("*") if p.is_file()]
        targets += [REPO / f for f in SCOPE_FILES]
    else:
        targets = [Path(a) for a in argv]
    findings: list[str] = []
    for target in targets:
        if target.exists() and in_scope(target):
            findings.extend(lint(target))
    if findings:
        print("prose_lint: banned pattern(s) — rewrite the sentence, do not synonym-patch:", file=sys.stderr)
        for f in findings:
            print("  " + f, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
