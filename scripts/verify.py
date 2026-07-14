#!/usr/bin/env python3
"""The full local verification, mirroring CI exactly: run before any commit.

    .venv/bin/python scripts/verify.py
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO, "website")
PY = os.path.join(REPO, ".venv", "bin", "python")
RUFF = os.path.join(REPO, ".venv", "bin", "ruff")

STEPS: list[tuple[str, list[str], str]] = [
    ("rustfmt", ["cargo", "fmt", "--manifest-path", "core/Cargo.toml", "--check"], REPO),
    ("clippy",
     ["cargo", "clippy", "--manifest-path", "core/Cargo.toml", "--all-targets", "--", "-D", "warnings"],
     REPO),
    ("cargo test", ["cargo", "test", "--manifest-path", "core/Cargo.toml"], REPO),
    ("ruff", [RUFF, "check", "isobenefit_qgis", "tests", "scripts"], REPO),
    ("prose lint", [PY, "scripts/prose_lint.py", "--all"], REPO),
    # plugins.qgis.org parses metadata.txt with configparser (interpolation ON), so a bare
    # % in any value rejects the upload; parsing here catches that before a tag
    ("metadata parse", [
        PY, "-c",
        "import configparser; c = configparser.ConfigParser(); "
        "c.read('isobenefit_qgis/metadata.txt'); "
        "[c.get('general', k) for k in c.options('general')]",
    ], REPO),
    ("pytest", [PY, "-m", "pytest", "tests", "core/tests_py", "-q"], REPO),
    ("astro check", ["npx", "astro", "check"], WEB),
    ("astro build", ["npm", "run", "build"], WEB),
]


def main() -> int:
    for title, cmd, cwd in STEPS:
        print(f"== {title} ==", flush=True)
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:], file=sys.stderr)
            print(f"FAILED: {title}")
            return 1
        tail = (result.stdout or result.stderr).strip().splitlines()
        print(tail[-1] if tail else "OK")
    print("VERIFY-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
