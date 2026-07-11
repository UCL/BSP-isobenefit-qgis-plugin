#!/bin/bash
# The full local verification, mirroring CI exactly: run before any commit.
#   bash scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== cargo (CI parity) =="
cargo fmt --manifest-path core/Cargo.toml --check && echo "fmt OK"
cargo clippy --manifest-path core/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo test --manifest-path core/Cargo.toml 2>&1 | grep "test result" | head -1

echo "== ruff =="
.venv/bin/ruff check isobenefit_qgis tests scripts/fetch_scenario.py scripts/render_scenario_gallery.py

echo "== pytest (plugin + core bindings) =="
.venv/bin/python -m pytest tests core/tests_py -q | tail -2

echo "== website (types + build) =="
cd website
npx astro check 2>&1 | tail -1
npm run build > /dev/null 2>&1 && echo "site build OK"
cd ..
echo "VERIFY-DONE"
