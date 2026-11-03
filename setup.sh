#!/usr/bin/env bash
# One uv env per piece: each keeps its own uv.lock; .python-version pins 3.14.
set -euo pipefail
cd "$(dirname "$0")"
for d in biocomp-spine rescue-data; do echo "== uv sync $d"; (cd "$d" && uv sync); done
