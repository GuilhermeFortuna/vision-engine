#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

chmod +x ci.sh .githooks/pre-commit .githooks/pre-push
git config core.hooksPath .githooks

echo "Git hooks installed (core.hooksPath=.githooks)"
echo "  pre-commit: ./ci.sh lint"
echo "  pre-push:   ./ci.sh"
