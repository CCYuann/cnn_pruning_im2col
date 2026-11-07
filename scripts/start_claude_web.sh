#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .claude_research/venv/bin/python ]]; then
  python3 -m venv .claude_research/venv
fi
if ! .claude_research/venv/bin/python -c 'import anthropic' 2>/dev/null; then
  .claude_research/venv/bin/python -m pip install anthropic
fi
exec .claude_research/venv/bin/python scripts/claude_research_web.py "$@"
