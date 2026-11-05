#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .claude_research/venv/bin/python ]]; then
  python3 -m venv .claude_research/venv
fi
if ! .claude_research/venv/bin/python -c 'import anthropic' 2>/dev/null; then
  .claude_research/venv/bin/python -m pip install anthropic
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  read -rsp '請輸入新 API Key（不顯示、不存檔）: ' ANTHROPIC_API_KEY
  printf '\n'
  export ANTHROPIC_API_KEY
fi
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-fable-5-1}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://www.myai168.com/ntou/api/anthropic}"
exec .claude_research/venv/bin/python scripts/claude_research_chat.py "$@"
