#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-runs/fixed_checkpoint_high_ratio_sweep}
SERVICE=${SERVICE:-cnn-pruning-high-ratio-gpu1-v2.service}
EXPECTED_RESULTS=${EXPECTED_RESULTS:-78}
PYTHON_BIN=${PYTHON_BIN:-/usr/local/anaconda3/bin/python}
STATUS_LOG="$ROOT/hourly_monitor.log"
SUMMARY_FILE="runs/fixed_checkpoint_all_ratio_summary/all_methods_summary.csv"

mkdir -p "$ROOT"

timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')
completed=$(find "$ROOT" -maxdepth 1 -name '*_results.csv' -type f | wc -l)
service_state=$(systemctl --user is-active "$SERVICE" 2>/dev/null || true)
gpu_state=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits 2>&1 || true)
latest_result=$(find "$ROOT" -maxdepth 1 -name '*_results.csv' -type f -printf '%T@ %f\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
latest_log=$(find "$ROOT/logs" -name '*.log' -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -1 | cut -d' ' -f2-)
latest_line=""
if [[ -n "$latest_log" ]]; then
  latest_line=$(tail -1 "$latest_log")
fi

{
  echo "[$timestamp] completed=$completed/$EXPECTED_RESULTS service=$service_state"
  echo "$gpu_state" | sed 's/^/  GPU /'
  echo "  latest_result=${latest_result:-none}"
  echo "  latest_log=${latest_log:-none}"
  echo "  latest_line=${latest_line:-none}"
} >> "$STATUS_LOG"

if (( completed < EXPECTED_RESULTS )) && [[ "$service_state" != "active" ]]; then
  echo "  action=restart $SERVICE" >> "$STATUS_LOG"
  systemctl --user restart "$SERVICE"
fi

if (( completed >= EXPECTED_RESULTS )); then
  echo "  action=all results complete" >> "$STATUS_LOG"
  if [[ ! -s "$SUMMARY_FILE" ]]; then
    echo "  action=generate combined tables and charts" >> "$STATUS_LOG"
    "$PYTHON_BIN" scripts/summarize_fixed_checkpoint_all_ratios.py >> "$STATUS_LOG" 2>&1
  fi
fi
