#!/usr/bin/env bash
# Presentation demo: balance at the centre, video recorded as a backup.
# Settings: experiment_profile.PROFILES["tuned_balance"] (values and the evidence for each are documented there).
# Place the ball anywhere on the plate (inside the camera view), then press Enter when asked. Runs until you press Ctrl+C (never Ctrl+Z); tuning runs keep using --duration-s.
set -u
cd "$(dirname "$0")"
TAG="demo_balance_$(date +%Y%m%d_%H%M%S)"
./run_real_balance.sh --confirm --record-video --profile=tuned_balance --tag="$TAG" "$@"
echo
# The demo runs until Ctrl+C, so score this run's own log (run_metrics --last-of-tag skips interrupted runs).
sidecar="$(grep -l "\"tag\": \"$TAG\"" logs/run_*.json 2>/dev/null | tail -1)"
if [[ -n "$sidecar" ]]; then
    result="$(../.venv/bin/python run_metrics.py "${sidecar%.json}.csv" 2>/dev/null | grep '^run_')"
    echo "RESULT: ${result:-(no summary: the ball was hardly tracked)}"
fi
echo "Video and log: $(pwd)/logs/  (tag $TAG)"
