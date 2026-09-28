#!/usr/bin/env bash
# Presentation demo: hexagon path, video recorded as a backup.
# Settings: experiment_profile.PROFILES["tuned_hexagon"] (values and the evidence for each are documented there).
# Place the ball near the centre, then press Enter when asked. Runs until you press Ctrl+C (never Ctrl+Z); tuning runs keep using --duration-s.
set -u
cd "$(dirname "$0")"
TAG="demo_hexagon_$(date +%Y%m%d_%H%M%S)"
./run_real_balance.sh --confirm --record-video --path=hexagon --profile=tuned_hexagon --tag="$TAG" "$@"
echo
# The demo runs until Ctrl+C, so score this run's own log (run_metrics --last-of-tag skips interrupted runs).
sidecar="$(grep -l "\"tag\": \"$TAG\"" logs/run_*.json 2>/dev/null | tail -1)"
if [[ -n "$sidecar" ]]; then
    result="$(../.venv/bin/python run_metrics.py "${sidecar%.json}.csv" 2>/dev/null | grep '^run_')"
    echo "RESULT: ${result:-(no summary: the ball was hardly tracked)}"
fi
echo "Video and log: $(pwd)/logs/  (tag $TAG)"
