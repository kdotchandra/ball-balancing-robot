#!/usr/bin/env bash
# Presentation demo: balance at the centre, video recorded as a backup.
# Tuned 2026-09-28 (tags bal_C2_fc06_r2, bal_final_nodither): stiffer LQR gain, measured servo time constant,
# filtered ball velocity, stiction compensation, and a bias trim that learns only within 2 cm of the centre so
# it cannot wind up while a ball placed far away is pulled in. Pulled the ball in from 5-9 cm in every trial.
# Place the ball anywhere on the plate (inside the camera view), then press Enter when asked. Runs until you press Ctrl+C (never Ctrl+Z); tuning runs keep using --duration-s.
set -u
cd "$(dirname "$0")"
TAG="demo_balance_$(date +%Y%m%d_%H%M%S)"
./run_real_balance.sh --confirm --record-video --k-scale=0.143 --kv-scale=1.80 --ta=0.11 --vel-ab=0.7,0.35 --friction-comp=0.6,2 --trim-radius-cm=2 --tag="$TAG" "$@"
echo
# The demo runs until Ctrl+C, so score this run's own log (run_metrics --last-of-tag skips interrupted runs).
sidecar="$(grep -l "\"tag\": \"$TAG\"" logs/run_*.json 2>/dev/null | tail -1)"
if [[ -n "$sidecar" ]]; then
    result="$(../.venv/bin/python run_metrics.py "${sidecar%.json}.csv" 2>/dev/null | grep '^run_')"
    echo "RESULT: ${result:-(no summary: the ball was hardly tracked)}"
fi
echo "Video and log: $(pwd)/logs/  (tag $TAG)"
