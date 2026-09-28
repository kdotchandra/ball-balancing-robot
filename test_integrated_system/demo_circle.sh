#!/usr/bin/env bash
# Presentation demo: circle path, video recorded as a backup.
# Tuned 2026-09-28 (tag final_nodither_r10_circle): the balance gain in the hold phase, a stiffer gain while the
# path runs (--path-k-full), filtered ball velocity and stiction compensation, plus a 0.25 deg dead-zone lead.
# Dead-zone lead sweep on the circle (tags sweep_dz*_circle, final_dz025_circle): tracking error 1.96 / 1.71 /
# 1.56 / 1.40 cm at 0 / 0.15 / 0.25 / 0.40 deg; 0.70 deg fell into a 0.74 Hz limit cycle and 0.40 deg did so
# once on the hexagon (sweep_dz040_hexagon), so the demo keeps 0.25 deg (no limit cycle in 13 runs).
# Place the ball near the centre, then press Enter when asked. Runs until you press Ctrl+C (never Ctrl+Z); tuning runs keep using --duration-s.
set -u
cd "$(dirname "$0")"
TAG="demo_circle_$(date +%Y%m%d_%H%M%S)"
./run_real_balance.sh --confirm --record-video --path=circle --k-scale=0.143 --kv-scale=1.80 --ta=0.11 --vel-ab=0.7,0.35 --friction-comp=0.6,2 --path-k-full=1.047,0.571,0.474 --dz-lead-deg=0.25 --tag="$TAG" "$@"
echo
# The demo runs until Ctrl+C, so score this run's own log (run_metrics --last-of-tag skips interrupted runs).
sidecar="$(grep -l "\"tag\": \"$TAG\"" logs/run_*.json 2>/dev/null | tail -1)"
if [[ -n "$sidecar" ]]; then
    result="$(../.venv/bin/python run_metrics.py "${sidecar%.json}.csv" 2>/dev/null | grep '^run_')"
    echo "RESULT: ${result:-(no summary: the ball was hardly tracked)}"
fi
echo "Video and log: $(pwd)/logs/  (tag $TAG)"
