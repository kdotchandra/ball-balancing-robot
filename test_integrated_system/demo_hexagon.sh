#!/usr/bin/env bash
# Presentation demo: hexagon path, video recorded as a backup.
# Tuned 2026-09-28 (tag final_nodither_r10_hexagon): the balance gain in the hold phase, a stiffer gain while the
# path runs (--path-k-full), filtered ball velocity and stiction compensation. NO dead-zone lead: on the hexagon
# it fell into a ~0.6-0.8 Hz limit cycle and lost the ball (0.40 deg: 1 of 5 runs; 0.25 deg: 1 of 4, incl. demo run
# 20260928T141442Z), while without it final_nodither_r10_hexagon ran 6 of 6 without oscillating (~1.5-2.1 cm).
# Place the ball near the centre, then press Enter when asked. Runs until you press Ctrl+C (never Ctrl+Z); tuning runs keep using --duration-s.
set -u
cd "$(dirname "$0")"
TAG="demo_hexagon_$(date +%Y%m%d_%H%M%S)"
./run_real_balance.sh --confirm --record-video --path=hexagon --k-scale=0.143 --kv-scale=1.80 --ta=0.11 --vel-ab=0.7,0.35 --friction-comp=0.6,2 --path-k-full=1.047,0.571,0.474 --tag="$TAG" "$@"
echo
# The demo runs until Ctrl+C, so score this run's own log (run_metrics --last-of-tag skips interrupted runs).
sidecar="$(grep -l "\"tag\": \"$TAG\"" logs/run_*.json 2>/dev/null | tail -1)"
if [[ -n "$sidecar" ]]; then
    result="$(../.venv/bin/python run_metrics.py "${sidecar%.json}.csv" 2>/dev/null | grep '^run_')"
    echo "RESULT: ${result:-(no summary: the ball was hardly tracked)}"
fi
echo "Video and log: $(pwd)/logs/  (tag $TAG)"
