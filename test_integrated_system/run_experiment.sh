#!/usr/bin/env bash
# Run one of the three evaluation experiments, N valid trials, always with the same fixed arguments.
#
#   ./run_experiment.sh balance            # 10 trials, ball placed freely, 30 s each
#   ./run_experiment.sh circle             # 10 trials, circle radius 3 cm, 60 s each
#   ./run_experiment.sh hexagon 10         # 10 trials, hexagon around the 3 cm circle, 60 s each
#   ./run_experiment.sh circle 2 --tag=ladder_0.05 -- --path-ki=0.05   # extra launcher flags after --
#   ./run_experiment.sh circle --dry-run   # print the commands, run nothing
#
# Every trial goes through run_real_balance.sh --confirm, so its own "Press Enter to start" prompt (and its
# abort when stdin is closed) is still what stands between this script and the servos. The tuned
# configuration is the profile in experiment_profile.py; anything extra after "--" is an override, is
# shown by the launcher and is stored in the run's sidecar.
#
# Trials are counted from the sidecar files (logs/run_*.json) by tag, so the script resumes where it
# stopped. A trial where the ball was never seen (set-up problem) or that you stopped with Ctrl+C is kept
# in logs/ but not counted; a trial where the ball fell off IS counted, as a failure.
set -u

PROJECT_ROOT="${EXPERIMENT_ROOT:-/home/rpi5/capstone_design}"   # EXPERIMENT_ROOT only for testing the script itself
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
cd "$PROJECT_ROOT/test_integrated_system" || exit 1

usage() {
    echo "Usage: $0 <balance|circle|hexagon> [N=10] [--tag=NAME] [--dry-run] [-- <extra run_real_balance.sh flags>]"
    exit 2
}

[[ $# -ge 1 ]] || usage
EXPERIMENT="$1"; shift
case "$EXPERIMENT" in balance|circle|hexagon) ;; *) usage ;; esac

N=10; TAG="$EXPERIMENT"; DRY=0; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --tag=*) TAG="${1#*=}" ;;
        --) shift; EXTRA=("$@"); break ;;
        *[!0-9]*) usage ;;
        *) N="$1" ;;
    esac
    shift
done
[[ "$N" -ge 1 ]] || usage

eval "$("$VENV_PYTHON" experiment_profile.py --shell)"
if [[ "$EXPERIMENT" == "balance" ]]; then
    FIXED=(--duration-s="$PROFILE_BALANCE_DURATION")
    WHAT="balance at the centre, ball placed freely ($PROFILE_BALANCE_DURATION s)"
else
    FIXED=(--path="$EXPERIMENT" --duration-s="$PROFILE_PATH_DURATION")
    WHAT="$EXPERIMENT path, size 3 cm; ends $PROFILE_PATH_DURATION s after the first detection (6 s hold + 5 s ramp + path)"
fi
COMMAND=(./run_real_balance.sh --confirm "${FIXED[@]}" --tag="$TAG" "${EXTRA[@]}")

echo "Experiment: $EXPERIMENT  ($WHAT)"
echo "Tag: $TAG   trials wanted: $N   configuration: experiment_profile.py${EXTRA[*]:+  + overrides: ${EXTRA[*]}}"
echo "Each trial runs:  ${COMMAND[*]}"
if [[ "$DRY" == "1" ]]; then
    echo "(dry run: nothing was started)"
    exit 0
fi

counted() { "$VENV_PYTHON" run_metrics.py --count-tag "$TAG"; }
newest_sidecar() { ls -t logs/run_*.json 2>/dev/null | head -1; }

while true; do
    have=$(counted)
    if [[ "$have" -ge "$N" ]]; then break; fi
    echo
    echo "================ $EXPERIMENT: trial $((have + 1)) of $N ================"
    if [[ "$EXPERIMENT" == "balance" ]]; then
        echo "Put the ball anywhere on the plate (free placement, but not at the very top edge of the image)."
    else
        echo "Put the ball near the centre of the plate and let it settle; the path starts by itself after ~6 s."
    fi
    before=$(newest_sidecar)
    "${COMMAND[@]}"
    after=$(newest_sidecar)
    if [[ -z "$after" || "$after" == "$before" ]]; then
        echo "No new run was recorded (the launcher stopped before starting)."
    else
        reason=$("$VENV_PYTHON" - "$after" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("ended_reason", "unknown"))
PY
)
        case "$reason" in
            completed|ball_lost)
                echo "Trial counted ($reason):"
                "$VENV_PYTHON" run_metrics.py --last-of-tag "$TAG" ;;
            never_tracked)
                echo "The ball was never seen: set-up problem, NOT counted (log kept). Check the ball and the light, then retry." ;;
            *)
                echo "Trial was stopped early ($reason): NOT counted (log kept)." ;;
        esac
    fi
    have=$(counted)
    if [[ "$have" -lt "$N" ]]; then
        read -r -p "[Enter] next trial   [q] stop here: " answer || { echo; echo "Input closed; stopping."; break; }
        [[ "$answer" == "q" || "$answer" == "Q" ]] && break
    fi
done

echo
echo "================ $EXPERIMENT: $(counted) of $N trials counted ================"
if [[ "$(counted)" -ge 1 ]]; then
    "$VENV_PYTHON" run_metrics.py --tag "$TAG"
fi
