#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/rpi5/capstone_design"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SERIAL_PORT="/dev/serial0"

# The tuned configuration (experiment_profile.py) is the default for everything below. Flags override it;
# an override is listed in the summary and stored in the run's sidecar file.
PROFILE_K_SCALE=0.076; PROFILE_KV_SCALE=2.8; PROFILE_TRIM_KI=0.125; PROFILE_TILT_LIMIT_DEG=8
PROFILE_COMMAND_PERIOD=0.04; PROFILE_MOVE_MS=50; PROFILE_SAVE_FRAMES=5; PROFILE_PATH_KI=0
if [[ -x "$VENV_PYTHON" ]]; then
    eval "$("$VENV_PYTHON" "$PROJECT_ROOT/test_integrated_system/experiment_profile.py" --shell)"
fi
TILT_LIMIT_DEG=""
MAX_TILT_RATE_DEG_S=""
TRIM_KI=""
TRIM_INIT_DEG=""
DETECTOR=""
SAVE_FRAMES=""
K_SCALE=""
K_FULL=""
PATH_K_FULL=""
TA_S=""
VEL_AB=""
DITHER=""
FRICTION_COMP=""
TRIM_RADIUS_CM=""
KV_SCALE=""
PATH_SHAPE=""
PATH_PERIOD_S=""
PATH_HOLD_S=""
PATH_RAMP_S=""
DURATION_S=""
PATH_SIZE_CM=""
PATH_TRIM_KI=""
DZ_LEAD_DEG=""
PATH_KI=""
TAG=""
RECORD_VIDEO=0
SERVO_FEEDBACK_LOG=0
COMMAND_PERIOD="$PROFILE_COMMAND_PERIOD"
MOVE_MS="$PROFILE_MOVE_MS"
CONFIRMED=0

for argument in "$@"; do
    case "$argument" in
        --confirm)
            CONFIRMED=1
            ;;
        --tilt-limit-deg=*)
            TILT_LIMIT_DEG="${argument#*=}"
            ;;
        --max-tilt-rate-deg-s=*)
            MAX_TILT_RATE_DEG_S="${argument#*=}"
            ;;
        --trim-ki=*)
            TRIM_KI="${argument#*=}"
            ;;
        --trim-init-deg=*)
            TRIM_INIT_DEG="${argument#*=}"
            ;;
        --k-scale=*)
            K_SCALE="${argument#*=}"
            ;;
        --kv-scale=*)
            KV_SCALE="${argument#*=}"
            ;;
        --k-full=*)
            K_FULL="${argument#*=}"
            ;;
        --path-k-full=*)
            PATH_K_FULL="${argument#*=}"
            ;;
        --ta=*)
            TA_S="${argument#*=}"
            ;;
        --vel-ab=*)
            VEL_AB="${argument#*=}"
            ;;
        --dither=*)
            DITHER="${argument#*=}"
            ;;
        --friction-comp=*)
            FRICTION_COMP="${argument#*=}"
            ;;
        --trim-radius-cm=*)
            TRIM_RADIUS_CM="${argument#*=}"
            ;;
        --path=*)
            PATH_SHAPE="${argument#*=}"
            ;;
        --path-period-s=*)
            PATH_PERIOD_S="${argument#*=}"
            ;;
        --path-hold-s=*)
            PATH_HOLD_S="${argument#*=}"
            ;;
        --path-ramp-s=*)
            PATH_RAMP_S="${argument#*=}"
            ;;
        --path-ki=*)
            PATH_KI="${argument#*=}"
            ;;
        --tag=*)
            TAG="${argument#*=}"
            ;;
        --path-size-cm=*)
            PATH_SIZE_CM="${argument#*=}"
            ;;
        --path-trim-ki=*)
            PATH_TRIM_KI="${argument#*=}"
            ;;
        --dz-lead-deg=*)
            DZ_LEAD_DEG="${argument#*=}"
            ;;
        --duration-s=*)
            DURATION_S="${argument#*=}"
            ;;
        --command-period=*)
            COMMAND_PERIOD="${argument#*=}"
            ;;
        --move-ms=*)
            MOVE_MS="${argument#*=}"
            ;;
        --detector=*)
            DETECTOR="${argument#*=}"
            ;;
        --save-frames=*)
            SAVE_FRAMES="${argument#*=}"
            ;;
        --record-video)
            RECORD_VIDEO=1
            ;;
        --log-servo-feedback)
            SERVO_FEEDBACK_LOG=1
            ;;
        *)
            echo "Unknown argument: $argument"
            echo "Usage: $0 --confirm [--tilt-limit-deg=DEGREES] [--max-tilt-rate-deg-s=RATE] [--trim-ki=KI] [--trim-init-deg=X,Y] [--detector=old|bgsub] [--save-frames=N] [--k-scale=S] [--kv-scale=V] [--k-full=K1,K2,K3] [--path-k-full=K1,K2,K3] [--ta=SEC] [--vel-ab=ALPHA,BETA] [--dither=AMP_DEG,FREQ_HZ] [--friction-comp=U_DEG,V0_CMS] [--trim-radius-cm=CM] [--command-period=SEC] [--move-ms=MS] [--path=circle|ellipse|hexagon] [--path-ki=KI] [--tag=NAME] [--path-size-cm=CM] [--path-period-s=S] [--path-hold-s=S] [--path-ramp-s=S] [--duration-s=S] [--path-trim-ki=KI] [--dz-lead-deg=DEG] [--record-video] [--log-servo-feedback]"
            exit 2
            ;;
    esac
done

if [[ "$CONFIRMED" != "1" ]]; then
    echo "Refusing real servo output. Add --confirm after checking the mechanism and emergency power cutoff."
    echo "Usage: $0 --confirm [--tilt-limit-deg=DEGREES] [--max-tilt-rate-deg-s=RATE] [--trim-ki=KI] [--trim-init-deg=X,Y] [--detector=old|bgsub] [--save-frames=N] [--k-scale=S] [--kv-scale=V] [--k-full=K1,K2,K3] [--path-k-full=K1,K2,K3] [--ta=SEC] [--vel-ab=ALPHA,BETA] [--dither=AMP_DEG,FREQ_HZ] [--friction-comp=U_DEG,V0_CMS] [--trim-radius-cm=CM] [--command-period=SEC] [--move-ms=MS] [--path=circle|ellipse|hexagon] [--path-ki=KI] [--tag=NAME] [--path-size-cm=CM] [--path-period-s=S] [--path-hold-s=S] [--path-ramp-s=S] [--duration-s=S] [--path-trim-ki=KI] [--dz-lead-deg=DEG] [--record-video] [--log-servo-feedback]"
    exit 2
fi

if ! [[ "$COMMAND_PERIOD" =~ ^0?\.[0-9]+$|^[0-9]+(\.[0-9]+)?$ ]] || ! [[ "$MOVE_MS" =~ ^[0-9]+$ ]]; then
    echo "Bad --command-period ($COMMAND_PERIOD) or --move-ms ($MOVE_MS): need numbers."
    exit 2
fi
if ! awk -v p="$COMMAND_PERIOD" -v m="$MOVE_MS" 'BEGIN{exit !(p>=0.02 && p<=0.5 && m>=20 && m<=500)}'; then
    echo "Out of range: --command-period must be 0.02..0.5 s and --move-ms 20..500 ms."
    exit 2
fi

NUM_RE='^[0-9]+([.][0-9]+)?$'
if [[ -n "$K_FULL" ]]; then
    if ! [[ "$K_FULL" =~ ^[0-9]+([.][0-9]+)?,[0-9]+([.][0-9]+)?,[0-9]+([.][0-9]+)?$ ]]; then
        echo "Bad --k-full: '$K_FULL' must be K1,K2,K3, e.g. 0.748,0.476,0.428"
        exit 2
    fi
    if [[ -n "$K_SCALE" || -n "$KV_SCALE" ]]; then
        echo "--k-full replaces --k-scale/--kv-scale; give one or the other."
        exit 2
    fi
fi
if [[ -n "$PATH_K_FULL" ]]; then
    if ! [[ "$PATH_K_FULL" =~ ^[0-9]+([.][0-9]+)?,[0-9]+([.][0-9]+)?,[0-9]+([.][0-9]+)?$ ]]; then
        echo "Bad --path-k-full: '$PATH_K_FULL' must be K1,K2,K3, e.g. 0.748,0.476,0.428"
        exit 2
    fi
    if [[ -z "$PATH_SHAPE" ]]; then
        echo "--path-k-full only acts while a path is running; add --path=circle|ellipse|hexagon."
        exit 2
    fi
fi
if [[ -n "$VEL_AB" ]] && ! [[ "$VEL_AB" =~ ^[0-9]*[.]?[0-9]+,[0-9]*[.]?[0-9]+$ ]]; then
    echo "Bad --vel-ab: '$VEL_AB' must be ALPHA,BETA, e.g. 0.5,0.15"
    exit 2
fi
if [[ -n "$DITHER" ]] && ! [[ "$DITHER" =~ ^[0-9]*[.]?[0-9]+,[0-9]*[.]?[0-9]+$ ]]; then
    echo "Bad --dither: '$DITHER' must be AMP_DEG,FREQ_HZ, e.g. 0.4,5"
    exit 2
fi
if [[ -n "$FRICTION_COMP" ]] && ! [[ "$FRICTION_COMP" =~ ^[0-9]*[.]?[0-9]+,[0-9]*[.]?[0-9]+$ ]]; then
    echo "Bad --friction-comp: '$FRICTION_COMP' must be U_DEG,V0_CMS, e.g. 0.6,2"
    exit 2
fi
if [[ -n "$TRIM_RADIUS_CM" ]] && { ! [[ "$TRIM_RADIUS_CM" =~ $NUM_RE ]] || ! awk -v v="$TRIM_RADIUS_CM" 'BEGIN{exit !(v>=0.5 && v<=15)}'; }; then
    echo "Bad --trim-radius-cm: '$TRIM_RADIUS_CM' must be between 0.5 and 15"
    exit 2
fi
if [[ -n "$TA_S" ]] && { ! [[ "$TA_S" =~ $NUM_RE ]] || ! awk -v v="$TA_S" 'BEGIN{exit !(v>=0.02 && v<=1.0)}'; }; then
    echo "Bad --ta: '$TA_S' must be a number of seconds between 0.02 and 1.0"
    exit 2
fi
if [[ -n "$TRIM_INIT_DEG" ]] && ! [[ "$TRIM_INIT_DEG" =~ ^[+-]?[0-9]+([.][0-9]+)?,[+-]?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Bad --trim-init-deg: '$TRIM_INIT_DEG' must be X,Y in degrees, e.g. 1.9,-0.4"
    exit 2
fi
for pair in "path-ki:$PATH_KI" "path-size-cm:$PATH_SIZE_CM" "path-trim-ki:$PATH_TRIM_KI" "dz-lead-deg:$DZ_LEAD_DEG" "path-period-s:$PATH_PERIOD_S" "path-hold-s:$PATH_HOLD_S" "path-ramp-s:$PATH_RAMP_S" "duration-s:$DURATION_S"; do
    if [[ -n "${pair#*:}" ]] && ! [[ "${pair#*:}" =~ $NUM_RE ]]; then
        echo "Bad --${pair%%:*}: '${pair#*:}' must be a non-negative number."
        exit 2
    fi
done
if [[ -z "$PATH_SHAPE" ]] && [[ -n "$PATH_TRIM_KI" || -n "$DZ_LEAD_DEG" || -n "$PATH_KI" ]]; then
    echo "--path-ki, --path-trim-ki and --dz-lead-deg only act while a path is running; add --path=circle|ellipse|hexagon."
    exit 2
fi
if [[ -n "$PATH_KI" ]] && ! awk -v v="$PATH_KI" 'BEGIN{exit !(v>=0 && v<=0.3)}'; then
    echo "--path-ki must be between 0 and 0.3 rad/(m*s) (beyond that the linear phase margin falls quickly)."
    exit 2
fi
if [[ -n "$PATH_SIZE_CM" ]]; then
    if [[ -z "$PATH_SHAPE" ]]; then
        echo "--path-size-cm needs --path=circle|ellipse|hexagon."
        exit 2
    fi
    if ! awk -v v="$PATH_SIZE_CM" 'BEGIN{exit !(v>=1 && v<=7)}'; then
        echo "--path-size-cm must be between 1 and 7 (the camera loses the ball beyond about 9 cm)."
        exit 2
    fi
fi
if [[ -n "$PATH_SHAPE" && "$PATH_SHAPE" != "circle" && "$PATH_SHAPE" != "ellipse" && "$PATH_SHAPE" != "hexagon" ]]; then
    echo "Bad --path: '$PATH_SHAPE' must be circle, ellipse or hexagon."
    exit 2
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Missing Python environment: $VENV_PYTHON"
    exit 1
fi

if [[ ! -e "$SERIAL_PORT" ]]; then
    echo "Missing serial port: $SERIAL_PORT"
    exit 1
fi

while read -r pid state command; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    if [[ "$state" == T* ]]; then
        echo "Removing stopped camera process PID=$pid"
        kill -KILL "$pid" 2>/dev/null || true
        continue
    fi
    echo "Another camera application is running (PID=$pid, state=$state): $command"
    echo "Stop it with Ctrl+C, then retry. Do not use Ctrl+Z."
    exit 1
done < <(ps -eo pid=,stat=,args= | awk '/python.*(web_stream\.py|03_ball_detector\.py|test_integrated_system\/main\.py|\/main\.py --headless)/ && !/awk/ {print $1, $2, $0}')

echo "REAL SERVO TEST"
echo "Serial: $SERIAL_PORT"
OVERRIDES=""
differs() { [[ -n "$1" ]] && awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 != b+0)}'; }
differs "$TILT_LIMIT_DEG" "$PROFILE_TILT_LIMIT_DEG" && OVERRIDES+=" tilt-limit=${TILT_LIMIT_DEG}"
[[ -n "$MAX_TILT_RATE_DEG_S" ]] && OVERRIDES+=" max-tilt-rate=${MAX_TILT_RATE_DEG_S}"
differs "$TRIM_KI" "$PROFILE_TRIM_KI" && OVERRIDES+=" trim-ki=${TRIM_KI}"
[[ -n "$TRIM_INIT_DEG" && "$TRIM_INIT_DEG" != "0,0" ]] && OVERRIDES+=" trim-init-deg=${TRIM_INIT_DEG}"
differs "$K_SCALE" "$PROFILE_K_SCALE" && OVERRIDES+=" k-scale=${K_SCALE}"
differs "$KV_SCALE" "$PROFILE_KV_SCALE" && OVERRIDES+=" kv-scale=${KV_SCALE}"
[[ -n "$K_FULL" ]] && OVERRIDES+=" k-full=${K_FULL}"
[[ -n "$PATH_K_FULL" ]] && OVERRIDES+=" path-k-full=${PATH_K_FULL}"
[[ -n "$TA_S" ]] && OVERRIDES+=" ta=${TA_S}"
[[ -n "$VEL_AB" ]] && OVERRIDES+=" vel-ab=${VEL_AB}"
[[ -n "$DITHER" ]] && OVERRIDES+=" dither=${DITHER}"
[[ -n "$FRICTION_COMP" ]] && OVERRIDES+=" friction-comp=${FRICTION_COMP}"
[[ -n "$TRIM_RADIUS_CM" ]] && OVERRIDES+=" trim-radius-cm=${TRIM_RADIUS_CM}"
differs "$SAVE_FRAMES" "$PROFILE_SAVE_FRAMES" && OVERRIDES+=" save-frames=${SAVE_FRAMES}"
differs "$COMMAND_PERIOD" "$PROFILE_COMMAND_PERIOD" && OVERRIDES+=" command-period=${COMMAND_PERIOD}"
differs "$MOVE_MS" "$PROFILE_MOVE_MS" && OVERRIDES+=" move-ms=${MOVE_MS}"
differs "$PATH_KI" "$PROFILE_PATH_KI" && OVERRIDES+=" path-ki=${PATH_KI}"
[[ -n "$K_FULL" ]] && echo "Gains: K = [${K_FULL}] set directly (--k-full; k-scale/kv-scale below are not used)"
echo "Profile: balance gain set  k=${K_SCALE:-$PROFILE_K_SCALE} kv=${KV_SCALE:-$PROFILE_KV_SCALE}  trim-ki=${TRIM_KI:-$PROFILE_TRIM_KI}  tilt limit=${TILT_LIMIT_DEG:-$PROFILE_TILT_LIMIT_DEG} deg"
echo "Servo timing: command every ${COMMAND_PERIOD} s, move time ${MOVE_MS} ms"
if [[ -z "$OVERRIDES" ]]; then
    echo "Overrides: none (standard test configuration)"
else
    echo "Overrides:${OVERRIDES}   <-- NOT the standard test configuration"
fi
[[ -n "$TAG" ]] && echo "Tag: ${TAG}"
[[ "$SERVO_FEEDBACK_LOG" == 1 ]] && echo "Servo feedback: measured positions logged (servoN_fb); commands may wait up to ~8 ms for the bus"
if [[ -n "$PATH_SHAPE" ]]; then
    echo "Path: ${PATH_SHAPE}, size ${PATH_SIZE_CM:-standard (circle radius 3 cm / hexagon inscribed radius 3 cm)}${PATH_SIZE_CM:+ cm}, period ${PATH_PERIOD_S:-20 (default)} s, hold ${PATH_HOLD_S:-6 (default)} s, ramp ${PATH_RAMP_S:-5 (default)} s"
fi
echo "Path integral (LQI) gain: ${PATH_KI:-$PROFILE_PATH_KI} (ramp and path phases only)"
if [[ -n "$PATH_TRIM_KI" || -n "$DZ_LEAD_DEG" ]]; then
    echo "Path-only tuning: trim ki ${PATH_TRIM_KI:-same as --trim-ki}, dead-zone lead ${DZ_LEAD_DEG:-0} deg"
    echo "  (these act during the ramp and path phases only; the hold phase and balance runs are unchanged)"
fi
if [[ -n "$DURATION_S" ]]; then
    echo "Run ends by itself ${DURATION_S} s after the ball is first tracked, or 3 s after it is lost."
fi
echo "Detector: ${DETECTOR:-old (default)}"
if [[ "$DETECTOR" == "bgsub" ]]; then
    echo ""
    echo "TAKE THE BALL OFF THE PLATFORM before starting."
    echo "The tracker records the empty platform as its background for about 1 second, then"
    echo "prints 'place the ball now' -- put the ball on only after you see that line."
fi
echo "Saving every ${SAVE_FRAMES:-$PROFILE_SAVE_FRAMES}th frame to logs/frames_<time>/ for debugging"
echo ""
echo "Emergency power cutoff ready: confirm before continuing."
read -r -p "Press Enter to start, or Ctrl+C to abort: " || { echo; echo "No confirmation received (input closed); aborting without moving the servos."; exit 1; }

cd "$PROJECT_ROOT/test_integrated_system"

TILT_ARGS=()
if [[ -n "$TILT_LIMIT_DEG" ]]; then
    TILT_ARGS=(--tilt-limit-deg "$TILT_LIMIT_DEG")
fi
if [[ -n "$MAX_TILT_RATE_DEG_S" ]]; then
    TILT_ARGS+=(--max-tilt-rate-deg-s "$MAX_TILT_RATE_DEG_S")
fi
if [[ -n "$TRIM_KI" ]]; then
    TILT_ARGS+=(--trim-ki "$TRIM_KI")
fi
if [[ -n "$TRIM_INIT_DEG" ]]; then
    TILT_ARGS+=(--trim-init-deg "$TRIM_INIT_DEG")
fi
if [[ -n "$DETECTOR" ]]; then
    TILT_ARGS+=(--detector "$DETECTOR")
fi
if [[ -n "$SAVE_FRAMES" ]]; then
    TILT_ARGS+=(--save-frames "$SAVE_FRAMES")
fi
if [[ -n "$K_SCALE" ]]; then
    TILT_ARGS+=(--k-scale "$K_SCALE")
fi
if [[ -n "$KV_SCALE" ]]; then
    TILT_ARGS+=(--kv-scale "$KV_SCALE")
fi
if [[ -n "$K_FULL" ]]; then
    TILT_ARGS+=(--k-full "$K_FULL")
fi
if [[ -n "$PATH_K_FULL" ]]; then
    TILT_ARGS+=(--path-k-full "$PATH_K_FULL")
fi
if [[ -n "$TA_S" ]]; then
    TILT_ARGS+=(--ta "$TA_S")
fi
if [[ -n "$VEL_AB" ]]; then
    TILT_ARGS+=(--vel-ab "$VEL_AB")
fi
if [[ -n "$DITHER" ]]; then
    TILT_ARGS+=(--dither "$DITHER")
fi
if [[ -n "$FRICTION_COMP" ]]; then
    TILT_ARGS+=(--friction-comp "$FRICTION_COMP")
fi
if [[ -n "$TRIM_RADIUS_CM" ]]; then
    TILT_ARGS+=(--trim-radius-cm "$TRIM_RADIUS_CM")
fi
if [[ -n "$PATH_SHAPE" ]]; then
    TILT_ARGS+=(--path "$PATH_SHAPE")
fi
if [[ -n "$PATH_PERIOD_S" ]]; then
    TILT_ARGS+=(--path-period-s "$PATH_PERIOD_S")
fi
if [[ -n "$PATH_HOLD_S" ]]; then
    TILT_ARGS+=(--path-hold-s "$PATH_HOLD_S")
fi
if [[ -n "$PATH_RAMP_S" ]]; then
    TILT_ARGS+=(--path-ramp-s "$PATH_RAMP_S")
fi
if [[ -n "$DURATION_S" ]]; then
    TILT_ARGS+=(--duration-s "$DURATION_S")
fi
if [[ -n "$PATH_SIZE_CM" ]]; then
    TILT_ARGS+=(--path-size-cm "$PATH_SIZE_CM")
fi
if [[ -n "$PATH_KI" ]]; then
    TILT_ARGS+=(--path-ki "$PATH_KI")
fi
if [[ -n "$TAG" ]]; then
    TILT_ARGS+=(--tag "$TAG")
fi
if [[ -n "$PATH_TRIM_KI" ]]; then
    TILT_ARGS+=(--path-trim-ki "$PATH_TRIM_KI")
fi
if [[ -n "$DZ_LEAD_DEG" ]]; then
    TILT_ARGS+=(--dz-lead-deg "$DZ_LEAD_DEG")
fi
if [[ "$RECORD_VIDEO" == 1 ]]; then
    TILT_ARGS+=(--record-video)
fi

exec env \
    SERVO_OUTPUT=1 \
    SERVO_PORT="$SERIAL_PORT" \
    SERVO_READBACK=0 \
    SERVO_FEEDBACK_LOG="$SERVO_FEEDBACK_LOG" \
    SERVO_COMMAND_PERIOD="$COMMAND_PERIOD" \
    SERVO_MOVE_MS="$MOVE_MS" \
    SERVO_MAX_DELTA=35 \
    "$VENV_PYTHON" main.py \
    --headless \
    "${TILT_ARGS[@]}"
