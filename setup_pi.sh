#!/usr/bin/env bash
# Set up (or check) a Raspberry Pi for the ball-and-plate project.
#
#   ./setup_pi.sh            check only: reports what is present or missing, changes nothing
#   ./setup_pi.sh --install  installs what is missing (apt packages need sudo), then checks again
#
# Tested on Raspberry Pi 5, Raspberry Pi OS / Debian 13 (trixie), Python 3.13.
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"
MODE="${1:---check}"
APT_PACKAGES=(python3-picamera2 python3-libcamera ffmpeg)
FAILS=0

ok()   { printf '  [ ok ] %s\n' "$1"; }
bad()  { printf '  [MISS] %s\n' "$1"; FAILS=$((FAILS + 1)); }
note() { printf '  [note] %s\n' "$1"; }

install() {
    echo "== installing"
    local missing=()
    for pkg in "${APT_PACKAGES[@]}"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if ((${#missing[@]})); then
        echo "  sudo apt install -y ${missing[*]}"
        sudo apt install -y "${missing[@]}" || exit 1
    fi
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "  python3 -m venv $VENV"
        python3 -m venv "$VENV" || exit 1
    fi
    echo "  $VENV/bin/pip install -r requirements-dev.txt"
    "$VENV/bin/python" -m pip install -q --upgrade pip && "$VENV/bin/python" -m pip install -q -r requirements-dev.txt || exit 1
}

check() {
    echo "== system"
    local pyver
    pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)"
    [[ -n "$pyver" ]] && ok "python3 $pyver" || bad "python3"
    for pkg in "${APT_PACKAGES[@]}"; do
        dpkg -s "$pkg" >/dev/null 2>&1 && ok "apt $pkg" || bad "apt $pkg   (sudo apt install $pkg)"
    done
    /usr/bin/python3 -c "import picamera2" 2>/dev/null && ok "picamera2 importable" || bad "picamera2 not importable"
    command -v ffmpeg >/dev/null && ok "ffmpeg (H.264 videos for the presentation kit)" || bad "ffmpeg"

    echo "== hardware"
    if [[ -e /dev/serial0 ]]; then
        ok "/dev/serial0 -> $(readlink /dev/serial0)  (servo bus, 115200 baud)"
    else
        bad "/dev/serial0 missing: raspi-config -> Interface Options -> Serial Port: login shell NO, hardware YES; reboot"
    fi
    id -nG | grep -qw dialout && ok "user in group dialout (serial access)" || bad "user not in dialout: sudo usermod -aG dialout $USER; log in again"
    if command -v rpicam-hello >/dev/null; then
        if rpicam-hello --list-cameras 2>/dev/null | grep -qi "ov9281\|Available cameras"; then
            ok "camera: $(rpicam-hello --list-cameras 2>/dev/null | grep -m1 -E '^[0-9]+ :' | sed 's/^ *//')"
        else
            bad "no camera listed by rpicam-hello (ribbon cable / dtoverlay=ov9281 in /boot/firmware/config.txt)"
        fi
    else
        note "rpicam-hello not installed; camera not checked"
    fi
    pgrep -af "python.*(web_stream|main.py)" >/dev/null && note "a camera program is running now (stop it before a demo)"

    echo "== project"
    if [[ -x "$VENV/bin/python" ]]; then
        ok "venv $VENV ($("$VENV/bin/python" --version 2>&1))"
        if "$VENV/bin/python" - <<'PY' 2>/dev/null
import re, sys
from importlib.metadata import version, PackageNotFoundError
missing = []
for line in open("requirements.txt"):
    line = line.split("#")[0].strip()
    if not line or ";" in line:
        continue
    name, _, want = line.partition("==")
    try:
        have = version(name)
    except PackageNotFoundError:
        missing.append(f"{name} (not installed)")
        continue
    if want and have != want:
        missing.append(f"{name} {have} != {want}")
if missing:
    print("\n".join(missing))
sys.exit(1 if missing else 0)
PY
        then
            ok "requirements.txt satisfied"
        else
            bad "requirements.txt not satisfied: ./setup_pi.sh --install"
        fi
        "$VENV/bin/python" -c "import pytest" 2>/dev/null && ok "pytest (tests: .venv/bin/python -m pytest test_integrated_system/tests)" \
            || bad "pytest missing: .venv/bin/pip install -r requirements-dev.txt"
    else
        bad "venv missing: ./setup_pi.sh --install"
    fi
    for f in ping_pong_tracker/calib_config.json ping_pong_tracker/detection_config.json \
             test_integrated_system/servo_calibration/config/servo_geometry.json \
             test_integrated_system/servo_calibration/config/servo_limits.json \
             test_integrated_system/servo_calibration/config/servo_mapping.json; do
        [[ -f "$f" ]] && ok "calibration $f" || bad "calibration $f missing (see docs/HANDOFF.md, calibration order)"
    done
    free_gb="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)"
    ((free_gb >= 5)) && ok "disk free ${free_gb} GB" || bad "disk free only ${free_gb} GB (videos and debug frames need space)"
}

case "$MODE" in
    --check) ;;
    --install) install ;;
    *) echo "usage: $0 [--check | --install]"; exit 2 ;;
esac
check
echo
if ((FAILS)); then
    echo "$FAILS item(s) missing."
    exit 1
fi
echo "All checks passed."
