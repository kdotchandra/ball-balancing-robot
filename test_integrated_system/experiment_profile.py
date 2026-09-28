"""The tuned configuration every evaluation test runs with -- the single source of truth.

main.py takes its defaults from here and run_real_balance.sh reads them through `--shell`, so there is
no second copy that can drift, and a forgotten flag can no longer silently give a different setup.
Explicit command-line flags still override; a run started with an override is recorded as such in its
sidecar file (logs/run_<stamp>.json) and the launcher lists the overrides before it starts.

    python experiment_profile.py --shell     # NAME=value lines for the shell launcher
    python experiment_profile.py             # human-readable table
"""

from __future__ import annotations

import sys

# --- controller (the "balance gain set", used by EVERY test) -------------------------------------
K_SCALE = 0.076              # position gain scale on the LQR gain K_AXIS
KV_SCALE = 2.8               # velocity gain scale relative to K_SCALE
TRIM_KI = 0.125              # bias-trim integrator, rad/(m*s); balance and hold phase
TILT_LIMIT_DEG = 8.0         # symmetric tilt limit on both axes
MAX_TILT_RATE_DEG_S = 30.0   # slew-rate limit of the command that drives the servos

# --- servo timing --------------------------------------------------------------------------------
COMMAND_PERIOD_S = 0.04      # one servo command every 40 ms
MOVE_MS = 50                 # servo transition time per command
SAVE_FRAMES = 5              # keep every 5th camera frame for debugging

# --- path following (ramp and path phases only; balance mode never sees these) -------------------
PATH_KI = 0.0                # integral (LQI) gain on the tracking error, rad/(m*s); 0 = off (set by the tuning ladder)
PATH_KI_MAX = 0.3            # linear analysis: phase margin stays >= 28 deg up to here at the balance gains
PATH_PERIOD_S = 20.0         # seconds per lap
PATH_HOLD_S = 6.0            # balance at the centre before the path starts
PATH_RAMP_S = 5.0            # the path grows from the centre over this time
CIRCLE_RADIUS_CM = 3.0
HEXAGON_APOTHEM_CM = 3.0     # the hexagon encloses the 3 cm circle (its inscribed circle has this radius)

# --- how long each kind of trial runs ------------------------------------------------------------
BALANCE_DURATION_S = 30.0
PATH_DURATION_S = 60.0
NEVER_TRACKED_TIMEOUT_S = 30.0   # a trial where the ball is never seen is a set-up problem, not a result

SHELL_EXPORTS = {
    "PROFILE_K_SCALE": K_SCALE,
    "PROFILE_KV_SCALE": KV_SCALE,
    "PROFILE_TRIM_KI": TRIM_KI,
    "PROFILE_TILT_LIMIT_DEG": TILT_LIMIT_DEG,
    "PROFILE_COMMAND_PERIOD": COMMAND_PERIOD_S,
    "PROFILE_MOVE_MS": MOVE_MS,
    "PROFILE_SAVE_FRAMES": SAVE_FRAMES,
    "PROFILE_PATH_KI": PATH_KI,
    "PROFILE_BALANCE_DURATION": BALANCE_DURATION_S,
    "PROFILE_PATH_DURATION": PATH_DURATION_S,
}


def as_dict() -> dict:
    return {name[len("PROFILE_"):].lower(): value for name, value in SHELL_EXPORTS.items()}


def main() -> None:
    if "--shell" in sys.argv[1:]:
        for name, value in SHELL_EXPORTS.items():
            print(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}")
        return
    print("tuned profile (used by every test unless a flag overrides it):")
    for name, value in SHELL_EXPORTS.items():
        print(f"  {name[len('PROFILE_'):].lower():<18}{value:g}")


if __name__ == "__main__":
    main()
