"""The tuned configuration every evaluation test runs with -- the single source of truth.

main.py takes its defaults from here and run_real_balance.sh reads them through `--shell`, so there is
no second copy that can drift, and a forgotten flag can no longer silently give a different setup.
Explicit command-line flags still override; a run started with an override is recorded as such in its
sidecar file (logs/run_<stamp>.json) and the launcher lists the overrides before it starts.

    python experiment_profile.py --shell       # NAME=value lines for the shell launcher
    python experiment_profile.py               # human-readable table
    python experiment_profile.py --list        # the named profiles below
    python experiment_profile.py --flags NAME  # one named profile as run_real_balance.sh flags

The values above are the "standard" set that the chapter-4 report was measured with; they stay the
default. PROFILES holds named, later-tuned sets that run_real_balance.sh applies with --profile=NAME
(explicit flags given after it still override single values).
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


# --- named profiles (run_real_balance.sh --profile=NAME) -------------------------------------------
# Tuned 2026-09-27/28 after the presentation, on the measured model (servo time constant 0.11 s, plant gain
# 0.113 m/s^2 per deg; see theory_limits.MEASURED). Each value is the flag it becomes. Evidence per setting:
#   k-scale/kv-scale 0.143/1.80  LQR re-designed on the measured model; balance runs pass 4/5 (tags bal_*, exp_lqr_*)
#   path-k-full                  stiffer gain for the ramp/path phases only (gain scheduling); hexagon error
#                                1.47 vs 1.81 cm (exp_sched_hex_small_stiff)
#   ta 0.11                      measured servo/plate time constant (tag presentation_room_servo_fb*)
#   vel-ab 0.7,0.35              alpha-beta velocity filter: 3x less velocity noise for ~20 ms extra lag; the
#                                slower 0.5,0.15 fell into a 0.6 Hz limit cycle (exp_sched_hex_stiff_velab)
#   friction-comp 0.6,2          stiction compensation; hexagon error -19 % (exp_c2_fc06, p = 0.03)
#   trim-radius-cm 2             balance only: the bias trim no longer winds up while a far ball is pulled in
#                                (bal_C2_fc06_r2); on a path it made the error worse, so the path profiles keep 10 cm
#   dz-lead-deg 0.25             circle only: 1.37 vs 1.71 cm over 6 pairs (final_dz025_circle, p = 0.04); 0.70 deg
#                                fell into a limit cycle, and on the hexagon even 0.25-0.40 deg did now and then
# Dither (--dither) was tried and dropped: no accuracy gain, ~40 % more servo motion.
_TUNED_COMMON = {
    "k-scale": "0.143",
    "kv-scale": "1.80",
    "ta": "0.11",
    "vel-ab": "0.7,0.35",
    "friction-comp": "0.6,2",
}
PROFILES: dict[str, dict[str, str]] = {
    "standard": {},
    "tuned_balance": {**_TUNED_COMMON, "trim-radius-cm": "2"},
    "tuned_circle": {**_TUNED_COMMON, "path-k-full": "1.047,0.571,0.474", "dz-lead-deg": "0.25"},
    "tuned_hexagon": {**_TUNED_COMMON, "path-k-full": "1.047,0.571,0.474"},
}


def profile_flags(name: str) -> list[str]:
    """run_real_balance.sh flags for a named profile ("--k-scale=0.143", ...)."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}")
    return [f"--{flag}={value}" for flag, value in PROFILES[name].items()]


def as_dict() -> dict:
    return {name[len("PROFILE_"):].lower(): value for name, value in SHELL_EXPORTS.items()}


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--flags"]:
        if len(args) != 2 or args[1] not in PROFILES:
            print(f"usage: experiment_profile.py --flags NAME   (NAME: {', '.join(PROFILES)})", file=sys.stderr)
            sys.exit(2)
        print(" ".join(profile_flags(args[1])))
        return
    if args[:1] == ["--list"]:
        for name in PROFILES:
            print(f"{name:<15}{' '.join(profile_flags(name)) or '(the standard values above, no extra flags)'}")
        return
    if "--shell" in sys.argv[1:]:
        for name, value in SHELL_EXPORTS.items():
            print(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}")
        return
    print("tuned profile (used by every test unless a flag overrides it):")
    for name, value in SHELL_EXPORTS.items():
        print(f"  {name[len('PROFILE_'):].lower():<18}{value:g}")


if __name__ == "__main__":
    main()
