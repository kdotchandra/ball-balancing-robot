"""Tuning knobs that act ONLY while a reference path is moving.

Balance mode and path mode share one control loop and one `--trim-ki`; they differ only in which
flags a run was started with. So anything added for path following has to be gated on the path
phase, or a balance run started from the same command line would behave differently. Both helpers
here return exactly the old behaviour when their knob is left at its default, and both ignore the
hold phase (the platform is simply balancing at the centre then) as well as plain balance runs.

Phases come from trajectory.py: 0 = balance run, 1 = hold at centre, 2 = amplitude ramp, 3 = path.
"""

from __future__ import annotations

import math

from trajectory import PHASE_PATH, PHASE_RAMP

ACTIVE_PHASES = (PHASE_RAMP, PHASE_PATH)

# A command smaller than this is treated as "no direction yet", so sensor noise around zero does
# not flip the lead back and forth.
DIRECTION_THRESHOLD_RAD = math.radians(0.02)
# The lead is faded in over this much command, so the platform is not kicked by a full step of lead
# the instant the command leaves zero.
FULL_LEAD_AT_RAD = math.radians(0.3)


def select_trim_ki(base_ki: float, path_trim_ki: float | None, path_phase: int) -> float:
    """Which bias-trim gain to use on this iteration.

    The path needs a faster trim than balancing does: on a path the trim has to keep pushing the
    plate past the dead zone, while during balance the same speed would overshoot the centre.
    `--trim-ki` therefore keeps its meaning everywhere except the ramp and path phases.
    """
    if path_trim_ki is None or path_phase not in ACTIVE_PHASES:
        return base_ki
    return path_trim_ki


def select_gain(balance_k, path_k, path_phase: int):
    """Which LQR gain vector to use on this iteration (gain scheduling).

    Balancing needs stiffness against the plate's stick-slip (a high K1/(1+K3)); following a moving
    reference needs damping (a larger actuator-state gain K3). `--path-k-full` therefore replaces the
    gain only in the ramp and path phases; the hold phase and balance runs keep the main gain.
    """
    if path_k is None or path_phase not in ACTIVE_PHASES:
        return balance_k
    return path_k


class DeadZoneLead:
    """Push the tilt command past the plate's backlash / friction band, per axis.

    The plate does not respond at all until the tilt exceeds roughly the measured dead zone
    (theory_limits.MEASURED['dead_zone_effective']). When the ball has to be moved along a path
    that means the controller spends most of its command just crossing the play. This adds a small
    offset in the direction the command is already heading, which is the classic inverse-backlash
    trick. MEASURED RESULT (real plate, 3 cm circle, path gains 0.15/1.8): it made tracking WORSE.
    With 0.5 deg the ball oscillated at ~0.9 Hz (0.2-2 Hz motion 2-3x the baseline, one lost ball);
    with 1.0 deg the oscillation reached the 8 deg tilt limit and the ball was lost after 19 s.
    The earlier reasoning that it "costs no phase margin" was wrong: a sign-switching offset has a
    describing-function gain of 4*lead/(pi*A) at command amplitude A, i.e. x2-3 at the sub-degree
    commands that matter here, so it acts as extra loop gain -- and the loop has only ~8 deg of
    phase margin at those gains. Kept as an opt-in flag (default 0), not recommended.

    With lead_deg = 0 (the default) `offset` returns 0.0 for every phase, so nothing changes.
    """

    def __init__(self, lead_deg: float = 0.0) -> None:
        if lead_deg < 0.0:
            raise ValueError("dead-zone lead must not be negative")
        self.lead_rad = math.radians(lead_deg)
        self.lead_deg = lead_deg
        self._direction = {"x": 0.0, "y": 0.0}

    @property
    def enabled(self) -> bool:
        return self.lead_rad > 0.0

    def reset(self) -> None:
        """Forget the direction, for when the platform is neutralized after losing the ball."""
        self._direction = {"x": 0.0, "y": 0.0}

    def offset(self, axis: str, cmd_rad: float, path_phase: int) -> float:
        """Extra tilt to add to this axis' target, in radians (0.0 when inactive)."""
        if axis not in self._direction:
            raise KeyError(f"axis must be 'x' or 'y', got {axis!r}")
        if cmd_rad > DIRECTION_THRESHOLD_RAD:
            self._direction[axis] = 1.0
        elif cmd_rad < -DIRECTION_THRESHOLD_RAD:
            self._direction[axis] = -1.0
        if not self.enabled or path_phase not in ACTIVE_PHASES:
            return 0.0
        fade = min(1.0, abs(cmd_rad) / FULL_LEAD_AT_RAD)
        return self._direction[axis] * self.lead_rad * fade
