"""Reference paths for the path-following mode: circle, ellipse, rounded hexagon.

A run has three phases: hold at the centre (0), an amplitude ramp so the reference leaves
the centre gently (2), then the full path (3). Phase 1 in the log is 'hold'.
Numbering used in the log column path_phase: 0 = not a path run, 1 = hold, 2 = ramp, 3 = path.
"""

from __future__ import annotations

import math

import numpy as np

SHAPES = ("circle", "ellipse", "hexagon")
MAX_RADIUS_M = 0.09        # hard limit: the camera loses the ball beyond this
WARN_RADIUS_M = 0.07
ELLIPSE_MINOR_RATIO = 0.67
# Fixed sizes, deliberately not adjustable from the command line: every path stays about 3 cm
# from the centre, far inside the 9 cm limit, so a test cannot be started with an unsafe size.
# radius / semi-major axis / circumradius. The hexagon is the one that ENCLOSES the 3 cm circle: its
# inscribed circle (apothem) is 3 cm, so its circumradius is 3 / cos(30 deg) = 3.464 cm.
HEXAGON_APOTHEM_CM = 3.0
TJ_APOTHEM = HEXAGON_APOTHEM_CM
FIXED_SIZE_CM = {"circle": 3.0, "ellipse": 3.0, "hexagon": HEXAGON_APOTHEM_CM / math.cos(math.pi / 6.0)}
_TABLE_N = 720
_FD_EPS_S = 1e-3

PHASE_NONE, PHASE_HOLD, PHASE_RAMP, PHASE_PATH = 0, 1, 2, 3


def _unit_curve(shape: str) -> np.ndarray:
    """One lap of the shape as (N, 2) points, unit size, evenly spaced in phase u."""
    u = np.arange(_TABLE_N) / _TABLE_N
    if shape == "circle":
        return np.stack([np.cos(2 * math.pi * u), np.sin(2 * math.pi * u)], axis=1)
    if shape == "ellipse":
        return np.stack([np.cos(2 * math.pi * u), ELLIPSE_MINOR_RATIO * np.sin(2 * math.pi * u)], axis=1)
    if shape == "hexagon":
        corners = np.array([[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(7)])
        # constant speed along the perimeter: u picks a point on the hexagon by arc length
        pos = u * 6.0
        idx = np.minimum(pos.astype(int), 5)
        frac = (pos - idx)[:, None]
        poly = corners[idx] * (1.0 - frac) + corners[idx + 1] * frac
        # A sharp corner flips the velocity direction instantly; the ball cannot follow that,
        # so round the corners with a circular moving average over 1/12 of the lap.
        window = _TABLE_N // 12
        kernel = np.ones(window) / window
        out = np.empty_like(poly)
        for axis in range(2):
            ext = np.concatenate([poly[-window:, axis], poly[:, axis], poly[:window, axis]])
            smooth = np.convolve(ext, kernel, mode="same")
            out[:, axis] = smooth[window:-window]
        return out
    raise ValueError(f"unknown path shape '{shape}' (choose from {', '.join(SHAPES)})")


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class PathReference:
    """reference position and velocity as a function of time since the run started."""

    def __init__(self, shape: str, period_s: float = 20.0, hold_s: float = 6.0, ramp_s: float = 5.0,
                 size_cm: float | None = None) -> None:
        if size_cm is None:
            if shape not in FIXED_SIZE_CM:
                raise ValueError(f"unknown path shape '{shape}' (choose from {', '.join(SHAPES)})")
            size_cm = FIXED_SIZE_CM[shape]
        if size_cm <= 0 or period_s <= 0 or hold_s < 0 or ramp_s < 0:
            raise ValueError("path size and period must be positive; hold and ramp must not be negative")
        self.shape, self.size_m, self.period_s = shape, size_cm / 100.0, period_s
        self.hold_s, self.ramp_s = hold_s, ramp_s
        self._curve = _unit_curve(shape)
        self.max_radius_m = float(np.max(np.hypot(self._curve[:, 0], self._curve[:, 1]))) * self.size_m
        if self.max_radius_m > MAX_RADIUS_M:
            raise ValueError(
                f"{shape} of size {size_cm:g} cm reaches {self.max_radius_m * 100:.1f} cm from the centre; "
                f"the limit is {MAX_RADIUS_M * 100:.0f} cm")
        self.warning = None
        if self.max_radius_m > WARN_RADIUS_M:
            self.warning = (f"path reaches {self.max_radius_m * 100:.1f} cm from the centre; the detector "
                            f"often loses the ball beyond about 7-9 cm")

    def phase(self, t: float) -> int:
        if t < self.hold_s:
            return PHASE_HOLD
        if t < self.hold_s + self.ramp_s:
            return PHASE_RAMP
        return PHASE_PATH

    def _position(self, t: float) -> tuple[float, float]:
        if t < self.hold_s:
            return 0.0, 0.0
        amp = _smoothstep((t - self.hold_s) / self.ramp_s) if self.ramp_s > 0 else 1.0
        u = ((t - self.hold_s) / self.period_s) % 1.0
        pos = u * _TABLE_N
        i0 = int(pos) % _TABLE_N
        i1 = (i0 + 1) % _TABLE_N
        f = pos - int(pos)
        p = self._curve[i0] * (1.0 - f) + self._curve[i1] * f
        return float(p[0] * self.size_m * amp), float(p[1] * self.size_m * amp)

    def at(self, t: float) -> tuple[float, float, float, float]:
        """(x_ref, y_ref, vx_ref, vy_ref) in metres and metres per second."""
        x, y = self._position(t)
        x1, y1 = self._position(t + _FD_EPS_S)
        x0, y0 = self._position(max(0.0, t - _FD_EPS_S))
        span = (t + _FD_EPS_S) - max(0.0, t - _FD_EPS_S)
        return x, y, (x1 - x0) / span, (y1 - y0) / span

    def describe(self) -> str:
        speed = 2.0 * math.pi * self.size_m / self.period_s * 100.0 if self.shape == "circle" else float("nan")
        note = f", about {speed:.2f} cm/s" if speed == speed else ""
        enclosing = (f", encloses a {TJ_APOTHEM:.2f} cm circle" if self.shape == "hexagon" else "")
        return (f"{self.shape}, size {self.size_m * 100:.2f} cm{enclosing} (max {self.max_radius_m * 100:.1f} cm from centre), "
                f"period {self.period_s:g} s{note}, hold {self.hold_s:g} s, ramp {self.ramp_s:g} s")
