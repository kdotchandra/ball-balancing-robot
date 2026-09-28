"""Alpha-beta filter for the ball velocity (opt-in with main.py --vel-ab=ALPHA,BETA).

The controller used a raw one-frame difference of the detected position. Detection noise is only
0.3-0.8 mm, but differencing it at 30 Hz gave 16-42 mm/s of velocity noise, which the velocity gain
turned into 0.4-1.1 deg of plate jitter -- more than the servo dead zone (~0.3 deg).

Replaying the logged positions (hexagon runs of 2026-09-28) through candidate filters:
    raw difference           31 mm/s noise, 15 ms lag
    EMA 30 ms                16 mm/s,       34 ms
    alpha-beta 0.5/0.15      4.3 mm/s,      57 ms   <- 7x less noise for ~40 ms more lag
A variant that also predicts with the plate tilt had slightly less lag but a larger false velocity
while the ball sticks to the plate, so this filter uses the position measurements only.
"""

from __future__ import annotations


class AlphaBeta:
    """One axis. update() takes a position measurement and the time since the previous one."""

    def __init__(self, alpha: float, beta: float) -> None:
        if not (0.0 < alpha <= 1.0 and 0.0 < beta <= 2.0):
            raise ValueError("alpha must be in (0, 1] and beta in (0, 2]")
        self.alpha = alpha
        self.beta = beta
        self.x = 0.0
        self.v = 0.0
        self.ready = False

    def reset(self, x: float) -> None:
        self.x = x
        self.v = 0.0
        self.ready = True

    def update(self, x_meas: float, dt: float) -> float:
        if not self.ready:
            self.reset(x_meas)
            return 0.0
        dt = max(1e-3, dt)
        x_pred = self.x + self.v * dt
        residual = x_meas - x_pred
        self.x = x_pred + self.alpha * residual
        self.v = self.v + (self.beta / dt) * residual
        return self.v
