"""Stiction compensation (opt-in with main.py --friction-comp=U_DEG,V0_CMS).

The ball sticks to the plate until the tilt passes a break-away angle (0.5-2 deg measured; rolling
friction once it moves is at most ~0.6 deg). With the linear gains the plate only reaches that angle once
the ball is 3-5 cm off the reference, or after the slow bias trim has wound up -- and a wound-up trim
throws the ball past the reference once it breaks free.

This adds a fixed extra tilt U towards the reference while the ball is nearly still:
    full while the ball is slower than a small fraction of V0, fading linearly to zero at speed V0,
    so the push disappears as soon as the ball breaks free (unlike the trim, nothing is left over);
    zero within 0.5 cm of the reference and full beyond 1.5 cm, so it cannot chatter across it.
Sign convention matches the bias trim in main.py: the x tilt pushes with -e_x, the y tilt with +e_y.
Use it with a filtered velocity (--vel-ab): the raw velocity noise (~2 cm/s) would switch it randomly.
"""

from __future__ import annotations

import math

E_MIN_M = 0.005    # below this distance from the reference: no push
E_FULL_M = 0.015   # full push beyond this distance


class FrictionComp:
    def __init__(self, u_deg: float, v0_cms: float) -> None:
        if not (0.0 < u_deg <= 2.0 and 0.2 <= v0_cms <= 10.0):
            raise ValueError("friction compensation needs 0 < U <= 2 deg and 0.2 <= V0 <= 10 cm/s")
        self.u_rad = math.radians(u_deg)
        self.v0 = v0_cms / 100.0

    def offsets(self, err_x: float, err_y: float, vx: float, vy: float) -> tuple[float, float]:
        """Extra plate tilt (rad, x and y axes) for position error err = ball - reference [m] and ball
        velocity v [m/s]."""
        dist = math.hypot(err_x, err_y)
        if dist <= E_MIN_M:
            return 0.0, 0.0
        near = min(1.0, (dist - E_MIN_M) / (E_FULL_M - E_MIN_M))
        still = max(0.0, 1.0 - math.hypot(vx, vy) / self.v0)
        u = self.u_rad * near * still
        return -u * err_x / dist, u * err_y / dist
