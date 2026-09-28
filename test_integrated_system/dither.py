"""Deliberate plate dither (opt-in with main.py --dither=AMP_DEG,FREQ_HZ).

The raw velocity estimate used to shake the plate by ~0.8 deg of random noise. That noise also kept the
servos out of their ~0.3 deg dead zone. With the velocity filtered (--vel-ab) the command became
smooth, small corrections stayed inside the dead zone, and the hexagon runs of 2026-09-28 fell into a
0.58 Hz limit cycle. A known-size square wave, fast enough that the ball cannot follow it, gives the
servos that motion back without the random noise.

The dither is added only to the angle that is turned into servo positions: it bypasses the slew-rate
limiter (which would shrink it) and never enters the controller's plate-tilt estimate. x and y run a
quarter period apart so the two axes do not step at the same instant.
"""

from __future__ import annotations

import math


class SquareDither:
    def __init__(self, amplitude_deg: float, freq_hz: float) -> None:
        if not (0.0 < amplitude_deg <= 1.0 and 0.5 <= freq_hz <= 15.0):
            raise ValueError("dither amplitude must be in (0, 1] deg and frequency in [0.5, 15] Hz")
        self.amplitude_rad = math.radians(amplitude_deg)
        self.freq_hz = freq_hz

    def _square(self, t: float) -> float:
        return 1.0 if (t * self.freq_hz) % 1.0 < 0.5 else -1.0

    def offsets(self, t: float) -> tuple[float, float]:
        quarter = 0.25 / self.freq_hz
        return self.amplitude_rad * self._square(t), self.amplitude_rad * self._square(t + quarter)
