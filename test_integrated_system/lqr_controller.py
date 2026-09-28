from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from params import K_AXIS, TILT_LIMIT_RAD


@dataclass
class LQRController:
    k_axis: np.ndarray = field(default_factory=lambda: K_AXIS.copy())
    tilt_limit_rad: float = TILT_LIMIT_RAD

    def compute(
        self,
        x_m: float,
        y_m: float,
        x_dot: float,
        y_dot: float,
        theta_x_actual: float,
        theta_y_actual: float,
        x_ref: float,
        y_ref: float,
        x_ref_dot: float = 0.0,
        y_ref_dot: float = 0.0,
        xi_x: float = 0.0,
        xi_y: float = 0.0,
        k_i: float = 0.0,
    ) -> tuple[float, float]:
        # Matches notebook axis formulation for decoupled x/y control channels.
        # theta_y_actual is negated going in and phi_cmd is negated coming out because
        # kinematics.py's Ry(theta)@Rx(phi) rotation order gives phi the opposite sign
        # convention from theta_y here; the double negation keeps this y-channel
        # consistent with that convention.
        # The velocity entry is the velocity ERROR: with a moving reference the ball should move
        # at the reference's velocity, not stand still. (No acceleration feed-forward: even a
        # fast path needs well under 0.2 deg of tilt for that, less than the plate's dead zone.)
        state_x = np.array([x_m - x_ref, x_dot - x_ref_dot, theta_x_actual], dtype=float)
        state_y = np.array([y_m - y_ref, y_dot - y_ref_dot, -theta_y_actual], dtype=float)

        # Integral (LQI) state: xi = integral of the position error, gain k_i. It is used only while
        # following a path (main.py passes k_i = 0 otherwise), where it is the classical cure for a
        # plate that sticks: the integrated error keeps growing until the tilt passes the break-away
        # angle. It enters exactly like the other error terms, so the y sign convention above holds.
        theta_x_cmd = float(-(np.dot(self.k_axis, state_x) + k_i * xi_x))
        phi_cmd = float(-(np.dot(self.k_axis, state_y) + k_i * xi_y))
        theta_y_cmd = -phi_cmd

        theta_x_cmd = float(np.clip(theta_x_cmd, -self.tilt_limit_rad, self.tilt_limit_rad))
        theta_y_cmd = float(np.clip(theta_y_cmd, -self.tilt_limit_rad, self.tilt_limit_rad))
        return theta_x_cmd, theta_y_cmd
