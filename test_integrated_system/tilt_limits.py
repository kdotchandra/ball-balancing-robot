from __future__ import annotations

import math

import numpy as np

from actuator import ServoSafety

COUNTS_PER_DEG = 1000.0 / 240.0
# Absorbs np.rint's up-to-0.5-count rounding error only. NOT servo_limits.json's
# safety_margin_counts, which is already baked into mechanical_min/mechanical_max
# (re-subtracting it there collapses one direction's travel to exactly zero).
DEFAULT_ROUNDING_MARGIN_COUNTS = 1


def _axis_bounds_rad(
    coeffs: np.ndarray,
    neutral: np.ndarray,
    mech_min: np.ndarray,
    mech_max: np.ndarray,
    margin_counts: float,
) -> tuple[float, float]:
    pos_candidates = []
    neg_candidates = []
    for coeff, neutral_i, lo, hi in zip(coeffs, neutral, mech_min, mech_max):
        if abs(coeff) < 1e-9:
            continue  # this servo isn't driven by this axis
        denom = coeff * COUNTS_PER_DEG
        theta_at_hi = (hi - margin_counts - neutral_i) / denom
        theta_at_lo = (lo + margin_counts - neutral_i) / denom
        pos_candidates.append(max(theta_at_hi, theta_at_lo))
        neg_candidates.append(min(theta_at_hi, theta_at_lo))
    if not pos_candidates:
        raise RuntimeError("axis_mapping has all-zero coefficients for this axis; cannot derive a tilt limit")
    return math.radians(max(neg_candidates)), math.radians(min(pos_candidates))


def compute_platform_tilt_limits(
    safety: ServoSafety,
    mapping: np.ndarray,
    margin_counts: float = DEFAULT_ROUNDING_MARGIN_COUNTS,
) -> dict[str, tuple[float, float]]:
    """Derive safe (min_rad, max_rad) platform tilt bounds per axis from calibration.

    `mapping` is the (3, 2) [tilt_x_coeffs, tilt_y_coeffs] matrix as built by
    main.py's load_servo_mapping(). Bounds are asymmetric because the calibrated
    mechanical range around neutral is itself asymmetric per servo.
    """
    tilt_x_coeffs, tilt_y_coeffs = mapping[:, 0], mapping[:, 1]
    return {
        "tilt_x": _axis_bounds_rad(tilt_x_coeffs, safety.neutral, safety.minimum, safety.maximum, margin_counts),
        "tilt_y": _axis_bounds_rad(tilt_y_coeffs, safety.neutral, safety.minimum, safety.maximum, margin_counts),
    }


if __name__ == "__main__":
    from main import load_servo_mapping, load_servo_safety

    safety = load_servo_safety()
    mapping = load_servo_mapping()
    limits = compute_platform_tilt_limits(safety, mapping)
    for axis, (lo, hi) in limits.items():
        print(f"{axis}: [{math.degrees(lo):+.3f}, {math.degrees(hi):+.3f}] deg")
