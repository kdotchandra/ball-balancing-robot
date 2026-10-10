from __future__ import annotations

import math

import numpy as np

from params import ANCHOR_DROP, BASE_HEIGHT, H0, J, K_ACT, L1, L2, PSI, RB, RP


def linearized_ik(phi: float, theta: float, h: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # For control commands we use deviation around nominal pose: delta_p = [phi, theta, delta_h].
    p_ref = np.array([phi, theta, h], dtype=float)
    z_ref = J @ p_ref
    q_lin = z_ref / K_ACT
    return z_ref, q_lin


def rotation_matrix_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def base_point(index: int) -> np.ndarray:
    ang = PSI[index]
    return np.array([RB * math.cos(ang), RB * math.sin(ang), BASE_HEIGHT], dtype=float)


def platform_anchor_world(phi: float, theta: float, h: float, index: int) -> np.ndarray:
    x_local = RP * math.cos(PSI[index])
    y_local = RP * math.sin(PSI[index])
    # h is the plate top surface; the ball joint hangs ANCHOR_DROP below it and tilts with the plate.
    anchor_local = np.array([x_local, y_local, -ANCHOR_DROP], dtype=float)

    cx, sx = math.cos(phi), math.sin(phi)
    cy, sy = math.cos(theta), math.sin(theta)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    r = ry @ rx
    return np.array([0.0, 0.0, h], dtype=float) + r @ anchor_local


def world_to_leg_plane(target_world: np.ndarray, index: int) -> np.ndarray:
    leg_yaw = PSI[index]  # legs point outward from the servo axis toward the anchor
    rel_world = np.asarray(target_world, dtype=float) - base_point(index)
    return rotation_matrix_z(-leg_yaw) @ rel_world


_cos_q2_clamp_warned = False


def solve_leg_ik(
    phi: float,
    theta: float,
    h: float,
    index: int,
    elbow_sign: float = 1.0,
) -> tuple[float, float, np.ndarray]:
    target_world = platform_anchor_world(phi, theta, h, index)
    rel_leg = world_to_leg_plane(target_world, index)

    x_local = float(rel_leg[0])
    z_local = float(rel_leg[2])

    raw_cos_q2 = (x_local**2 + z_local**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = float(np.clip(raw_cos_q2, -1.0, 1.0))
    global _cos_q2_clamp_warned
    if not _cos_q2_clamp_warned and abs(raw_cos_q2) > 1.0:
        print(
            f"[kinematics] WARNING: cos_q2={raw_cos_q2:.4f} out of [-1,1], clamped; "
            "leg geometry (RB/RP/L1/L2/H0/PLATFORM_OFFSET) may be inconsistent with the "
            "neutral pose. This only affects the diagnostic servo_theory/servo_err output, "
            "not real actuation (which uses the linearized servo_mapping matrix directly)."
        )
        _cos_q2_clamp_warned = True
    q2 = elbow_sign * math.acos(cos_q2)

    k1 = L1 + L2 * math.cos(q2)
    k2 = L2 * math.sin(q2)
    q1 = math.atan2(z_local, x_local) - math.atan2(k2, k1)

    return q1, q2, rel_leg


def all_legs_exact_ik(phi: float, theta: float, h: float = H0) -> list[tuple[float, float, np.ndarray]]:
    return [solve_leg_ik(phi, theta, h, i) for i in range(3)]
