"""Slow 3D kinematic simulation of the 3RRS platform.

This visualizes coordinated servo motion, link geometry, platform tilt, and
basic kinematic limits. It does not prove structural strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Geometry of the built 3RRS (single source: params.py). H0 is the plate top surface.
from params import ANCHOR_DROP, H0, L1, L2, PSI, RB, RP

# Slow trajectory settings.
SIMULATION_SECONDS = 20.0
FRAME_RATE = 30
TILT_AMPLITUDE_DEG = 3.0
HEIGHT_AMPLITUDE_MM = 4.0

# Plot colors.
BASE_COLOR = "#4c566a"
PLATE_COLOR = "#2e8b8b"
LINK1_COLORS = ("#d95f02", "#1b9e77", "#7570b3")
LINK2_COLOR = "#e6ab02"
TARGET_COLOR = "#cc0000"


@dataclass
class LegPose:
    base: np.ndarray
    elbow: np.ndarray
    platform_anchor: np.ndarray
    q1: float
    q2: float
    length_error: float


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_point(index: int) -> np.ndarray:
    angle = PSI[index]
    return np.array([RB * math.cos(angle), RB * math.sin(angle), 0.0])


def platform_anchor(phi: float, theta: float, height: float, index: int, drop: float = ANCHOR_DROP) -> np.ndarray:
    # The ball joint hangs `drop` below the plate top surface and tilts with the plate.
    local = np.array([
        RP * math.cos(PSI[index]),
        RP * math.sin(PSI[index]),
        -drop,
    ])
    # Match the platform rotation convention used by the IK notebook.
    rotation = rotation_y(theta) @ rotation_x(phi)
    return np.array([0.0, 0.0, height]) + rotation @ local


def solve_leg(phi: float, theta: float, height: float, index: int) -> LegPose:
    target = platform_anchor(phi, theta, height, index)
    base = base_point(index)
    yaw = PSI[index]  # servos sit inside the anchor circle, legs point outward
    relative = rotation_z(-yaw) @ (target - base)
    x_local = float(relative[0])
    z_local = float(relative[2])

    cosine_q2 = (x_local**2 + z_local**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    reachable_cosine = float(np.clip(cosine_q2, -1.0, 1.0))
    q2 = math.acos(reachable_cosine)
    q1 = math.atan2(z_local, x_local) - math.atan2(
        L2 * math.sin(q2), L1 + L2 * math.cos(q2)
    )

    elbow_local = np.array([L1 * math.cos(q1), 0.0, L1 * math.sin(q1)])
    elbow = base + rotation_z(yaw) @ elbow_local
    reconstructed = elbow + rotation_z(yaw) @ np.array([
        L2 * math.cos(q1 + q2),
        0.0,
        L2 * math.sin(q1 + q2),
    ])
    return LegPose(base, elbow, target, q1, q2, float(np.linalg.norm(reconstructed - target)))


def trajectory(time_s: float) -> tuple[float, float, float]:
    # Incommensurate low-frequency components give a smooth, non-repeating path.
    phi = math.radians(TILT_AMPLITUDE_DEG) * (
        0.65 * math.sin(2.0 * math.pi * time_s / 7.0)
        + 0.35 * math.sin(2.0 * math.pi * time_s / 11.0 + 0.8)
    )
    theta = math.radians(TILT_AMPLITUDE_DEG) * (
        0.60 * math.sin(2.0 * math.pi * time_s / 8.5 + 1.2)
        + 0.40 * math.sin(2.0 * math.pi * time_s / 13.0)
    )
    height = H0 + 1e-3 * HEIGHT_AMPLITUDE_MM * math.sin(2.0 * math.pi * time_s / 9.0)
    return phi, theta, height


def set_axes_equal(axis: plt.Axes, limit: float = 0.25) -> None:
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-0.02, 0.28)
    axis.set_box_aspect((1.0, 1.0, 1.2))


def circle_points(radius: float, height: float, count: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, count)
    return radius * np.cos(angles), radius * np.sin(angles), np.full(count, height)


def main() -> None:
    times = np.arange(0.0, SIMULATION_SECONDS, 1.0 / FRAME_RATE)
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    set_axes_equal(axis)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title("3RRS Platform: Slow Coordinated Kinematic Simulation")

    base_x, base_y, base_z = circle_points(RB, 0.0)
    axis.plot(base_x, base_y, base_z, color=BASE_COLOR, linewidth=2.0, label="base")
    for index in range(3):
        base = base_point(index)
        axis.scatter(*base, color=BASE_COLOR, s=35)

    link1_lines = [axis.plot([], [], [], color=LINK1_COLORS[i], linewidth=4)[0] for i in range(3)]
    link2_lines = [axis.plot([], [], [], color=LINK2_COLOR, linewidth=4)[0] for _ in range(3)]
    anchor_points = [axis.plot([], [], [], "o", color=TARGET_COLOR, markersize=6)[0] for _ in range(3)]
    plate_line = axis.plot([], [], [], color=PLATE_COLOR, linewidth=3, label="plate")[0]
    center_point = axis.plot([], [], [], "o", color="black", markersize=5)[0]
    info = axis.text2D(0.02, 0.95, "", transform=axis.transAxes, family="monospace")
    axis.legend(loc="upper right")

    def update(frame_index: int):
        time_s = times[frame_index]
        phi, theta, height = trajectory(time_s)
        legs = [solve_leg(phi, theta, height, index) for index in range(3)]
        anchors = np.array([leg.platform_anchor for leg in legs])

        for index, leg in enumerate(legs):
            link1_lines[index].set_data_3d(
                [leg.base[0], leg.elbow[0]],
                [leg.base[1], leg.elbow[1]],
                [leg.base[2], leg.elbow[2]],
            )
            link2_lines[index].set_data_3d(
                [leg.elbow[0], leg.platform_anchor[0]],
                [leg.elbow[1], leg.platform_anchor[1]],
                [leg.elbow[2], leg.platform_anchor[2]],
            )
            anchor_points[index].set_data_3d(
                [leg.platform_anchor[0]],
                [leg.platform_anchor[1]],
                [leg.platform_anchor[2]],
            )

        plate = np.array([platform_anchor(phi, theta, height, index, drop=0.0) for index in (0, 1, 2, 0)])
        plate_line.set_data_3d(plate[:, 0], plate[:, 1], plate[:, 2])
        center_point.set_data_3d([0.0], [0.0], [height])
        max_reconstruction_error = max(leg.length_error for leg in legs)
        q1_deg = np.rad2deg([leg.q1 for leg in legs])
        q2_deg = np.rad2deg([leg.q2 for leg in legs])
        info.set_text(
            f"t = {time_s:5.2f} s\n"
            f"phi = {math.degrees(phi):+5.2f} deg\n"
            f"theta = {math.degrees(theta):+5.2f} deg\n"
            f"h = {height * 1000:6.1f} mm\n"
            f"q1 = [{q1_deg[0]:5.1f}, {q1_deg[1]:5.1f}, {q1_deg[2]:5.1f}] deg\n"
            f"q2 = [{q2_deg[0]:5.1f}, {q2_deg[1]:5.1f}, {q2_deg[2]:5.1f}] deg\n"
            f"IK endpoint error = {max_reconstruction_error * 1000:.4f} mm"
        )
        return [*link1_lines, *link2_lines, *anchor_points, plate_line, center_point, info]

    FuncAnimation(figure, update, frames=len(times), interval=1000 / FRAME_RATE, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
