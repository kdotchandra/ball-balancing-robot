from __future__ import annotations

import argparse
import math

import numpy as np

from config import load_config, require_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tilt feedback to safe servo positions without moving hardware")
    parser.add_argument("--tilt-x-deg", type=float, default=0.0)
    parser.add_argument("--tilt-y-deg", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=1.0)
    args = parser.parse_args()
    limits = load_config("servo_limits.json")
    geometry = load_config("servo_geometry.json")
    mapping = load_config("servo_mapping.json")
    minimum = np.array(require_values(limits, "mechanical_min"), dtype=float)
    maximum = np.array(require_values(limits, "mechanical_max"), dtype=float)
    neutral = np.array(require_values(geometry, "neutral_position"), dtype=float)
    if mapping.get("status") != "mapping_measured":
        raise SystemExit("Mapping is not measured; complete 04_servo_mapping.py first")
    if args.confidence < 0.45:
        print(f"REJECT confidence={args.confidence:.3f}; positions unchanged: {neutral.astype(int).tolist()}")
        return
    matrix = np.array([mapping["axis_mapping"]["tilt_x"], mapping["axis_mapping"]["tilt_y"]], dtype=float).T
    tilt = np.deg2rad([args.tilt_x_deg, args.tilt_y_deg])
    q_rad = matrix @ tilt
    proposed = neutral + np.rad2deg(q_rad) * (1000.0 / 240.0)
    safe = np.clip(np.rint(proposed), minimum, maximum).astype(int)
    print(f"tilt_deg={[args.tilt_x_deg, args.tilt_y_deg]}")
    print(f"q_deviation_deg={np.rad2deg(q_rad).round(4).tolist()}")
    print(f"proposed_positions={np.rint(proposed).astype(int).tolist()}")
    print(f"safe_positions={safe.tolist()}")
    print("DRY_RUN: no serial command was sent")


if __name__ == "__main__":
    main()
