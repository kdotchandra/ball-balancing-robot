from __future__ import annotations

import argparse
import math

import numpy as np

from config import SERVO_IDS, load_config, require_values
from hx35h import HX35HBus


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one bounded tilt command directly to HX-35H")
    parser.add_argument("--port", required=True)
    parser.add_argument("--tilt-x-deg", type=float, default=0.0)
    parser.add_argument("--tilt-y-deg", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--move-ms", type=int, default=700)
    parser.add_argument("--tilt-limit-deg", type=float, default=1.0)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    if not args.confirm or not args.enable:
        raise SystemExit("Refusing hardware movement: require both --confirm and --enable")
    if abs(args.tilt_x_deg) > args.tilt_limit_deg or abs(args.tilt_y_deg) > args.tilt_limit_deg:
        raise SystemExit("Requested tilt exceeds the configured test limit")
    if args.confidence < 0.45:
        raise SystemExit("Confidence is too low; no command sent")

    limits = load_config("servo_limits.json")
    geometry = load_config("servo_geometry.json")
    mapping = load_config("servo_mapping.json")
    minimum = np.array(require_values(limits, "mechanical_min"), dtype=float)
    maximum = np.array(require_values(limits, "mechanical_max"), dtype=float)
    neutral = np.array(require_values(geometry, "neutral_position"), dtype=float)
    matrix = np.array([mapping["axis_mapping"]["tilt_x"], mapping["axis_mapping"]["tilt_y"]], dtype=float).T
    q_rad = matrix @ np.deg2rad([args.tilt_x_deg, args.tilt_y_deg])
    proposed = neutral + np.rad2deg(q_rad) * (1000.0 / 240.0)
    safe = np.clip(np.rint(proposed), minimum, maximum).astype(int)
    print(f"About to send servo_ids={list(SERVO_IDS)} positions={safe.tolist()}")
    input("Final physical safety check: press Enter to send, Ctrl+C to abort: ")
    bus = HX35HBus(args.port)
    try:
        for servo_id, position in zip(SERVO_IDS, safe):
            bus.move(servo_id, int(position), args.move_ms)
        feedback = [bus.read_position(servo_id) for servo_id in SERVO_IDS]
        print(f"SENT positions={safe.tolist()} feedback={feedback}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
