from __future__ import annotations

import argparse
from datetime import datetime, timezone

from config import SERVO_IDS, load_config, require_values, save_config
from hx35h import HX35HBus


def main() -> None:
    parser = argparse.ArgumentParser(description="One-servo-at-a-time direction and axis mapping test")
    parser.add_argument("--port", required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--move-ms", type=int, default=700)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to move: add --confirm after securing the mechanism")

    geometry = load_config("servo_geometry.json")
    limits = load_config("servo_limits.json")
    neutral = require_values(geometry, "neutral_position")
    minimum = require_values(limits, "mechanical_min")
    maximum = require_values(limits, "mechanical_max")
    bus = HX35HBus(args.port)
    observations = []
    try:
        for index, servo_id in enumerate(SERVO_IDS):
            for sign in (1, -1):
                test = neutral.copy()
                test[index] = max(minimum[index], min(maximum[index], neutral[index] + sign * args.step))
                print(f"TEST servo={servo_id} command={test}; inspect platform tilt")
                bus.move(servo_id, test[index], args.move_ms)
                input("Press Enter after observing, or Ctrl+C to abort...")
                observation = input("Describe tilt direction/axis (e.g. x+, y-, mixed): ").strip()
                observations.append({"servo_id": servo_id, "position": test[index], "offset": sign * args.step, "observation": observation})
                bus.move(servo_id, neutral[index], args.move_ms)
                input("Confirm platform returned to neutral, then press Enter...")
    finally:
        bus.close()

    print("Enter measured signs: +1 or -1, or 0 if not applicable")
    position_sign = [int(input(f"position sign for servo {servo_id}: ")) for servo_id in SERVO_IDS]
    tilt_x = [float(input(f"tilt_x coefficient servo {servo_id}: ")) for servo_id in SERVO_IDS]
    tilt_y = [float(input(f"tilt_y coefficient servo {servo_id}: ")) for servo_id in SERVO_IDS]
    mapping = load_config("servo_mapping.json")
    mapping.update({
        "position_sign": position_sign,
        "axis_mapping": {"tilt_x": tilt_x, "tilt_y": tilt_y},
        "neutral_reference": neutral,
        "test_step_counts": args.step,
        "observations": observations,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "mapping_measured",
    })
    path = save_config("servo_mapping.json", mapping)
    print(f"SAVED {path}")


if __name__ == "__main__":
    main()
