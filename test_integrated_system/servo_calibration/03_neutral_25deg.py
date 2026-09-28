from __future__ import annotations

import argparse
from datetime import datetime, timezone

from config import load_config, require_values, save_config
from hx35h import HX35HBus


def main() -> None:
    parser = argparse.ArgumentParser(description="Set and record a neutral position candidate from servo feedback")
    parser.add_argument("--port", required=True)
    parser.add_argument("--move-ms", type=int, default=800)
    parser.add_argument("--floor-margin", type=int, default=None)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to move: add --confirm after securing the mechanism")

    limits = load_config("servo_limits.json")
    floor = require_values(limits, "floor_threshold")
    minimum = require_values(limits, "mechanical_min")
    maximum = require_values(limits, "mechanical_max")
    floor_margin = args.floor_margin
    if floor_margin is None:
        floor_margin = int(limits.get("safety_margin_counts", 10))
    floor_safe = [max(low, value - floor_margin) for value, low in zip(floor, minimum)]
    bus = HX35HBus(args.port)
    try:
        print(f"Floor thresholds: {floor}")
        print(f"Moving to floor-safe reference: {floor_safe}")
        for servo_id, position in enumerate(floor_safe, start=1):
            bus.move(servo_id, position, args.move_ms)
        print("Floor-safe feedback:", [bus.read_position(i) for i in (1, 2, 3)])
        print("Now adjust by servo positions. This records a neutral candidate; verify the plate is stable.")
        while True:
            raw = input("Enter p1,p2,p3 to move, or done: ").strip().lower()
            if raw == "done":
                break
            positions = [int(value) for value in raw.split(",")]
            if len(positions) != 3:
                print("Use p1,p2,p3")
                continue
            if any(value < low or value > high for value, low, high in zip(positions, minimum, maximum)):
                print(f"Rejected: positions must remain within min={minimum}, max={maximum}")
                continue
            for servo_id, position in enumerate(positions, start=1):
                bus.move(servo_id, position, args.move_ms)
            print("Feedback:", [bus.read_position(i) for i in (1, 2, 3)])
    finally:
        bus.close()

    neutral = [int(value) for value in input("Final measured neutral positions p1,p2,p3: ").split(",")]
    if len(neutral) != 3 or any(value < low or value > high for value, low, high in zip(neutral, minimum, maximum)):
        raise ValueError("neutral must contain 3 positions within mechanical limits")
    geometry = load_config("servo_geometry.json")
    geometry["neutral_position"] = neutral
    geometry["link1_angle_from_floor_deg"] = 25.0
    geometry["floor_safe_reference"] = floor_safe
    geometry["h0_mm"] = float(input("H0 estimate in mm (or 0 if not measured): "))
    geometry["measured_at_utc"] = datetime.now(timezone.utc).isoformat()
    geometry["status"] = "neutral_position_measured"
    path = save_config("servo_geometry.json", geometry)
    print(f"SAVED {path}")


if __name__ == "__main__":
    main()
