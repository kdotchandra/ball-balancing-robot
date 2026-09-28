from __future__ import annotations

import argparse
from datetime import datetime, timezone

from config import SERVO_IDS, load_config, require_values, save_config
from hx35h import HX35HBus


def ask_yes_no(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive soft mechanical-limit sweep")
    parser.add_argument("--port", required=True)
    parser.add_argument("--servo", type=int, choices=SERVO_IDS, required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--move-ms", type=int, default=700)
    parser.add_argument("--confirm", action="store_true", help="Allow movement; required")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to move: add --confirm after securing the mechanism and emergency power cutoff")

    data = load_config("servo_limits.json")
    floor = require_values(data, "floor_threshold")
    index = SERVO_IDS.index(args.servo)
    bus = HX35HBus(args.port)
    current = bus.read_position(args.servo)
    print(f"Servo {args.servo}: current={current}, measured floor={floor[index]}")
    print("Move only while watching for contact, binding, noise, or rising current. Type stop to abort.")
    min_seen = max_seen = current
    try:
        while True:
            command = input("target position, or stop: ").strip().lower()
            if command == "stop":
                break
            target = int(command)
            if not 0 <= target <= 1000:
                print("Position must be 0..1000")
                continue
            if target > floor[index]:
                print(f"Rejected by floor safety: target {target} > floor threshold {floor[index]}")
                continue
            if not ask_yes_no(f"Move servo {args.servo} to {target}?"):
                continue
            bus.move(args.servo, target, args.move_ms)
            feedback = bus.read_position(args.servo)
            min_seen = min(min_seen, feedback)
            max_seen = max(max_seen, feedback)
            print(f"FEEDBACK servo={args.servo} position={feedback} range_seen=[{min_seen},{max_seen}]")
            if not ask_yes_no("Continue? Stop immediately if near the mechanical stop"):
                break
    finally:
        bus.close()

    measured_min = int(input(f"Enter SAFE minimum for servo {args.servo} (not contact point): "))
    measured_max = int(input(f"Enter SAFE maximum for servo {args.servo} (not contact point): "))
    if measured_min >= measured_max:
        raise ValueError("safe minimum must be less than safe maximum")
    minimum = data.get("mechanical_min", [None] * 3)
    maximum = data.get("mechanical_max", [None] * 3)
    minimum[index] = measured_min
    maximum[index] = measured_max
    data["mechanical_min"] = minimum
    data["mechanical_max"] = maximum
    data["measured_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["status"] = "mechanical_partial"
    if all(value is not None for value in minimum + maximum):
        data["status"] = "mechanical_measured"
    path = save_config("servo_limits.json", data)
    print(f"SAVED {path}")


if __name__ == "__main__":
    main()
