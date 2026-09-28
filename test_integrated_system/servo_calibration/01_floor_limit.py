from __future__ import annotations

import argparse
from datetime import datetime, timezone

from config import SERVO_IDS, load_config, save_config
from hx35h import HX35HBus


def main() -> None:
    parser = argparse.ArgumentParser(description="Read current HX-35H positions as floor limits")
    parser.add_argument("--port", required=True, help="Pi serial device, e.g. /dev/ttyUSB0")
    args = parser.parse_args()
    bus = HX35HBus(args.port)
    try:
        positions = [bus.read_position(servo_id) for servo_id in SERVO_IDS]
        data = load_config("servo_limits.json")
        data["floor_threshold"] = positions
        data["measured_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["port"] = args.port
        data["status"] = "floor_measured"
        path = save_config("servo_limits.json", data)
        print(f"FLOOR_LIMITS servo_ids={list(SERVO_IDS)} positions={positions}")
        print(f"SAVED {path}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
