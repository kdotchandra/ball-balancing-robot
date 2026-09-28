from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
SERVO_IDS = (1, 2, 3)


def load_config(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; complete the previous calibration step first")
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(name: str, data: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def require_values(data: dict, key: str) -> list[int]:
    values = data.get(key)
    if not isinstance(values, list) or len(values) != 3 or any(value is None for value in values):
        raise ValueError(f"{key} must contain three measured values")
    return [int(value) for value in values]
