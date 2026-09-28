"""Capture raw, un-annotated camera frames for offline detector tuning.

Place the ball at a position on the platform, give it a label, and the script
saves lossless grayscale PNGs at several exposure/gain settings so that the
threshold method and the exposure can be tuned against what the detector
actually sees (screenshots of the web viewer have overlays drawn into them and
are lossy, so they cannot be used for this).

Stop any other program holding the camera (web_stream.py, main.py) first.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append("/usr/lib/python3/dist-packages")
from picamera2 import Picamera2

WIDTH, HEIGHT = 1280, 800

# (label, controls) -- None means "leave auto-exposure on and let it converge"
SETTINGS = [
    ("auto", None),
    ("exp33ms_gain4", {"AeEnable": False, "ExposureTime": 33000, "AnalogueGain": 4.0}),
    ("exp16ms_gain8", {"AeEnable": False, "ExposureTime": 16000, "AnalogueGain": 8.0}),
    ("exp8ms_gain16", {"AeEnable": False, "ExposureTime": 8000, "AnalogueGain": 15.9}),
]

SUGGESTED = [
    "center",
    "corner_that_fails",
    "bright_area",
    "dark_area",
    "edge_left",
    "edge_right",
    "edge_top",
    "edge_bottom",
]


def open_camera() -> Picamera2:
    try:
        camera = Picamera2()
    except Exception as exc:
        raise SystemExit(
            f"Cannot open the camera: {exc}\n"
            "Stop web_stream.py / main.py first (they hold the camera), then retry."
        )
    config = camera.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
        controls={"FrameRate": 30},
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)
    return camera


def settle_and_capture(camera: Picamera2, controls: dict | None) -> tuple[np.ndarray, dict]:
    if controls is None:
        camera.set_controls({"AeEnable": True})
        settle_frames = 30  # auto-exposure needs time to converge
    else:
        camera.set_controls(controls)
        settle_frames = 10
    for _ in range(settle_frames):
        camera.capture_metadata()
    frame = camera.capture_array("main")
    meta = camera.capture_metadata()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    info = {
        "exposure_us": int(meta.get("ExposureTime", -1)),
        "analogue_gain": float(meta.get("AnalogueGain", -1.0)),
        "frame_mean": round(float(gray.mean()), 2),
        "frame_p99": round(float(np.percentile(gray, 99)), 1),
        "frame_max": int(gray.max()),
    }
    return gray, info


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "logs" / f"raw_capture_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {out_dir}")
    print(f"For each ball position, {len(SETTINGS)} frames are captured (one per exposure setting).")
    print("Suggested positions: " + ", ".join(SUGGESTED))
    print("Press Enter on an empty label when you are done.\n")

    camera = open_camera()
    records: list[dict] = []
    try:
        index = 0
        while True:
            default = SUGGESTED[index] if index < len(SUGGESTED) else f"pos{index + 1}"
            label = input(f"[{index + 1}] Place the ball, then label this position [{default}]: ").strip()
            if label == "" and index >= len(SUGGESTED):
                break
            if label == "":
                label = default
            if label.lower() in {"q", "quit", "done", "stop"}:
                break

            for setting_name, controls in SETTINGS:
                gray, info = settle_and_capture(camera, controls)
                filename = f"{index + 1:02d}_{label}_{setting_name}.png"
                cv2.imwrite(str(out_dir / filename), gray)  # PNG = lossless
                info.update({"file": filename, "position": label, "setting": setting_name})
                records.append(info)
                print(
                    f"    {setting_name:<16} exposure={info['exposure_us']:>6}us gain={info['analogue_gain']:>5.1f} "
                    f"mean={info['frame_mean']:>6} p99={info['frame_p99']:>5} max={info['frame_max']:>3} -> {filename}"
                )
            index += 1
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; saving what was captured so far.")
    finally:
        camera.stop()
        camera.close()
        (out_dir / "metadata.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    print(f"\nDone: {len(records)} frames in {out_dir}")


if __name__ == "__main__":
    main()
