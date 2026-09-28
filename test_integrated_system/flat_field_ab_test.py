"""A/B test of the flat-field correction on the SAME camera frames.

1. record:  roll the ball by hand around the plate, especially along all four edges.
            Frames are saved exactly as BallTracker sees them (mirrored, 640x400 grey,
            no flat-field correction).
2. compare: replays that recording through BallTracker twice -- correction OFF and ON --
            at the recorded timing, and reports detection rate and confidence per region.

    .venv/bin/python test_integrated_system/flat_field_ab_test.py record --duration 45
    .venv/bin/python test_integrated_system/flat_field_ab_test.py compare
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ["BALL_FLAT_FIELD"] = "0"   # record the raw image; compare switches the map on by hand

import cv2
import numpy as np

import ball_tracker
from ball_tracker import FLAT_FIELD_PATH, BallTracker

LOG_DIR = Path(__file__).resolve().parent / "logs"
REGIONS = ("centre <5cm", "left edge", "right edge", "top edge", "bottom edge")


def record(duration: float) -> None:
    tracker = BallTracker(mirror=True)
    frames, stamps = [], []
    try:
        for _ in range(10):
            tracker.read()
        print(f"RECORDING {duration:.0f}s -- roll the ball around, especially along all four edges")
        start = time.perf_counter()
        while (t := time.perf_counter() - start) < duration:
            _, _, _, frame, _, _, _ = tracker.read()
            if frame.size <= 1:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            frames.append(cv2.resize(gray, (tracker.proc_width, tracker.proc_height), interpolation=cv2.INTER_AREA))
            stamps.append(t)
    finally:
        tracker.release()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"flatfield_ab_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.npz"
    np.savez_compressed(path, frames=np.array(frames), t=np.array(stamps))
    print(f"saved {len(frames)} frames to {path}")


class _ReplayCamera:
    """Stands in for the camera. Frames are stored at detection size; repeating each pixel 2x2
    makes BallTracker's INTER_AREA downscale give back exactly the stored frame."""

    def __init__(self, frames: np.ndarray, stamps: np.ndarray) -> None:
        self.frames, self.stamps, self.i = frames, stamps, 0
        self.start = None

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        if self.start is None:
            self.start = time.perf_counter() - self.stamps[0]
        wait = self.start + self.stamps[self.i] - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        frame = np.repeat(np.repeat(self.frames[self.i], 2, axis=0), 2, axis=1)
        self.i += 1
        return True, frame

    def release(self) -> None:
        pass


def replay(frames: np.ndarray, stamps: np.ndarray, gain: np.ndarray | None) -> list:
    camera = _ReplayCamera(frames, stamps)
    original_open = BallTracker._open_camera
    original_picam = ball_tracker.Picamera2
    BallTracker._open_camera = lambda self, index: camera
    ball_tracker.Picamera2 = None   # makes read() use camera.read()
    try:
        tracker = BallTracker(mirror=False)   # frames are already mirrored
    finally:
        BallTracker._open_camera = original_open
        ball_tracker.Picamera2 = original_picam
    tracker.flat_gain = gain   # recorded through the mirrored pipeline, so the map applies unflipped
    results = []
    for _ in range(len(frames)):
        ok, x_m, y_m, _, _, _, det = tracker.read()
        results.append((x_m * 100, y_m * 100, det[3]) if ok and det is not None else None)
    return results


def region(x_cm: float, y_cm: float) -> str:
    if np.hypot(x_cm, y_cm) < 5:
        return REGIONS[0]
    if abs(x_cm) >= abs(y_cm):
        return REGIONS[1] if x_cm < 0 else REGIONS[2]
    return REGIONS[3] if y_cm > 0 else REGIONS[4]


def compare(path: Path | None) -> None:
    if path is None:
        path = sorted(LOG_DIR.glob("flatfield_ab_*.npz"))[-1]
    data = np.load(path)
    frames, stamps = data["frames"], data["t"]
    gain = np.load(FLAT_FIELD_PATH)["gain"].astype(np.float32)
    print(f"{path.name}: {len(frames)} frames, {stamps[-1]:.0f}s")
    off = replay(frames, stamps, None)
    on = replay(frames, stamps, gain)

    # Where the ball is in each frame, from whichever run saw it (ON first).
    table = {r: {"frames": 0, "off": [], "on": []} for r in REGIONS}
    unknown = 0
    for a, b in zip(off, on):
        seen = b or a
        if seen is None:
            unknown += 1
            continue
        row = table[region(seen[0], seen[1])]
        row["frames"] += 1
        if a:
            row["off"].append(a[2])
        if b:
            row["on"].append(b[2])

    print(f"\n{'region':<14}{'frames':>7}  {'OFF seen':>9} {'conf med':>8}   {'ON seen':>8} {'conf med':>8}")
    for name in REGIONS:
        row = table[name]
        n = row["frames"]
        if n == 0:
            print(f"{name:<14}{0:>7}  (ball not there during the recording)")
            continue
        def cell(c):
            return f"{100 * len(c) / n:8.1f}% {np.median(c) if c else float('nan'):8.2f}"
        print(f"{name:<14}{n:>7}  {cell(row['off'])}   {cell(row['on'])}")
    print(f"\nframes where neither run saw a ball (hand, ball off plate, or both missed): {unknown}")
    off_total = sum(r is not None for r in off)
    on_total = sum(r is not None for r in on)
    print(f"total detected: OFF {off_total}/{len(frames)}   ON {on_total}/{len(frames)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--duration", type=float, default=45.0)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("file", nargs="?", type=Path)
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.duration)
    else:
        compare(args.file)


if __name__ == "__main__":
    main()
