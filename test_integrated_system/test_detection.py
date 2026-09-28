"""Detection-only test: runs the ball tracker with no servo output and no control loop.

Use this to check the vision pipeline on its own before risking the mechanism: roll the
ball around by hand, especially into the spots where detection used to drop out, and the
summary at the end says how often it was seen and where it was lost.
"""

from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np
from pathlib import Path

from ball_tracker import BallTracker

PRINT_INTERVAL_S = 0.25
LOG_FIELDS = ["t_s", "dt_s", "detected", "confidence", "x_m", "y_m", "x_px", "y_px", "radius_px", "frame_age_s"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection-only test (no servo output)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 = until Ctrl+C")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--detector", choices=("old", "bgsub"), default="old")
    parser.add_argument("--save-frames", type=int, default=0, help="save every Nth frame to logs/frames_<time>/")
    args = parser.parse_args()

    log_file = log_writer = None
    if not args.no_log:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"detect_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv"
        log_file = open(log_path, "w", newline="", encoding="utf-8", buffering=1)
        log_writer = csv.writer(log_file)
        log_writer.writerow(LOG_FIELDS)
        print(f"Logging to {log_path}")

    tracker = BallTracker(camera_index=args.camera, mirror=not args.no_mirror,
                          detector_mode=args.detector, save_frames_every=args.save_frames)
    print("\nNO SERVO OUTPUT. Roll the ball by hand, especially into the spots that used to fail.")
    print("Ctrl+C to stop and print the summary.\n")

    start = time.perf_counter()
    last_print = start
    frames = detected_frames = 0
    confidences: list[float] = []
    frame_ages: list[float] = []
    dropouts: list[tuple[float, float, float]] = []   # (duration_s, last_x_m, last_y_m)
    miss_streak = 0
    miss_started = 0.0
    last_seen = (0.0, 0.0)

    try:
        while True:
            ok, x_m, y_m, frame, mask, dt, det = tracker.read()
            now = time.perf_counter()
            t = now - start
            frames += 1
            if tracker.last_frame_age_s == tracker.last_frame_age_s:
                frame_ages.append(tracker.last_frame_age_s)

            if ok and det is not None:
                detected_frames += 1
                confidences.append(float(det[3]))
                last_seen = (x_m, y_m)
                if miss_streak >= 3:
                    dropouts.append((now - miss_started, last_seen[0], last_seen[1]))
                miss_streak = 0
            else:
                if miss_streak == 0:
                    miss_started = now
                miss_streak += 1

            if log_writer is not None:
                log_writer.writerow([
                    f"{t:.4f}", f"{dt:.4f}", int(bool(ok)),
                    f"{det[3]:.3f}" if det is not None else "",
                    f"{x_m:.5f}" if x_m is not None else "",
                    f"{y_m:.5f}" if y_m is not None else "",
                    f"{det[0]:.1f}" if det is not None else "",
                    f"{det[1]:.1f}" if det is not None else "",
                    f"{det[2]:.1f}" if det is not None else "",
                    f"{tracker.last_frame_age_s:.4f}",
                ])

            if now - last_print >= PRINT_INTERVAL_S:
                rate = 100.0 * detected_frames / max(frames, 1)
                fps = frames / max(t, 1e-6)
                if ok and det is not None:
                    line = (f"[SEEN] xy=({x_m:+.3f},{y_m:+.3f})m conf={det[3]:.2f} "
                            f"r={det[2]:.0f}px | seen {rate:5.1f}% | {fps:4.1f} Hz")
                else:
                    line = (f"[MISS] missed {miss_streak:3d} frames in a row      "
                            f"     | seen {rate:5.1f}% | {fps:4.1f} Hz")
                print(f"\r{line:<100}", end="", flush=True)
                last_print = now

            if args.duration > 0 and t >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        tracker.release()
        if log_file is not None:
            log_file.close()

    elapsed = time.perf_counter() - start
    if miss_streak >= 3:   # still lost when the run ended
        dropouts.append((time.perf_counter() - miss_started, last_seen[0], last_seen[1]))

    print("\n================ SUMMARY ================")
    print(f"ran {elapsed:.1f}s, {frames} frames ({frames / max(elapsed, 1e-6):.1f} Hz)")
    rate = 100.0 * detected_frames / max(frames, 1)
    print(f"ball seen in {detected_frames}/{frames} frames = {rate:.1f}%")
    if frame_ages:
        ages = np.array(frame_ages) * 1000.0
        print(f"camera frame age when the loop gets it (vision half of the loop delay): median {np.median(ages):.0f} ms, "
              f"p95 {np.percentile(ages, 95):.0f} ms, max {ages.max():.0f} ms  ({len(ages)} frames)")
    if detected_frames == 0:
        print("the ball was never detected -- was it on the platform after the background was captured?")
        return
    conf_sorted = sorted(confidences)
    print(f"confidence: min={conf_sorted[0]:.2f} p10={conf_sorted[len(conf_sorted) // 10]:.2f} "
          f"median={conf_sorted[len(conf_sorted) // 2]:.2f}")
    print(f"dropouts (3+ frames in a row): {len(dropouts)}")
    for duration, lx, ly in sorted(dropouts, key=lambda d: -d[0])[:10]:
        print(f"   lost {duration:5.2f}s  -- last seen at ({lx:+.3f},{ly:+.3f}) m, "
              f"{math.hypot(lx, ly) * 100:.1f} cm from center")
    if not dropouts:
        print("   none -- detection held for the whole run")


if __name__ == "__main__":
    main()
