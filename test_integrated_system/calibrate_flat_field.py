"""Flat-field (lighting) calibration for the ball detector.

The image gets much darker away from the centre (lens vignetting plus room lighting): in the
presentation room the plate measured 167 grey levels in the middle but 85-110 near the edges,
so a ball at the edge (148) was darker than the bare plate in the middle and no single
brightness threshold could separate ball from plate everywhere.

This script photographs the EMPTY plate, fits a smooth brightness surface to it (ignoring the
dark servo arms and cables), and saves a per-pixel gain map. BallTracker multiplies every
frame by that map before detection, so the plate comes out equally bright everywhere and the
existing thresholds in detection_config.json apply across the whole plate.

Redo it whenever the robot moves to another room or the lighting changes. Delete the output
file (or run with BALL_FLAT_FIELD=0) to go back to the uncorrected detector.

Usage:  take the ball off the plate, then
    .venv/bin/python test_integrated_system/calibrate_flat_field.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["BALL_FLAT_FIELD"] = "0"   # measure the raw image, not one corrected by an older map

import cv2
import numpy as np

from ball_tracker import FLAT_FIELD_PATH, MAX_PLATE_RADIUS_CM, BallTracker

POLY_DEGREE = 4
# Gain limits. 3.0 let the polynomial, extrapolated into the plate corner, lift a dark object there
# into a false ball at about (8, 7.7) cm; capped at 2.0 that false target all but disappeared in the
# A/B replay while the edge gains held (the 10 cm circle needs at most about 2.3).
GAIN_MIN, GAIN_MAX = 0.5, 2.0


def _design_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([u ** i * v ** j for i in range(POLY_DEGREE + 1) for j in range(POLY_DEGREE + 1 - i)], axis=1)


def fit_surface(gray: np.ndarray, plate_radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial surface through the plate pixels, iteratively dropping outliers
    (servo arms and cables are much darker than the fit, reflections much brighter)."""
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    u = (xx - w / 2.0) / (w / 2.0)
    v = (yy - h / 2.0) / (w / 2.0)
    inside = np.hypot(xx - w / 2.0, yy - h / 2.0) <= plate_radius_px
    use = inside & (gray > 20)
    step = 4   # every 4th pixel is plenty for a smooth surface and keeps the fit fast
    sub = np.zeros_like(use)
    sub[::step, ::step] = True
    A_all = _design_matrix(u.ravel(), v.ravel())
    for _ in range(6):
        idx = np.flatnonzero((use & sub).ravel())
        coef, *_ = np.linalg.lstsq(A_all[idx], gray.ravel()[idx].astype(float), rcond=None)
        fit = (A_all @ coef).reshape(h, w)
        ratio = gray / np.maximum(fit, 1.0)
        use = inside & (ratio > 0.85) & (ratio < 1.15)
    return fit, use


def main() -> None:
    parser = argparse.ArgumentParser(description="Flat-field calibration (ball OFF the plate)")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--no-mirror", action="store_true", help="match main.py --no-mirror")
    args = parser.parse_args()

    tracker = BallTracker(mirror=not args.no_mirror)
    try:
        for _ in range(10):   # let the locked exposure settle
            tracker.read()
        frames = []
        while len(frames) < args.frames:
            ok, _, _, frame, _, _, _ = tracker.read()
            if frame.size <= 1:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            frames.append(cv2.resize(gray, (tracker.proc_width, tracker.proc_height), interpolation=cv2.INTER_AREA).astype(np.float32))
        meta = tracker.cap.capture_metadata() if hasattr(tracker.cap, "capture_metadata") else {}
    finally:
        tracker.release()

    mean = np.mean(frames, axis=0)
    h, w = mean.shape
    plate_radius_px = MAX_PLATE_RADIUS_CM * tracker.pixel_per_cm * tracker.proc_scale
    fit, used = fit_surface(mean, plate_radius_px)

    centre = fit[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].mean()
    gain = np.clip(centre / np.maximum(fit, 1.0), GAIN_MIN, GAIN_MAX).astype(np.float32)

    corrected = np.clip(mean * gain, 0, 255)
    yy, xx = np.mgrid[:h, :w]
    inside = np.hypot(xx - w / 2.0, yy - h / 2.0) <= plate_radius_px
    before = mean[used & inside]
    after = corrected[used & inside]
    stats = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames": len(frames),
        "mirror": not args.no_mirror,
        "proc_size": [w, h],
        "exposure_us": meta.get("ExposureTime"),
        "analogue_gain": meta.get("AnalogueGain"),
        "plate_centre_level": round(float(centre), 1),
        "plate_level_before": {"p5": round(float(np.percentile(before, 5)), 1), "p95": round(float(np.percentile(before, 95)), 1)},
        "plate_level_after": {"p5": round(float(np.percentile(after, 5)), 1), "p95": round(float(np.percentile(after, 95)), 1)},
        "gain_range": [round(float(gain[inside].min()), 2), round(float(gain[inside].max()), 2)],
        "plate_pixels_used_pct": round(100.0 * float(used.sum()) / float(inside.sum()), 1),
    }
    np.savez_compressed(FLAT_FIELD_PATH, gain=gain, info=json.dumps(stats))

    preview = np.hstack([mean, corrected, np.clip(gain / GAIN_MAX * 255, 0, 255)]).astype(np.uint8)
    preview_path = FLAT_FIELD_PATH.with_suffix(".png")
    cv2.imwrite(str(preview_path), preview)
    print(json.dumps(stats, indent=2))
    print(f"Saved {FLAT_FIELD_PATH}")
    print(f"Preview (raw | corrected | gain map): {preview_path}")


if __name__ == "__main__":
    main()
