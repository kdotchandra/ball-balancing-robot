"""Slide assets for the project presentation, written to logs/presentation/.

  vision       raw grayscale frame -> brightness threshold -> after noise removal
               (the same steps and constants as BallTracker._get_mask, at the tracker's 640 px width)
  calibration  checkerboard photo with the detected corners, a cm grid on the plate view, pixel<->cm table
  detection    Hough circles, the contour/circularity fallback, and the centre converted to cm (tracker's own code)
  tracking     path-following error with both metrics (cross-track and time-based) against the sourced target
  graph        estimated ball velocity and commanded tilt against time for one run log
  replay       top-view animation (MP4) of one run log: ball, trail and reference
  h264 FILE    re-encode a video (e.g. logs/video_<stamp>.mp4 from main.py --record-video) to H.264 so
               PowerPoint/Keynote play it; --start/--length cut a clip

    python presentation_assets.py                       # everything, default sources
    python presentation_assets.py vision --capture      # grab a fresh frame from the camera first
    python presentation_assets.py graph --run 20260924T135504Z --t-max 20
    python presentation_assets.py replay --balance-run 20260924T135504Z --path-run 20260924T141823Z
    python presentation_assets.py h264 logs/video_20260926T101500Z.mp4 --start 2 --length 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
TRACKER_DIR = ROOT.parent / "ping_pong_tracker"
LOGS = ROOT / "logs"
OUT = LOGS / "presentation"
FONT_FILE = ROOT / "assets" / "fonts" / "NotoSansThai.ttf"

DEFAULT_FRAME = LOGS / "raw_capture_20260923T225531Z" / "01_center_exp8ms_gain16.png"   # tracker caps exposure at 8 ms
DEFAULT_BOARD = TRACKER_DIR / "camera_capture.png"
DEFAULT_BALANCE_RUN = "20260924T135504Z"
DEFAULT_PATH_RUN = "20260924T141823Z"
BOARD_SQUARE_CM = 2.0
PROC_WIDTH = 640

C_X = "#2a78d6"
C_Y = "#eb6834"
C_INK = "#1f1f1e"
C_MUTED = "#6b6a63"
C_GRID = "#e4e3dc"

font_manager.fontManager.addfont(str(FONT_FILE))
plt.rcParams.update({
    "font.family": ["Noto Sans Thai", "DejaVu Sans"], "axes.unicode_minus": False,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.8, "axes.titlesize": 13, "axes.labelsize": 11, "savefig.dpi": 200,
})


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pixel_per_cm() -> float:
    return float(load_json(TRACKER_DIR / "calib_config.json")["pixel_per_cm"])


# ----------------------------------------------------------------------------------------------
# vision
# ----------------------------------------------------------------------------------------------

def capture_frame(path: Path) -> None:
    sys.path.append("/usr/lib/python3/dist-packages")
    from picamera2 import Picamera2
    from ball_tracker import BallTracker

    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"size": (1280, 800), "format": "RGB888"},
                                                       controls={"FrameRate": 30}))
    camera.start()
    try:
        BallTracker._lock_ae_awb(camera)       # same locked exposure as a control run
        for _ in range(10):
            camera.capture_metadata()
        frame = camera.capture_array("main")
    finally:
        camera.stop()
        camera.close()
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    print(f"[capture] saved {path}")


def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def vision(frame_path: Path, mirror: bool) -> None:
    cfg = load_json(TRACKER_DIR / "detection_config.json")
    gray_full = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if gray_full is None:
        raise SystemExit(f"cannot read {frame_path}")
    if mirror:
        gray_full = cv2.flip(gray_full, 1)                # the tracker mirrors every frame
    h_full, w_full = gray_full.shape
    proc = cv2.resize(gray_full, (PROC_WIDTH, round(h_full * PROC_WIDTH / w_full)), interpolation=cv2.INTER_AREA)

    blur_k, morph_k = int(cfg["blur_kernel"]), int(cfg["morph_kernel"])
    lo, hi = int(cfg["gray_min"]), int(cfg["gray_max"])
    blur = cv2.GaussianBlur(proc, (blur_k, blur_k), 0)
    thresh = cv2.inRange(blur, lo, hi)
    kernel = np.ones((morph_k, morph_k), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_before = cv2.connectedComponents(thresh)[0] - 1
    n_after = cv2.connectedComponents(clean)[0] - 1
    up = lambda m: cv2.resize(m, (w_full, h_full), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(OUT / "02_raw_gray.png"), gray_full)
    cv2.imwrite(str(OUT / "03_binary_threshold.png"), up(thresh))
    cv2.imwrite(str(OUT / "04_binary_denoised.png"), up(clean))
    strip = np.hstack([
        label_panel(proc, "1) raw grayscale"),
        label_panel(thresh, f"2) threshold {lo}-{hi}: {n_before} blobs"),
        label_panel(clean, f"3) open+close {morph_k}x{morph_k}: {n_after} blobs"),
    ])
    cv2.imwrite(str(OUT / "02-04_pipeline_strip.png"), strip)
    notes = {
        "source_frame": str(frame_path.relative_to(ROOT.parent)) if frame_path.is_relative_to(ROOT.parent) else str(frame_path),
        "mirrored": mirror, "processing_width_px": PROC_WIDTH,
        "steps": [f"GaussianBlur {blur_k}x{blur_k}", f"inRange gray {lo}-{hi}",
                  f"MORPH_OPEN {morph_k}x{morph_k}", f"MORPH_CLOSE {morph_k}x{morph_k} x2"],
        "white_fraction_threshold": round(float(thresh.mean() / 255), 4),
        "white_fraction_denoised": round(float(clean.mean() / 255), 4),
        "blobs_threshold": n_before, "blobs_denoised": n_after,
    }
    (OUT / "02-04_vision_notes.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print(f"[vision] {frame_path.name}: {n_before} white blobs after threshold -> {n_after} after noise removal")


def offline_tracker():
    """A BallTracker with only the detection state set up, so its real _detect_ball/_get_mask run without a camera."""
    from ball_tracker import BallTracker

    cfg = load_json(TRACKER_DIR / "detection_config.json")
    odd = lambda v: int(v) if int(v) % 2 else int(v) + 1
    tracker = BallTracker.__new__(BallTracker)
    tracker.detection_cfg = cfg
    tracker.hough_min_radius = int(cfg.get("hough_min_radius", 32))
    tracker.hough_max_radius = int(cfg.get("hough_max_radius", 48))
    tracker.expected_radius = float(cfg.get("expected_radius", 40))
    tracker.blur_kernel = odd(cfg.get("blur_kernel", 7))
    tracker.morph_kernel = odd(cfg.get("morph_kernel", 5))
    tracker.hough_blur_kernel = 9
    tracker.last_detection = None
    tracker.pixel_per_cm = pixel_per_cm()
    tracker.width, tracker.height = 1280, 800
    return tracker


def detection(frame_path: Path, mirror: bool) -> None:
    tracker = offline_tracker()
    gray_full = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if mirror:
        gray_full = cv2.flip(gray_full, 1)
    h_full, w_full = gray_full.shape
    proc = cv2.resize(gray_full, (PROC_WIDTH, round(h_full * PROC_WIDTH / w_full)), interpolation=cv2.INTER_AREA)
    scale = w_full / PROC_WIDTH

    # Hough circles exactly as _detect_ball calls it; which one survives is decided by _detect_ball itself.
    blurred = cv2.GaussianBlur(proc, (tracker.hough_blur_kernel, tracker.hough_blur_kernel), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=80, param2=22,
                               minRadius=tracker.hough_min_radius, maxRadius=tracker.hough_max_radius)
    raw = [] if circles is None else [tuple(c) for c in np.round(circles[0]).astype(int)]
    det = tracker._detect_ball(proc)

    hough = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    for cx, cy, r in raw:
        cv2.circle(hough, (cx, cy), r, (0, 0, 255), 1, cv2.LINE_AA)
    if det is not None:
        cv2.circle(hough, (int(det[0]), int(det[1])), int(det[2]), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(hough, (int(det[0]), int(det[1])), (0, 220, 0), cv2.MARKER_CROSS, 14, 2)
    rejected = len(raw) - (1 if det is not None else 0)

    # Fallback path (used only when no Hough circle passes): contours of the cleaned mask, judged by circularity.
    mask = tracker._get_mask(proc)
    contour_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_notes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        if area < 50 or per <= 0:
            continue
        circ = 4.0 * np.pi * area / (per * per)
        ok = circ >= 0.6 and 300 <= area <= 35000
        cv2.drawContours(contour_img, [cnt], -1, (0, 220, 0) if ok else (0, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(contour_img, f"circ {circ:.2f}", (x, max(48, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
        contour_notes.append({"area_px": round(area), "circularity": round(circ, 3), "passes_circularity_and_area": ok})

    result = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
    h, w = result.shape[:2]
    cv2.drawMarker(result, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    out_text = "no ball found"
    if det is not None:
        u, v, r = det[0] * scale, det[1] * scale, det[2] * scale
        x_m, y_m = tracker._px_to_m(u, v, w, h)
        cv2.circle(result, (int(u), int(v)), int(r), (0, 220, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(result, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.arrowedLine(result, (w // 2, h // 2), (int(u), int(v)), (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.2)
        out_text = f"(u, v) = ({u:.0f}, {v:.0f}) px  ->  (x, y) = ({x_m * 100:+.1f}, {y_m * 100:+.1f}) cm"
        cv2.putText(result, out_text, (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(result, out_text, (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(OUT / "05_hough_circles.png"), cv2.resize(hough, (w_full, h_full), interpolation=cv2.INTER_LINEAR))
    cv2.imwrite(str(OUT / "05_contour_circularity.png"), cv2.resize(contour_img, (w_full, h_full), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(OUT / "06_ball_position_cm.png"), result)
    strip = np.hstack([
        label_panel(hough, f"4) Hough: {len(raw)} circle(s), {rejected} rejected"),
        label_panel(contour_img, "4b) fallback: contour circularity >= 0.6"),
        label_panel(cv2.resize(result, (PROC_WIDTH, proc.shape[0]), interpolation=cv2.INTER_AREA), "5) centre -> (x, y) cm"),
    ])
    cv2.imwrite(str(OUT / "05-06_detection_strip.png"), strip)
    notes = {"hough_circles": [list(map(int, c)) for c in raw],
             "selected_proc_px": None if det is None else [round(float(det[0]), 1), round(float(det[1]), 1), round(float(det[2]), 1), round(float(det[3]), 3)],
             "output": out_text, "contours": contour_notes,
             "note": "the contour path only runs when no Hough circle passes the brightness/fill checks"}
    (OUT / "05-06_detection_notes.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print(f"[detection] {len(raw)} Hough circle(s), selected {notes['selected_proc_px']}; {out_text}")


# ----------------------------------------------------------------------------------------------
# tracking result against its target
# ----------------------------------------------------------------------------------------------

C_TARGET = "#8a8a84"
C_PASS = "#1a7f37"
C_FAIL = "#c62828"


def tracking_chart() -> None:
    import ch4_figures as CF
    import path_analysis as PA
    import run_metrics as RM
    import theory_limits as TH

    plt.rcParams["font.family"] = ["Noto Sans Thai", "DejaVu Sans"]
    m = CF.manifest()
    target = float(RM._limit("track_mm"))
    null_mm = TH.null_tracking_error_mm(0.030)
    shapes = (("circle", "วงกลม รัศมี 3 cm"), ("hexagon", "หกเหลี่ยม (ครอบวงกลม 3 cm)"))
    series = (("sim", "จำลอง", C_X, None), ("cross", "จริง: ระยะถึงเส้นทาง", C_Y, None),
              ("sync", "จริง: เทียบจุดอ้างอิง ณ เวลาเดียวกัน", C_Y, "//"))

    data = {}
    for key, _ in shapes:
        runs = m[key]["runs"]
        vals = [PA.run_summary(s, key) for s in runs]
        lost = CF._lost_flags(m, key)
        T, P, R = CF.sim_path(key, CF.balance_gains(m), CF.chosen_k_i(m))
        sel = T > CF.HOLD_S + CF.RAMP_S
        data[key] = {"sim": float(np.hypot(*(P[sel] - R[sel]).T).mean() * 1000),
                     "cross": [v["cross_mm"] for v in vals], "sync": [v["sync_mm"] for v in vals],
                     "lost": lost, "n": len(runs)}

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.30)
    width = 0.26
    label_y = -0.07                       # axes fraction: value/verdict rows sit under the bars
    for gi, (key, _) in enumerate(shapes):
        d = data[key]
        for si, (sk, label, colour, hatch) in enumerate(series):
            x = gi + (si - 1) * (width + 0.03)
            value = d[sk] if sk == "sim" else float(np.mean(d[sk]))
            ax.bar(x, value, width, color="white" if hatch else colour, edgecolor=colour, hatch=hatch,
                   linewidth=2, label=label if gi == 0 else None, zorder=2)
            ax.text(x, label_y, f"{value:.1f} mm", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=14, color=C_INK, weight="bold")
            if sk == "sim":
                ax.text(x, label_y - 0.075, "(ทฤษฎี)", transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=11, color=C_MUTED)
                continue
            rng = np.random.default_rng(gi * 10 + si)
            jitter = rng.uniform(-width / 3, width / 3, d["n"])
            for j, (v, lost) in enumerate(zip(d[sk], d["lost"])):
                ax.scatter(x + jitter[j], v, s=30, zorder=3, facecolor="white" if lost else C_INK,
                           edgecolor=C_INK, linewidth=1.2)
            ok = value <= target
            ax.text(x, label_y - 0.075, "ผ่าน ✓" if ok else "ไม่ผ่าน ✗", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=12, color=C_PASS if ok else C_FAIL, weight="bold")
        ax.text(gi, label_y - 0.17, f"{shapes[gi][1]}  ({d['n']} รอบ)", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=14, color=C_INK)
    ax.axhline(target, color=C_TARGET, ls="--", lw=2, zorder=1)
    ax.text(-0.57, target + 0.5, f"เกณฑ์ {target:g} mm", color=C_INK, fontsize=12, ha="left", va="bottom")
    ax.axhline(null_mm, color=C_TARGET, ls=":", lw=1.5, zorder=1)
    ax.text(-0.57, null_mm + 0.5, f"ไม่ควบคุมเลย (บอลนิ่งกลางแผ่น) = {null_mm:.0f} mm", color=C_MUTED, fontsize=11,
            ha="left", va="bottom")
    ax.set_xticks([])
    ax.set_xlim(-0.6, 1.45)
    ax.set_ylim(0, 42)
    ax.set_ylabel("ความคลาดเฉลี่ยช่วงเกาะเส้นทาง (mm)", fontsize=12)
    ax.grid(axis="x", visible=False)
    ax.scatter([], [], s=30, color=C_INK, label="แต่ละรอบ")
    ax.scatter([], [], s=30, facecolor="white", edgecolor=C_INK, label="แต่ละรอบ (ลูกบอลหลุด)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=11, ncol=5,
              handlelength=1.6, columnspacing=1.4)
    fig.suptitle("ผลการเดินตามเส้นทางเทียบเกณฑ์ ด้วยตัวชี้วัด 2 แบบ", fontsize=17, color=C_INK, y=0.975)
    fig.text(0.01, 0.012,
             f"ที่มาของเกณฑ์: {target:g} mm = ครึ่งหนึ่งของความคลาดเมื่อไม่ควบคุมเลย ({null_mm:.0f} mm = รัศมีเส้นทาง) จาก theory_limits.py\n"
             "ระยะถึงเส้นทาง = ระยะสั้นสุดจากลูกบอลถึงเส้น ไม่นับความล่าช้า  |  เทียบ ณ เวลาเดียวกัน = นับรวมช่วงที่บอลตามหลังจุดอ้างอิง  |  "
             "จำลอง = แบบจำลองเชิงเส้นที่เกนเดียวกัน delay ประมาณ 0.13 s",
             fontsize=9.5, color=C_MUTED, va="bottom", linespacing=1.6)
    fig.savefig(OUT / "14_tracking_two_metrics.png", facecolor="white")
    plt.close(fig)

    notes = {"target_mm": target, "target_source": "0.5 x do-nothing error (= path radius) in theory_limits.targets()",
             "do_nothing_mm": null_mm}
    for key, _ in shapes:
        d = data[key]
        notes[key] = {"runs": d["n"], "ball_lost_runs": int(sum(d["lost"])), "sim_mm": round(d["sim"], 2),
                      "cross_track_mean_mm": round(float(np.mean(d["cross"])), 2),
                      "time_based_mean_mm": round(float(np.mean(d["sync"])), 2),
                      "cross_track_runs_within_target": int(sum(v <= target for v in d["cross"])),
                      "time_based_runs_within_target": int(sum(v <= target for v in d["sync"]))}
    (OUT / "14_tracking_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[tracking] {json.dumps(notes, ensure_ascii=False)}")


# ----------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------

def find_board(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for pattern in ((9, 7), (7, 9), (8, 6), (6, 8)):
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            return corners.reshape(pattern[1], pattern[0], 2), pattern
    raise SystemExit("checkerboard not found")


def calibration(board_path: Path, plate_path: Path) -> None:
    ppc_system = pixel_per_cm()
    board = cv2.imread(str(board_path), cv2.IMREAD_GRAYSCALE)
    pts, pattern = find_board(board)
    spacing = np.concatenate([np.linalg.norm(np.diff(pts, axis=1), axis=2).ravel(),
                              np.linalg.norm(np.diff(pts, axis=0), axis=2).ravel()])
    ppc_board = float(spacing.mean()) / BOARD_SQUARE_CM

    vis = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern, pts.reshape(-1, 1, 2).astype(np.float32), True)
    p0, p1 = pts[0, 0], pts[0, 1]
    cv2.line(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 255), 4)
    cv2.putText(vis, f"20 mm square: mean {spacing.mean():.1f} px over {spacing.size} spacings -> {ppc_board:.2f} px/cm",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "05a_checkerboard_corners.png"), vis)

    plate = cv2.imread(str(plate_path), cv2.IMREAD_GRAYSCALE)
    plate = cv2.cvtColor(cv2.flip(plate, 1), cv2.COLOR_GRAY2BGR)
    h, w = plate.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    overlay = plate.copy()
    max_cm = int(max(w, h) / 2 / ppc_system) + 1
    for k in range(-max_cm, max_cm + 1):
        major = k % 5 == 0
        colour, thick = ((0, 220, 255), 2) if major else ((0, 160, 200), 1)
        x = int(round(cx + k * ppc_system))
        y = int(round(cy - k * ppc_system))
        cv2.line(overlay, (x, 0), (x, h), colour, thick)
        cv2.line(overlay, (0, y), (w, y), colour, thick)
        if major and k != 0:
            cv2.putText(overlay, f"{k:+d}", (x + 4, int(cy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, f"{k:+d}", (int(cx) + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    plate = cv2.addWeighted(overlay, 0.75, plate, 0.25, 0)
    cv2.arrowedLine(plate, (int(cx), int(cy)), (int(cx + 3 * ppc_system), int(cy)), (0, 0, 255), 3, tipLength=0.15)
    cv2.arrowedLine(plate, (int(cx), int(cy)), (int(cx), int(cy - 3 * ppc_system)), (0, 200, 0), 3, tipLength=0.15)
    cv2.putText(plate, "+x", (int(cx + 3 * ppc_system) + 6, int(cy) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(plate, "+y", (int(cx) + 8, int(cy - 3 * ppc_system)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.putText(plate, f"grid 1 cm = {ppc_system:.2f} px (labels in cm)", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "05b_plate_cm_grid.png"), plate)

    rows = [["1 px", f"{10.0 / ppc_system:.3f} mm"],
            ["1 cm", f"{ppc_system:.2f} px"],
            ["ลูกปิงปอง Ø 4.0 cm", f"{4.0 * ppc_system:.0f} px"],
            ["ขอบภาพแนวนอน (±640 px)", f"±{640 / ppc_system:.1f} cm"],
            ["ขอบภาพแนวตั้ง (±400 px)", f"±{400 / ppc_system:.1f} cm"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axis("off")
    ax.set_title(f"ตารางแปลงหน่วย pixel ↔ cm  (pixel_per_cm = {ppc_system:.2f}, ภาพ 1280×800)", color=C_INK, pad=10)
    table = ax.table(cellText=rows, colLabels=["ขนาดในภาพ / บนแผ่น", "ค่าที่แปลงได้"], loc="center", cellLoc="center")
    table.scale(1, 1.8)
    table.set_fontsize(12)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor(C_GRID)
        if r == 0:
            cell.set_facecolor("#f0efe9")
            cell.set_text_props(weight="bold")
    fig.text(0.5, 0.02, "x_cm = (u − 640) / pixel_per_cm,   y_cm = (400 − v) / pixel_per_cm", ha="center", color=C_MUTED)
    fig.savefig(OUT / "05c_pixel_cm_table.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    notes = {
        "pixel_per_cm_used_by_controller": round(ppc_system, 3),
        "checkerboard_photo": str(board_path.name), "checkerboard_pattern_inner_corners": list(pattern),
        "checkerboard_square_px_mean": round(float(spacing.mean()), 2), "checkerboard_square_px_std": round(float(spacing.std()), 2),
        "pixel_per_cm_from_this_photo": round(ppc_board, 3),
        "difference_percent": round(100 * (ppc_system - ppc_board) / ppc_system, 1),
    }
    (OUT / "05_calibration_notes.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print(f"[calibration] controller uses {ppc_system:.2f} px/cm; this checkerboard photo gives {ppc_board:.2f} px/cm "
          f"({notes['difference_percent']:+.1f} %)")


# ----------------------------------------------------------------------------------------------
# run logs
# ----------------------------------------------------------------------------------------------

def read_run(stamp: str) -> dict[str, np.ndarray]:
    path = LOGS / f"run_{stamp}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = ("t_s", "control_valid", "x_m", "y_m", "x_ref_m", "y_ref_m", "x_dot_mps", "y_dot_mps",
            "theta_x_cmd_deg", "theta_y_cmd_deg")
    data = {c: np.array([float(r[c] or "nan") for r in rows]) for c in cols}
    valid = data["control_valid"] > 0.5
    t0 = data["t_s"][valid][0]
    data = {c: v[valid] for c, v in data.items()}
    data["t"] = data["t_s"] - t0
    return data


def graph(stamp: str, t_max: float) -> None:
    d = read_run(stamp)
    keep = d["t"] <= t_max
    t = d["t"][keep]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], "x_m", "y_m", 100.0, "ตำแหน่งลูกบอล (cm)"),
        (axes[1], "x_dot_mps", "y_dot_mps", 100.0, "ความเร็วที่คำนวณได้ (cm/s)"),
        (axes[2], "theta_x_cmd_deg", "theta_y_cmd_deg", 1.0, "มุมเอียงที่สั่ง (องศา)"),
    ]
    for ax, cx, cy, scale, ylabel in panels:
        ax.axhline(0, color=C_MUTED, linewidth=0.8)
        ax.plot(t, d[cx][keep] * scale, color=C_X, linewidth=1.6, label="แกน X")
        ax.plot(t, d[cy][keep] * scale, color=C_Y, linewidth=1.6, label="แกน Y")
        ax.set_ylabel(ylabel)
        end_x, end_y = d[cx][keep][-1] * scale, d[cy][keep][-1] * scale
        x_above = end_x >= end_y
        for value, name, above in ((end_x, "X", x_above), (end_y, "Y", not x_above)):
            ax.annotate(name, (t[-1], value), xytext=(6, 7 if above else -7), textcoords="offset points",
                        color=C_INK, fontsize=10, va="center")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("เวลา (s) นับจากเห็นลูกบอลครั้งแรก")
    fig.suptitle(f"ความเร็วที่คำนวณได้และมุมที่สั่ง เทียบกับเวลา (run {stamp})", color=C_INK, fontsize=14)
    fig.savefig(OUT / "09_velocity_angle_vs_time.png", facecolor="white")
    plt.close(fig)

    tail = d["t"] >= min(10.0, d["t"][-1] / 2)
    stats = {
        "run": stamp, "window_s": t_max,
        "start_distance_cm": round(float(math.hypot(d["x_m"][0], d["y_m"][0]) * 100), 2),
        "after_10s_position_rms_cm": round(float(np.sqrt(np.mean(d["x_m"][tail] ** 2 + d["y_m"][tail] ** 2)) * 100), 2),
        "after_10s_velocity_std_cm_s": [round(float(np.std(d["x_dot_mps"][tail]) * 100), 2),
                                         round(float(np.std(d["y_dot_mps"][tail]) * 100), 2)],
        "after_10s_tilt_cmd_std_deg": [round(float(np.std(d["theta_x_cmd_deg"][tail])), 2),
                                        round(float(np.std(d["theta_y_cmd_deg"][tail])), 2)],
        "oscillation_period_s_x": dominant_period(d["t"][tail], d["theta_x_cmd_deg"][tail]),
        "oscillation_period_s_y": dominant_period(d["t"][tail], d["theta_y_cmd_deg"][tail]),
    }
    (OUT / "09_graph_notes.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[graph] run {stamp}: {json.dumps(stats)}")


def dominant_period(t: np.ndarray, y: np.ndarray) -> float | None:
    if len(t) < 64:
        return None
    grid = np.arange(t[0], t[-1], 1 / 30.0)
    yi = np.interp(grid, t, y)
    yi = (yi - yi.mean()) * np.hanning(len(yi))
    spec = np.abs(np.fft.rfft(yi))
    freq = np.fft.rfftfreq(len(yi), 1 / 30.0)
    band = (freq > 0.1) & (freq < 5.0)
    return round(float(1.0 / freq[band][np.argmax(spec[band])]), 2)


# ----------------------------------------------------------------------------------------------
# replay animation
# ----------------------------------------------------------------------------------------------

def replay(stamp: str, label: str, t_max: float) -> None:
    d = read_run(stamp)
    size, scale = 720, 720 / 24.0          # 24 cm field of view, px per cm
    c = size // 2
    to_px = lambda xm, ym: (int(round(c + xm * 100 * scale)), int(round(c - ym * 100 * scale)))
    fps = 30
    grid_t = np.arange(0.0, min(t_max, d["t"][-1]), 1.0 / fps)
    xs, ys = (np.interp(grid_t, d["t"], d[k]) for k in ("x_m", "y_m"))
    xr, yr = (np.interp(grid_t, d["t"], d[k]) for k in ("x_ref_m", "y_ref_m"))
    path = OUT / f"replay_{label}_{stamp}.mp4"
    raw = path.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    base = np.full((size, size, 3), 255, np.uint8)
    for k in range(-12, 13):
        colour = (200, 200, 200) if k % 5 else (150, 150, 150)
        cv2.line(base, (int(c + k * scale), 0), (int(c + k * scale), size), colour, 1)
        cv2.line(base, (0, int(c + k * scale)), (size, int(c + k * scale)), colour, 1)
    cv2.putText(base, "1 grid = 1 cm", (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    blue, orange = (214, 120, 42), (52, 104, 235)
    for i in range(len(grid_t)):
        frame = base.copy()
        start = max(0, i - fps * 20)
        ref_pts = np.array([to_px(a, b) for a, b in zip(xr[: i + 1], yr[: i + 1])], np.int32)
        if len(ref_pts) > 1:
            cv2.polylines(frame, [ref_pts], False, orange, 2, cv2.LINE_AA)
        trail = np.array([to_px(a, b) for a, b in zip(xs[start: i + 1], ys[start: i + 1])], np.int32)
        if len(trail) > 1:
            cv2.polylines(frame, [trail], False, blue, 2, cv2.LINE_AA)
        cv2.drawMarker(frame, to_px(xr[i], yr[i]), orange, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(frame, to_px(xs[i], ys[i]), int(2.0 * scale), blue, -1, cv2.LINE_AA)
        err = math.hypot(xs[i] - xr[i], ys[i] - yr[i]) * 100
        cv2.putText(frame, f"t = {grid_t[i]:5.1f} s   error = {err:4.1f} cm", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(frame, f"log replay: run {stamp} (ball = blue, reference = orange)", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    to_h264(raw, path)
    raw.unlink()
    print(f"[replay] {path.name}: {len(grid_t) / fps:.1f} s")


def to_h264(src: Path, dst: Path, start: float = 0.0, length: float = 0.0) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-i", str(src)]
    if length > 0:
        cmd += ["-t", f"{length}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", nargs="*", help="any of: vision detection calibration graph tracking replay (default: all), or: h264 FILE")
    parser.add_argument("--start", type=float, default=0.0, help="h264: clip start (s)")
    parser.add_argument("--length", type=float, default=0.0, help="h264: clip length (s), 0 = to the end")
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME, help="grayscale frame for the vision slides")
    parser.add_argument("--capture", action="store_true", help="grab a new frame from the camera for the vision slides")
    parser.add_argument("--no-mirror", action="store_true", help="the frame is already mirrored like the tracker's view")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD, help="checkerboard photo (20 mm squares)")
    parser.add_argument("--run", default=None, help="run stamp for the graph (default: a counted balance trial)")
    parser.add_argument("--t-max", type=float, default=20.0, help="seconds shown in the graph")
    parser.add_argument("--balance-run", default=DEFAULT_BALANCE_RUN)
    parser.add_argument("--path-run", default=DEFAULT_PATH_RUN)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.what[:1] == ["h264"]:
        if len(args.what) != 2:
            parser.error("usage: h264 FILE")
        src = Path(args.what[1])
        dst = OUT / f"{src.stem}_h264.mp4"
        to_h264(src, dst, args.start, args.length)
        print(f"[h264] {dst}")
        return
    parts = ("vision", "detection", "calibration", "graph", "tracking", "replay")
    unknown = set(args.what) - set(parts)
    if unknown:
        parser.error(f"unknown part(s): {', '.join(sorted(unknown))}")
    args.what = args.what or list(parts)
    OUT.mkdir(parents=True, exist_ok=True)

    frame = args.frame
    if args.capture:
        frame = OUT / f"capture_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.png"
        capture_frame(frame)
    if "vision" in args.what:
        vision(frame, mirror=not args.no_mirror)
    if "detection" in args.what:
        detection(frame, mirror=not args.no_mirror)
    if "calibration" in args.what:
        calibration(args.board, frame)
    if "graph" in args.what:
        graph(args.run or args.balance_run, args.t_max)
    if "tracking" in args.what:
        tracking_chart()
    if "replay" in args.what:
        replay(args.balance_run, "centering", 20.0)
        replay(args.path_run, "path", 60.0)
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
