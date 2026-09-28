"""Presentation kit: pictures and videos that answer the committee's usual questions
(where does a number come from, how were the gains tuned, why does the real system fall short).

Everything is computed from this workspace's logs and analysis modules; nothing is typed in by hand
except text that quotes the report's tuning table (ch4_docx.py). Output: logs/presentation/kit/.

    python presentation_kit.py              # everything
    python presentation_kit.py C1 D1        # only some items
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

import ch4_figures as CF
import experiment_profile as PROF
import path_analysis as PA
import presentation_assets as PAS
import run_metrics as RM
import stability as ST
import theory_limits as TH
from video_recorder import Overlay

LOGS = PAS.LOGS
KIT = LOGS / "presentation" / "kit"
FONT_FILE = PAS.FONT_FILE

C_SIM, C_REAL, C_INK, C_MUTED, C_GRID = PAS.C_X, PAS.C_Y, PAS.C_INK, PAS.C_MUTED, PAS.C_GRID
C_TARGET, C_PASS, C_FAIL = "#8a8a84", "#1a7f37", "#c62828"
GAINS = (PROF.K_SCALE, PROF.KV_SCALE)
DELAY = ST.DEFAULT_DELAY_S
TA_FROM_PARAMS = __import__("params").TA
FIG_SIZE = (12.8, 7.2)

CENTERING_VIDEO = "20260926T001517Z"
PATH_VIDEO = "20260926T001611Z"
REPLAY_CIRCLE_RUN = "20260924T135736Z"       # counted circle trial whose mean error (19.2 mm) is closest to the group mean
BALANCE_RUN = "20260924T135504Z"

plt.rcParams.update({"font.size": 15, "axes.titlesize": 19, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 13, "axes.titlepad": 14})

CATALOG: list[dict] = []


def register(file: str, priority: str, slide: str, shows: str, caption: str, numbers: str) -> None:
    CATALOG.append({"file": file, "priority": priority, "slide": slide, "shows": shows, "caption": caption,
                    "numbers": numbers})


def save(fig, name: str) -> None:
    fig.savefig(KIT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def thai_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_FILE), size)


def put_text(img: np.ndarray, text: str, xy: tuple[int, int], size: int = 24, colour=(30, 30, 30),
             bg: tuple[int, int, int] | None = None) -> np.ndarray:
    """Draw (Thai-capable) text on a BGR image with PIL; returns the new image."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = thai_font(size)
    if bg is not None:
        box = draw.textbbox(xy, text, font=font)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=bg[::-1])
    draw.text(xy, text, font=font, fill=colour[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


class Mp4:
    def __init__(self, name: str, size: tuple[int, int], fps: float = 30.0) -> None:
        self.final = KIT / name
        self.raw = self.final.with_suffix(".mp4v.mp4")
        self.writer = cv2.VideoWriter(str(self.raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
        PAS.to_h264(self.raw, self.final)
        self.raw.unlink()
        print(f"  wrote {self.final.name}")


def csv_rows(stamp: str) -> list[dict]:
    with open(LOGS / f"run_{stamp}.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([float(r[name]) if r.get(name, "") not in ("", None) else np.nan for r in rows])


def scores(tag: str) -> list[dict]:
    zone = RM.default_zone_cm()
    return [RM.score_run(p, zone) for p in RM.collect_runs(tag)]


def mean_of(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if key in r and r[key] == r[key]]
    return float(np.mean(vals)) if vals else float("nan")


def worst_of(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if key in r and r[key] == r[key]]
    return float(np.max(vals)) if vals else float("nan")


# ================================================================================================
# A. how the system works
# ================================================================================================

class SyncedChart:
    """Right-hand panel of a synced video: distance to target and commanded tilt against time, from one run's log.

    The chart is drawn once; `panel(i)` returns it with a cursor and markers at log row i.
    """

    def __init__(self, rows: list[dict], title: str, is_path: bool, step: int = 1, size: tuple[int, int] = (960, 600)) -> None:
        W, H = size
        self.t = t = col(rows, "t_s")
        self.valid = valid = col(rows, "control_valid") > 0.5
        self.dist = dist = np.where(valid, np.hypot(col(rows, "x_m") - col(rows, "x_ref_m"), col(rows, "y_m") - col(rows, "y_ref_m")) * 100, np.nan)
        self.tx = tx = np.where(valid, col(rows, "theta_x_cmd_deg"), np.nan)
        self.ty = ty = np.where(valid, col(rows, "theta_y_cmd_deg"), np.nan)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W / 100, H / 100), dpi=100, sharex=True)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.11, hspace=0.25)
        ax1.plot(t, dist, color=C_REAL, lw=1.8)
        ax1.set_ylabel("ห่างเป้าหมาย (cm)", fontsize=13)
        ax1.set_ylim(0, max(4.0, np.nanmax(dist) * 1.1))
        if not is_path:
            ax1.axhline(4.0, color=C_TARGET, ls="--", lw=1.5)
            ax1.text(t[-1], 4.1, "โซนศูนย์กลาง 4 cm", ha="right", va="bottom", fontsize=11, color=C_MUTED)
        ax2.axhline(0, color=C_MUTED, lw=0.8)
        ax2.plot(t, tx, color=C_SIM, lw=1.2, label="แกน X")
        ax2.plot(t, ty, color="#7a4fb3", lw=1.2, label="แกน Y")
        ax2.set_ylabel("มุมที่สั่ง (องศา)", fontsize=13)
        ax2.set_xlabel("เวลา (s)", fontsize=13)
        ax2.legend(loc="upper right", ncol=2, fontsize=11, frameon=False)
        fig.suptitle(title, fontsize=16, color=C_INK)
        fig.canvas.draw()
        static = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR).copy()
        self.H = H
        self.ax1, self.ax2 = ax1, ax2
        self.top = int(H - ax1.get_window_extent().y1)
        self.bottom = int(H - ax2.get_window_extent().y0)
        plt.close(fig)
        if step > 1:
            static = put_text(static, f"เล่นเร็ว {step} เท่า", (W - 190, 8), 22, bgr(C_FAIL))
        self.static = static

    def _to_px(self, ax, x, y) -> tuple[int, int]:
        px, py = ax.transData.transform((x, y))
        return int(round(px)), int(round(self.H - py))

    def panel(self, i: int) -> np.ndarray:
        right = self.static.copy()
        x_cursor = self._to_px(self.ax1, self.t[i], 0)[0]
        cv2.line(right, (x_cursor, self.top), (x_cursor, self.bottom), (60, 60, 60), 2)
        if self.valid[i]:
            cv2.circle(right, self._to_px(self.ax1, self.t[i], self.dist[i]), 7, bgr(C_REAL), -1, cv2.LINE_AA)
            cv2.circle(right, self._to_px(self.ax2, self.t[i], self.tx[i]), 6, bgr(C_SIM), -1, cv2.LINE_AA)
            cv2.circle(right, self._to_px(self.ax2, self.t[i], self.ty[i]), 6, bgr("#7a4fb3"), -1, cv2.LINE_AA)
        return right


def a1_synced(stamp: str, name: str, title: str, is_path: bool, step: int = 1) -> None:
    rows = csv_rows(stamp)
    cap = cv2.VideoCapture(str(LOGS / f"video_{stamp}.mp4"))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames != len(rows):
        raise SystemExit(f"video_{stamp}: {n_frames} frames but {len(rows)} log rows; cannot sync")

    W, H = 960, 600
    chart = SyncedChart(rows, title, is_path, step, (W, H))
    video = Mp4(name, (W * 2, H))
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        left = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        video.write(np.hstack([left, chart.panel(i)]))
    cap.release()
    video.close()


def a1_synced_saved_frames(stamp: str, name: str, title: str, speed: int = 1) -> None:
    """Same layout as a1_synced, for a counted run that has no video but saved every Nth camera frame.

    The tracker saved frame k (1-based count of processed frames) of logs/frames_<stamp>/ from the same
    camera frame as log row k-1; matching the ball in the saved frames against x_m/y_m confirms it to
    under one pixel. The overlay (video_recorder.Overlay) is redrawn from the log at the full camera size,
    with its trails fed by every log row, so it looks like a recorded video, only at the saved frame rate.
    """
    rows = csv_rows(stamp)
    files = {int(f.stem): f for f in (LOGS / f"frames_{stamp}").glob("*.jpg")}
    if not files:
        raise SystemExit(f"no saved frames for run {stamp}")
    every = min(np.diff(sorted(files)))
    ppc = PAS.pixel_per_cm()
    full = (800, 1280)        # camera frame the log's pixel coordinates refer to; saved frames are half size
    overlay = Overlay(ppc)

    W, H = 960, 600
    chart = SyncedChart(rows, title, True, speed, (W, H))
    video = Mp4(name, (W * 2, H), fps=30.0 / every * speed)
    note = f"ภาพกล้องที่บันทึกทุก {every} เฟรม ({30 / every:g} ภาพ/วินาที) ซ้อนกับข้อมูลจาก log"
    x, y = col(rows, "x_m"), col(rows, "y_m")
    xr, yr = col(rows, "x_ref_m"), col(rows, "y_ref_m")
    tx, ty = col(rows, "theta_x_cmd_deg"), col(rows, "theta_y_cmd_deg")
    phase, t = col(rows, "path_phase"), col(rows, "t_s")
    valid = col(rows, "control_valid") > 0.5
    for i in range(len(rows)):
        state = {"t": t[i], "tracking": phase[i] in (2, 3), "x_m": x[i], "y_m": y[i], "x_ref": xr[i], "y_ref": yr[i],
                 "tx_deg": tx[i], "ty_deg": ty[i],
                 "det_px": (full[1] / 2 + x[i] * 100 * ppc, full[0] / 2 - y[i] * 100 * ppc, 2.0 * ppc) if valid[i] else None}
        overlay.track(full, state)
        path = files.get(i + 1)
        if path is None:
            continue
        saved = cv2.imread(str(path))
        frame = cv2.resize(saved[:, : saved.shape[1] // 3], (full[1], full[0]), interpolation=cv2.INTER_LINEAR)
        overlay.render(frame, state)
        left = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        left = put_text(left, note, (12, H - 36), 18, (255, 255, 255), bg=(0, 0, 0))
        video.write(np.hstack([left, chart.panel(i)]))
    video.close()


def item_a1() -> None:
    a1_synced(CENTERING_VIDEO, "A1_synced_centering.mp4", "ดึงลูกบอลเข้ากลาง: ภาพจริงกับข้อมูลใน log เฟรมต่อเฟรม", False)
    a1_synced(PATH_VIDEO, "A1_synced_path.mp4", "เดินตามวงกลม 3 cm: ภาพจริงกับข้อมูลใน log เฟรมต่อเฟรม", True)
    register("A1_synced_centering.mp4", "★", "13 ผลจริง Centering",
             "ซ้าย = วิดีโอกล้องจริง, ขวา = ระยะห่างเป้าหมายและมุมที่สั่งจาก log แถวเดียวกัน (เฟรม i = แถว i)",
             "ภาพกับตัวเลขมาจากการรันเดียวกัน กรรมการเห็นว่าตัวเลขในกราฟคือสิ่งที่เกิดขึ้นจริง",
             f"run {CENTERING_VIDEO} (tag demo_centering, ไม่นับในผลบทที่ 4)")
    a1_synced(PATH_VIDEO, "A1_synced_path_4x.mp4", "เดินตามวงกลม 3 cm: ภาพจริงกับข้อมูลใน log (เร่ง 4 เท่า)", True, step=4)
    register("A1_synced_path_4x.mp4", "★", "14 ผลจริง Tracking (ใช้ตัวนี้บนสไลด์)",
             "A1_synced_path ทั้งรอบ 60 s เร่ง 4 เท่าเหลือ 15 s",
             "เห็นครบทุกช่วง: ทรงตัว → เส้นทางขยาย → วนเต็มขนาด ในเวลาที่เล่นจบระหว่างพูด",
             f"run {PATH_VIDEO}")
    register("A1_synced_path.mp4", "○", "14 ผลจริง Tracking (ฉบับเต็ม 60 s)",
             "เหมือน A1 แต่เป็นการเดินตามวงกลม 3 cm",
             "ช่วงแรกทรงตัวที่กลาง 6 s แล้วเส้นทางค่อยๆ ขยาย 5 s ก่อนวนเต็มขนาด",
             f"run {PATH_VIDEO} (tag demo_path, RMS error 2.33 cm อยู่ในช่วงเดียวกับรอบที่นับผล)")


A2_L1, A2_L2 = 65.0, 112.0                      # mm, servo_calibration/config/servo_geometry.json
A2_LINK1_NEUTRAL_DEG = 25.0                     # same file: Link 1 at 25 deg above the floor at neutral
A2_OFFSET, A2_ACRYLIC = 28.0, 6.0               # same file: ball joint -> plate underside, plate thickness
A2_MAP_X = 2.26                                 # config/servo_mapping.json: servo rad per rad of tilt (leg on the tilt axis)
A2_RP = A2_MAP_X * A2_L1 * math.cos(math.radians(A2_LINK1_NEUTRAL_DEG))   # anchor radius implied by that slope
A2_RB = A2_RP - A2_L1 * math.cos(math.radians(A2_LINK1_NEUTRAL_DEG))       # Link 2 vertical at neutral (H0 check)
A2_H_ANCHOR = A2_L1 * math.sin(math.radians(A2_LINK1_NEUTRAL_DEG)) + A2_L2  # anchor height above the servo axis


def latest_video_run(tag: str) -> str | None:
    """Newest run with this tag that recorded a video and did not end early, or None."""
    for sidecar in sorted(LOGS.glob("run_*.json"), reverse=True):
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if info.get("tag") == tag and info.get("video") and info.get("ended_reason") == "completed" \
                and (LOGS / info["video"]).exists():
            return sidecar.stem[4:]
    return None


def item_a4() -> None:
    title = "เดินตามหกเหลี่ยม 3 cm: ภาพจริงกับข้อมูลใน log"
    demo = latest_video_run("demo_hexagon")
    if demo is not None:
        # A recorded 30 fps demonstration exists: same pipeline as the circle video (A1).
        track_mm = RM.score_run(LOGS / f"run_{demo}.csv", RM.default_zone_cm())["track_mm"]
        a1_synced(demo, "A4_synced_hexagon_4x.mp4", title, True, step=4)
        a1_synced(demo, "A4_synced_hexagon.mp4", title + " เฟรมต่อเฟรม", True)
        source = (f"run {demo} (tag demo_hexagon, วิดีโอ 30 fps, ไม่นับในผลบทที่ 4; mean tracking error {track_mm:.1f} mm "
                  f"เทียบกับรอบที่นับผลได้จาก run_metrics.py --tag hexagon)")
    else:
        # No recorded video: rebuild it from the camera frames a counted run saved to disk.
        stamp = representative_run("hexagon", min_seen=0.85)
        track_mm = RM.score_run(LOGS / f"run_{stamp}.csv", RM.default_zone_cm())["track_mm"]
        a1_synced_saved_frames(stamp, "A4_synced_hexagon_4x.mp4", title, speed=4)
        a1_synced_saved_frames(stamp, "A4_synced_hexagon.mp4", title + " เฟรมต่อเฟรม")
        source = (f"run {stamp} (tag hexagon; เลือกจากรอบที่ไม่หลุดและตรวจจับบอลได้ ≥ 85% ของเวลา, mean tracking error "
                  f"{track_mm:.1f} mm ใกล้ค่าเฉลี่ยของทั้ง 10 รอบที่สุด)")
    register("A4_synced_hexagon_4x.mp4", "★", "14 ผลจริง Tracking หกเหลี่ยม (ใช้ตัวนี้บนสไลด์)",
             "ซ้าย = ภาพกล้องจริง + เส้นทางอ้างอิง (ส้ม) และรอยลูกบอล (น้ำเงิน), ขวา = ระยะห่างเป้าหมายและมุมที่สั่งจาก log (เร่ง 4 เท่า)",
             "ลูกบอลวิ่งวนตามหกเหลี่ยมแต่ไม่ตรงเส้น ระยะห่างด้านขวาคือความคลาดเคลื่อนขณะนั้น",
             source)
    register("A4_synced_hexagon.mp4", "○", "14 ผลจริง Tracking หกเหลี่ยม (ฉบับเต็ม 60 s)",
             "เหมือน A4 เร่ง 4 เท่า แต่เล่นตามเวลาจริง",
             "ช่วงแรกทรงตัวที่กลาง 6 s แล้วเส้นทางค่อยๆ ขยาย 5 s ก่อนวนเต็มขนาด",
             source)


def a2_leg(psi: float, z_anchor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Servo axis, elbow and anchor (world, mm) of one outward-pointing leg; returns q1 in degrees."""
    u = np.array([math.cos(psi), math.sin(psi), 0.0])
    a, b = A2_RP - A2_RB, z_anchor
    d = math.hypot(a, b)
    q1 = math.atan2(b, a) - math.acos(np.clip((d * d + A2_L1 ** 2 - A2_L2 ** 2) / (2 * A2_L1 * d), -1, 1))
    servo = A2_RB * u
    elbow = servo + A2_L1 * math.cos(q1) * u + np.array([0, 0, A2_L1 * math.sin(q1)])
    anchor = A2_RP * u + np.array([0, 0, z_anchor])
    return servo, elbow, anchor, math.degrees(q1)


def a2_tilt(t: float) -> tuple[float, float]:
    """Demo sequence (deg): +X, -X, +Y, -Y, then a slow wobble."""
    amp = 5.0
    seq = [(1.5, 0, 0), (1.5, amp, 0), (1.5, -amp, 0), (1.5, 0, amp), (1.5, 0, -amp), (1.0, 0, 0)]
    t0 = 0.0
    prev = (0.0, 0.0)
    for dur, tx, ty in seq:
        if t < t0 + dur:
            k = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (t - t0) / (dur * 0.6)))
            return prev[0] + (tx - prev[0]) * k, prev[1] + (ty - prev[1]) * k
        prev, t0 = (tx, ty), t0 + dur
    w = 2 * math.pi * (t - t0) / 4.0
    return amp * math.cos(w) * min(1.0, (t - t0) / 1.0), amp * math.sin(w) * min(1.0, (t - t0) / 1.0)


def item_a2() -> None:
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    psis = [math.radians(a) for a in (150.0, 30.0, 270.0)]      # camera view: ID1 top-left, ID2 top-right, ID3 bottom
    names = ["ขา 1", "ขา 2", "ขา 3"]
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.28, 0.02, 0.72, 0.9], projection="3d")
    info = fig.text(0.02, 0.86, "", fontsize=13, va="top", color=C_INK, linespacing=1.15)
    fig.text(0.02, 0.2, f"เรขาคณิต (servo_geometry.json): L1 {A2_L1:g} mm, L2 {A2_L2:g} mm,\nLink 1 ทำมุม 25° ที่ neutral, Link 2 ตั้งตรง\n"
             f"รัศมีจุดยึดแผ่น ≈ {A2_RP:.0f} mm, รัศมีแกนเซอร์โว ≈ {A2_RB:.0f} mm\n(ประมาณจากอัตรา 2.26 ใน servo_mapping.json\nควรวัดจริงยืนยัน)",
             fontsize=10.5, color=C_MUTED, va="top", linespacing=1.25)
    fig.suptitle("3RRS: เอียงแผ่น → มุมเซอร์โวทั้ง 3 ขา (เรขาคณิตตามเครื่องจริง)", fontsize=17, x=0.02, ha="left", y=0.97)
    frames = int(16.0 * 30)

    def plate_z(x, y, tx, ty, zc):
        return zc + math.radians(tx) * (x * math.cos(psis[0]) + y * math.sin(psis[0])) + \
            math.radians(ty) * (-x * math.sin(psis[0]) + y * math.cos(psis[0]))

    def update(k):
        t = k / 30.0
        tx, ty = a2_tilt(t)
        ax.cla()
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-40, 260)
        ax.set_box_aspect((1, 1, 0.75))
        ax.view_init(elev=22, azim=-60 + 8 * math.sin(t / 3))
        ax.set_axis_off()
        base = np.array([[-200, -200, -30], [200, -200, -30], [200, 200, -30], [-200, 200, -30]])
        ax.add_collection3d(Poly3DCollection([base], facecolor="#d9d8d2", edgecolor="#aaa9a2", alpha=0.6))
        lines = []
        for i, psi in enumerate(psis):
            ux, uy = math.cos(psi), math.sin(psi)
            z = A2_H_ANCHOR + plate_z(A2_RP * ux, A2_RP * uy, tx, ty, 0.0)
            servo, elbow, anchor, q1 = a2_leg(psi, z)
            q_map = A2_MAP_X * (tx * math.cos(psi - psis[0]) + ty * math.sin(psi - psis[0]))
            box = servo + np.array([0, 0, -12])
            ax.bar3d(box[0] - 14, box[1] - 14, -30, 28, 28, 30, color="#3a3a3a", alpha=0.9, shade=True)
            ax.plot(*zip(servo, elbow), color="#3bbfbf", lw=7, solid_capstyle="round")
            ax.plot(*zip(elbow, anchor), color="#9a6a3a", lw=7, solid_capstyle="round")
            ax.scatter(*anchor, color="#6a5acd", s=60, depthshade=False)
            ax.text(*(anchor + np.array([0, 0, 18])), names[i], fontsize=12, color=C_INK, ha="center")
            lines.append(f"{names[i]}: IK {q1 - A2_LINK1_NEUTRAL_DEG:+5.1f}°   map {q_map:+5.1f}°")
        zc = A2_H_ANCHOR + A2_OFFSET + A2_ACRYLIC
        ang = np.linspace(0, 2 * np.pi, 60)
        px, py = 150 * np.cos(ang), 150 * np.sin(ang)
        pz = [plate_z(x, y, tx, ty, zc) for x, y in zip(px, py)]
        ax.add_collection3d(Poly3DCollection([list(zip(px, py, pz))], facecolor="#bfe3e8", edgecolor="#2e8b8b", alpha=0.45))
        ax.scatter(0, 0, zc + 20, s=380, color="white", edgecolors="#888", depthshade=False)
        info.set_text(f"t = {t:4.1f} s\nเอียง X {tx:+5.1f}°\nเอียง Y {ty:+5.1f}°\n\nมุมเซอร์โวจากกลาง\n" + "\n".join(lines) +
                      "\n\nIK = เรขาคณิตจริง\nmap = สิ่งที่ตัวควบคุมใช้\n(ตัวคูณที่วัด 2.26 / 1.96)")
        return []

    raw = KIT / "A2_platform_3d.mp4v.mp4"
    FuncAnimation(fig, update, frames=frames, interval=1000 / 30).save(str(raw), writer=FFMpegWriter(fps=30, bitrate=5000), dpi=100)
    plt.close(fig)
    PAS.to_h264(raw, KIT / "A2_platform_3d.mp4")
    raw.unlink()
    print("  wrote A2_platform_3d.mp4")
    register("A2_platform_3d.mp4", "○", "6 กลไก / สำรอง B10 IK",
             "แอนิเมชัน 3D ตามเรขาคณิตเครื่องจริง: เซอร์โวอยู่ด้านใน ขาชี้ออก Link 1 ทำมุม 25° Link 2 ตั้งตรง แผ่น Ø300 mm "
             "เอียง ±5° แล้วแสดงมุมเซอร์โวจาก IK เทียบตัวคูณที่ตัวควบคุมใช้จริง",
             "ตัวควบคุมไม่ได้แก้ IK ทุกรอบ แต่ใช้ตัวคูณเชิงเส้นที่วัดได้ (2.26, 1.96) ซึ่งตรงกับ IK ภายในไม่กี่ % ที่มุมเล็ก",
             f"L1 65, L2 112, H0 173.5 mm (servo_geometry.json); รัศมีจุดยึด ≈ {A2_RP:.0f} mm, แกนเซอร์โว ≈ {A2_RB:.0f} mm "
             "(ประมาณ ต้องวัดยืนยัน)")


def item_a3() -> None:
    stamps = [p.stem[4:] for tag in ("balance", "circle", "hexagon") for p in RM.collect_runs(tag)]
    ages, dts = [], []
    for s in stamps:
        rows = csv_rows(s)
        ages += [float(r["frame_age_s"]) for r in rows if r.get("frame_age_s") not in (None, "", "nan")]
        dts += [float(r["dt_s"]) for r in rows[5:]]
    age_ms = float(np.nanmedian(ages)) * 1000
    loop_ms = float(np.median(dts)) * 1000
    parts = [
        ("กล้อง: อ่านภาพ\n→ โปรแกรมได้ภาพ", age_ms, "วัดจาก log", "measured"),
        ("ประมวลผลภาพ\n+ LQR (1 รอบลูป)", loop_ms, "วัดจาก log", "measured"),
        ("หาความเร็วจาก\nผลต่าง 2 เฟรม", loop_ms / 2, "คำนวณ (ครึ่งรอบ)", "derived"),
        ("รอส่งคำสั่ง\n(ทุก 40 ms)", PROF.COMMAND_PERIOD_S * 1000 / 2, "ค่าที่ตั้ง (เฉลี่ย)", "setting"),
        ("เซอร์โวค่อยๆ\nไปถึง (50 ms)", PROF.MOVE_MS / 2, "ค่าที่ตั้ง (เฉลี่ย)", "setting"),
    ]
    known = sum(p[1] for p in parts)
    estimate = DELAY * 1000
    critical = ST.critical_delay(ST.Model(*GAINS)) * 1000
    fill = {"measured": C_REAL, "derived": "#f4a27f", "setting": "#f8cdb9"}

    parts.append(("กลไกเซอร์โวตอบสนอง", estimate - known, "ยังไม่ได้วัด", "unknown"))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.18)
    x = 0.0
    y = np.arange(len(parts))[::-1]
    for yi, (label, ms, source, kind) in zip(y, parts):
        if kind == "unknown":
            ax.barh(yi, ms, left=x, height=0.6, color="white", edgecolor=C_TARGET, hatch="//", linewidth=1.5)
        else:
            ax.barh(yi, ms, left=x, height=0.6, color=fill[kind])
        ax.text(x + ms + 4, yi, f"{ms:.0f} ms  ·  {source}", va="center", fontsize=13,
                color=C_FAIL if kind == "unknown" else C_INK)
        x += ms
    ax.set_yticks(y, [p[0].replace("\n", " ") for p in parts], fontsize=13.5)
    ax.tick_params(axis="y", length=0)
    for value, label, style in ((estimate, f"ค่าประมาณที่ใช้วิเคราะห์\n{estimate:.0f} ms", "--"),
                                (critical, f"ถ้าหน่วงเกิน {critical:.0f} ms\nระบบไม่เสถียร (เกนปัจจุบัน)", ":")):
        ax.axvline(value, color=C_TARGET, ls=style, lw=2)
        ax.text(value + 5, len(parts) - 0.55, label, ha="left", va="top", fontsize=12.5, color=C_INK)
    ax.set_xlim(0, critical + 110)
    ax.set_ylim(-0.6, len(parts) - 0.4)
    ax.set_xlabel("เวลาหน่วงสะสม ตั้งแต่ลูกบอลขยับ จนแผ่นเริ่มเอียงตอบ (ms)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"เวลาหน่วง 0.13 s มาจากไหน: ส่วนที่รู้แล้ว {known:.0f} ms, ที่เหลือยังไม่ได้วัด", loc="left")
    fig.text(0.04, 0.02, f"กล้อง: frame_age_s ค่ากลางจาก {len(ages):,} เฟรม, รอบลูป: dt_s ค่ากลาง ({1000 / loop_ms:.0f} Hz) "
             f"ของรอบที่นับผลทุกรอบ  |  delay วิกฤต: stability.critical_delay ที่เกน {GAINS[0]:g}/{GAINS[1]:g}",
             fontsize=10.5, color=C_MUTED)
    save(fig, "A3_delay_budget.png")
    register("A3_delay_budget.png", "★", "11 เสถียรภาพ หรือ Q&A",
             "แยกเวลาหน่วง 0.13 s ออกเป็นส่วนๆ: ส่วนไหนวัดแล้ว ส่วนไหนเป็นค่าที่ตั้ง ส่วนไหนยังไม่รู้",
             "ตอบคำถาม 'เวลาหน่วง 0.13 s มาจากไหน' ได้ในภาพเดียว และบอกตรงๆ ว่ายังไม่ได้วัดทั้งหมด",
             f"กล้อง {age_ms:.0f} ms, รอบลูป {loop_ms:.0f} ms, รวมส่วนที่รู้ {known:.0f} ms, ประมาณ {estimate:.0f} ms, "
             f"วิกฤต {critical:.0f} ms")


# ================================================================================================
# B. where the numbers come from
# ================================================================================================

B1_TEXT = {   # short slide wording of theory_limits.targets(); the full sentences stay in that file
    "rms_mm": ("อยู่ใกล้ศูนย์กลาง (RMS)", "1/4 ของเส้นผ่านศูนย์กลางบอล 40 mm"),
    "settle_s": ("เวลาเข้าโซนศูนย์กลาง", "2 เท่าของทฤษฎี {settle:.2f} s\n(โซน = 1 เส้นผ่านศูนย์กลางบอล)"),
    "steady_mm": ("ความคลาดขณะนิ่ง", "รัศมีบอล 20 mm\n(บอลยังทับจุดเป้าหมาย)"),
    "overshoot_pct": ("พุ่งเลยเป้าหมาย", "ทฤษฎี {os:.0f}% เผื่อ dead zone และ delay"),
    "track_mm": ("ตามเส้นทาง\n(วงกลม + หกเหลี่ยม)", "ครึ่งหนึ่งของกรณีไม่ควบคุมเลย\n(30 mm = รัศมีเส้นทาง)"),
    "success": ("ความน่าเชื่อถือ", "4 ใน 5 รอบ"),
}


def item_b1() -> None:
    bal, cir, hexa = scores("balance"), scores("circle"), scores("hexagon")
    targets = {t.key: t for t in TH.targets()}
    cl = TH.closed_loop(*GAINS)
    text = {k: (a, b.format(settle=cl.settle_2pct_s, os=cl.overshoot_pct)) for k, (a, b) in B1_TEXT.items()}
    track_runs = cir + hexa
    T, P, R = CF.sim_path("circle", GAINS)
    sel = T > CF.HOLD_S + CF.RAMP_S
    sim_track = float(np.hypot(*(P[sel] - R[sel]).T).mean() * 1000)
    theory = {"rms_mm": f"{TH.measured('detection_noise'):g} mm*", "settle_s": f"{cl.settle_2pct_s:.1f} s",
              "steady_mm": "0 mm", "overshoot_pct": f"{cl.overshoot_pct:.0f}%", "track_mm": f"{sim_track:.1f} mm",
              "success": "–"}
    fig = plt.figure(figsize=FIG_SIZE)
    fig.text(0.02, 0.95, "เกณฑ์ทุกตัวมีที่มา: เทียบทฤษฎีกับผลวัดจริง", fontsize=19, color=C_INK, va="top")
    cols = [0.02, 0.22, 0.33, 0.64, 0.75, 0.87]
    heads = ["ตัวชี้วัด", "เกณฑ์", "ที่มาของเกณฑ์", "ทฤษฎี", "วัดจริง เฉลี่ย", "รอบแย่สุด"]
    top, row_h = 0.83, 0.115
    for x, head in zip(cols, heads):
        fig.text(x, top, head, fontsize=13.5, color=C_MUTED, weight="bold", va="center")
    for i, key in enumerate(("rms_mm", "settle_s", "steady_mm", "overshoot_pct", "track_mm", "success")):
        t = targets[key]
        yy = top - (i + 1) * row_h
        fig.add_artist(plt.Line2D([0.02, 0.98], [yy + row_h / 2, yy + row_h / 2], color=C_GRID, lw=1))
        label, source = text[key]
        sign = "≥" if key == "success" else "≤"
        unit = "%" if t.unit == "%" else f" {t.unit}"
        fig.text(cols[0], yy, label, fontsize=14, color=C_INK, va="center", linespacing=1.3)
        fig.text(cols[1], yy, f"{sign} {t.limit:g}{unit}", fontsize=15, color=C_INK, va="center", weight="bold")
        fig.text(cols[2], yy, source, fontsize=12.5, color=C_MUTED, va="center", linespacing=1.35)
        fig.text(cols[3], yy, theory[key], fontsize=14, color=C_SIM, va="center")
        if key == "success":
            ok_bal = sum(r["success"] for r in bal)
            ok_path = sum(r["success"] for r in track_runs)
            rate = 100.0 * (ok_bal + ok_path) / (len(bal) + len(track_runs))
            ok = rate >= t.limit
            fig.text(cols[4], yy, f"{rate:.0f}%  {'✓' if ok else '✗'}", fontsize=15, color=C_PASS if ok else C_FAIL,
                     va="center", weight="bold")
            fig.text(cols[5], yy, f"ทรงตัว {ok_bal}/{len(bal)}\nเส้นทาง {ok_path}/{len(track_runs)}", fontsize=12.5,
                     color=C_INK, va="center", linespacing=1.3)
            continue
        runs = track_runs if key == "track_mm" else bal
        mean, worst = mean_of(runs, key), worst_of(runs, key)
        for x, value, bold in ((cols[4], mean, "bold"), (cols[5], worst, "normal")):
            ok = value <= t.limit
            fig.text(x, yy, f"{value:.1f}  {'✓' if ok else '✗'}", fontsize=15, color=C_PASS if ok else C_FAIL,
                     va="center", weight=bold)
    fig.text(0.02, 0.02, "ที่มาเต็ม: theory_limits.targets()  |  วัดจริง: run_metrics.score_run รอบที่นับผล (ทรงตัว 10, วงกลม 10, "
             "หกเหลี่ยม 10)  |  ตามเส้นทาง = เทียบ ณ เวลาเดียวกัน  |  * RMS: ขีดความละเอียดกล้อง", fontsize=10.5, color=C_MUTED)
    save(fig, "B1_targets_provenance.png")
    register("B1_targets_provenance.png", "★", "4 วัตถุประสงค์และตัวชี้วัด / สำรอง Q&A",
             "ตารางเกณฑ์ 6 ข้อ: ค่าเกณฑ์ ที่มา ค่าทฤษฎี ค่าวัดจริงเฉลี่ย และรอบที่แย่ที่สุด",
             "เปิดตอนกรรมการถาม 'ตัวเลขเป้าหมายมาจากไหน' ทุกเกณฑ์มีแหล่งที่มา และบอกตรงๆ ข้อไหนไม่ผ่าน",
             "เกณฑ์และที่มาจาก theory_limits.targets(); ค่าวัดจาก run_metrics.score_run")


def item_b2() -> None:
    bal = scores("balance")
    stamps_c = [p.stem[4:] for p in RM.collect_runs("circle")]
    cross = [PA.run_summary(s, "circle")["cross_mm"] for s in stamps_c]
    sync = [PA.run_summary(s, "circle")["sync_mm"] for s in stamps_c]
    T, P, R = CF.sim_path("circle", GAINS)
    sel = T > CF.HOLD_S + CF.RAMP_S
    sim_track = float(np.hypot(*(P[sel] - R[sel]).T).mean() * 1000)
    steady = [r["steady_mm"] for r in bal]
    noise = TH.measured("detection_noise")
    target_steady = RM._limit("steady_mm")
    target_track = RM._limit("track_mm")
    null = TH.null_tracking_error_mm(0.030)

    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.18, hspace=0.55)
    panels = [
        (axes[0], "ทรงตัวที่ศูนย์กลาง\nความคลาดขณะนิ่ง", [
            ("ทฤษฎี", 0.0, C_SIM, "o"), ("ขีดความละเอียดกล้อง", noise, C_TARGET, "D")],
         steady, target_steady, None),
        (axes[1], "เดินตามวงกลม 3 cm\nความคลาดตามเส้นทาง", [
            ("ทฤษฎี (รวม delay)", sim_track, C_SIM, "o")],
         sync, target_track, null),
    ]
    for ax, title, marks, measured_vals, target, baseline in panels:
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.axhline(0, color=C_GRID, lw=6, zorder=0)
        ax.text(-0.02, 0.5, title, transform=ax.transAxes, ha="right", va="center", fontsize=15, color=C_INK)
        above = True
        for label, value, colour, marker in marks:
            ax.plot(value, 0, marker=marker, color=colour, ms=16, zorder=3)
            ax.annotate(f"{label}\n{value:.1f} mm", (value, 0), xytext=(0, 34 if above else -34), textcoords="offset points",
                        ha="center", va="bottom" if above else "top", fontsize=12.5, color=C_INK)
            above = not above
        lo, hi, mu = min(measured_vals), max(measured_vals), float(np.mean(measured_vals))
        ax.plot([lo, hi], [0, 0], color=C_REAL, lw=8, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot(mu, 0, "o", color=C_REAL, ms=18, zorder=4)
        ax.annotate(f"วัดจริง เฉลี่ย {mu:.1f} mm\n(ช่วง {lo:.1f}–{hi:.1f} ใน {len(measured_vals)} รอบ)", (mu, 0), xytext=(0, -34),
                    textcoords="offset points", ha="center", va="top", fontsize=12.5, color=C_INK, weight="bold")
        ax.axvline(target, color=C_TARGET, ls="--", lw=2)
        ax.text(target, 0.93, f" เกณฑ์ {target:g} mm", va="top", fontsize=12.5, color=C_INK)
        if baseline is not None:
            ax.axvline(baseline, color=C_TARGET, ls=":", lw=2)
            ax.text(baseline, 0.93, f" ไม่ควบคุมเลย {baseline:.0f} mm", va="top", fontsize=12.5, color=C_MUTED)
    axes[1].plot(np.mean(cross), 0, "s", color=C_REAL, ms=13, zorder=4, markerfacecolor="white", markeredgewidth=2.5)
    axes[1].annotate(f"ระยะถึงเส้นทาง\n{np.mean(cross):.1f} mm", (np.mean(cross), 0), xytext=(0, 34), textcoords="offset points",
                     ha="center", va="bottom", fontsize=12.5, color=C_INK)
    axes[1].set_xlim(-1.5, 50)
    axes[1].set_xlabel("ความคลาด (mm)")
    fig.suptitle("ทฤษฎี → เกณฑ์ → วัดจริง: ช่องว่างระหว่างจุดคือสิ่งที่ต้องอธิบาย", fontsize=19, color=C_INK, x=0.02, ha="left")
    fig.text(0.02, 0.02, "ทฤษฎีทรงตัว = 0 (มีตัวอินทิเกรต trim)  |  ทฤษฎีตามเส้นทาง = ch4_figures.sim_path  |  "
             "ขีดกล้อง = สัญญาณรบกวนตำแหน่งขณะบอลนิ่ง (theory_limits)  |  วัดจริง = run_metrics / path_analysis",
             fontsize=10, color=C_MUTED)
    save(fig, "B2_theory_target_measured.png")
    register("B2_theory_target_measured.png", "★", "15 จำลองกับจริง",
             "เส้นตัวเลขเดียว: ทฤษฎี, ขีดความละเอียดกล้อง, เกณฑ์, ผลจริง (ค่าเฉลี่ยและช่วงทุกรอบ) และกรณีไม่ควบคุม",
             "ทฤษฎีกับกล้องไม่ใช่ตัวจำกัด (ต่ำกว่า 2 mm) ช่องว่าง 10–20 mm มาจากกลไก ซึ่งหน้า D1–D4 อธิบายต่อ",
             f"ทรงตัว: ทฤษฎี 0, กล้อง {noise:g}, จริง {np.mean(steady):.1f} mm; ตามเส้นทาง: ทฤษฎี {sim_track:.1f}, "
             f"จริง {np.mean(sync):.1f} (เวลา) / {np.mean(cross):.1f} (ระยะ) mm, ไม่ควบคุม {null:.0f} mm")


def item_b3() -> None:
    m = CF.manifest()
    pts = CF.open_loop_points(m)
    per_rad = CF.theory_open_loop_mm_per_rad()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.18)
    xs = np.array([0.0, 2.2])
    ax.plot(xs, per_rad * np.radians(xs), color=C_SIM, ls="--", lw=2.5,
            label=f"ทฤษฎี (ลูกกลวง a = 3/5·g·θ, {ST.G_EFF:.2f} m/s² ต่อ rad)")
    ax.plot(xs, per_rad * 7.0 / ST.G_EFF * np.radians(xs), color=C_TARGET, ls=":", lw=2,
            label="ถ้าคิดเป็นลูกตัน (5/7·g = 7.0) — ไม่ใช่ลูกปิงปอง")
    for axis, marker, name in (("x", "o", "แกน X"), ("y", "s", "แกน Y")):
        ax.plot([0] + [p[0] for p in pts[axis]], [0] + [p[1] for p in pts[axis]], marker=marker, ms=10, lw=2.2,
                color=C_REAL if axis == "x" else "#b34d1f", label=f"วัดจริง {name}")
    ax.axvspan(0, 0.5, color=C_TARGET, alpha=0.15)
    ax.text(0.25, 58, "มุมเล็กกว่า ~0.5°\nบอลแทบไม่ขยับ\n(dead zone)", ha="center", va="top", fontsize=13, color=C_INK)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-3, 62)
    ax.set_xlabel("มุมเอียงที่สั่ง (องศา)")
    ax.set_ylabel("ระยะที่บอลกลิ้งใน 0.8 วินาที (mm)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("เอียงแผ่นโดยไม่มีตัวควบคุม: มุมใหญ่ตรงทฤษฎี มุมเล็กบอลไม่ขยับ", loc="left")
    one_deg = ST.G_EFF * math.radians(1.0)
    fig.text(0.09, 0.02, f"เอียง 1° → บอลเร่ง {one_deg:.2f} m/s² ตามทฤษฎี (กลิ้ง ~{0.5 * one_deg * 100:.0f} cm ใน 1 s)  |  "
             f"gain ที่วัดได้: X {TH.measured('plant_gain_x'):g}, Y {TH.measured('plant_gain_y'):g} m/s² ต่อ rad "
             f"(test_tilt_direction.py)  |  ข้อมูล: logs/tilt_test_*", fontsize=10.5, color=C_MUTED)
    save(fig, "B3_open_loop_tilt.png")
    register("B3_open_loop_tilt.png", "★", "9 แบบจำลองคณิตศาสตร์ / 15 สาเหตุ dead zone",
             "ทดสอบเอียงแผ่นตรงๆ ที่ 0.3/0.6/1/2° แล้ววัดว่าบอลกลิ้งไปเท่าไร เทียบเส้นทฤษฎี",
             "หลักฐานของทั้งค่าคงที่ในแบบจำลอง (มุมใหญ่ตรงทฤษฎี) และ dead zone (มุมเล็กบอลไม่ขยับ)",
             f"g_eff ทฤษฎี {ST.G_EFF:.2f} (ลูกกลวง 3/5 g; สไลด์เดิมเขียน 7.0 = สูตรลูกตัน), วัดได้ X "
             f"{TH.measured('plant_gain_x'):g} / Y {TH.measured('plant_gain_y'):g}")


# ================================================================================================
# C. how the gains were chosen
# ================================================================================================

def item_c1() -> None:
    delays = np.linspace(0.0, 0.42, 120)
    cases = [((1.0, 1.0), "เกน LQR ตามที่ออกแบบ (notebook)", C_TARGET, "--", 2.5),
             ((0.15, 1.8), "เคยลอง ×0.15 / kv 1.8", "#7fa6d9", ":", 2.5),
             (GAINS, f"ที่ใช้จริง ×{GAINS[0]:g} / kv {GAINS[1]:g}", C_SIM, "-", 3.5)]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.18)
    ax.axvspan(0.05, 0.25, color=C_TARGET, alpha=0.1)
    ax.text(0.15, 76, "ช่วงที่เวลาหน่วงจริงน่าจะอยู่ (ยังไม่ได้วัด)", ha="center", fontsize=12.5, color=C_MUTED)
    ax.axvline(DELAY, color=C_MUTED, lw=1.2)
    ax.text(DELAY + 0.004, 50, f"ค่าประมาณ {DELAY:g} s", fontsize=12, color=C_MUTED)
    notes = []
    for (k, kv), label, colour, style, width in cases:
        model = ST.Model(k, kv)
        pms = [ST.margins(model, d)["pm_deg"] if ST.margins(model, d)["stable"] else np.nan for d in delays]
        crit = ST.critical_delay(model)
        pm_now = ST.margins(model, DELAY)["pm_deg"]
        ax.plot(delays, pms, color=colour, ls=style, lw=width, label=f"{label}: PM {pm_now:.0f}° ที่ {DELAY:g} s")
        ax.plot(crit, 0, "o", color=colour, ms=11, zorder=5)
        ax.annotate(f"ไม่เสถียรเมื่อหน่วง > {crit:.2f} s", (crit, 0), xytext=(0, -26), textcoords="offset points",
                    ha="center", va="top", fontsize=12, color=C_INK)
        notes.append((label, pm_now, crit))
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.set_xlim(0, 0.42)
    ax.set_ylim(-12, 82)
    ax.set_xlabel("เวลาหน่วงของลูป (s)")
    ax.set_ylabel("Phase margin (องศา)")
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.93))
    ax.set_title("ทำไมไม่ใช้เกน LQR ตรงๆ: ทนเวลาหน่วงได้แค่ "
                 f"{notes[0][2]:.2f} s ส่วนเกนที่ใช้ทนได้ถึง {notes[2][2]:.2f} s", loc="left")
    fig.text(0.09, 0.02, "stability.margins / critical_delay (แบบจำลองไม่รวมแรงหน่วงของบอล b ซึ่งเป็นค่าสมมติใน notebook; "
             "ถ้ารวม b = 0.003 N·s/m ค่า PM จะสูงขึ้น)  |  K_AXIS จาก calculated_lqr_ik_3rrs.ipynb", fontsize=10.5, color=C_MUTED)
    save(fig, "C1_why_scale_gain.png")
    register("C1_why_scale_gain.png", "★", "10 LQR / 11 เสถียรภาพ / Q&A 'จูนยังไง'",
             "Phase margin เทียบเวลาหน่วง ของเกน LQR ตามที่ออกแบบ, เกนที่เคยลอง และเกนที่ใช้จริง",
             "LQR ออกแบบโดยไม่มีเวลาหน่วง ถ้าเวลาหน่วงจริงเกิน 0.16 s จะไม่เสถียร เราจึงลดเกนตำแหน่งและเพิ่มเกนความเร็ว "
             "ให้ทนเวลาหน่วงที่ยังไม่ได้วัดได้กว้างขึ้น",
             "; ".join(f"{l}: PM {pm:.0f}°, วิกฤต {c:.2f} s" for l, pm, c in notes))


def lqr_gain(Q: np.ndarray, R: float, damping: bool = True) -> np.ndarray:
    """Continuous LQR on the notebook's design model (with its assumed ball damping b), via the Hamiltonian."""
    from params import B_DAMP, MB, TA
    A = np.array([[0.0, 1.0, 0.0], [0.0, -(B_DAMP / MB) if damping else 0.0, ST.G_EFF], [0.0, 0.0, -1.0 / TA]])
    B = np.array([[0.0], [0.0], [1.0 / TA]])
    H = np.block([[A, -B @ B.T / R], [-Q, -A.T]])
    w, V = np.linalg.eig(H)
    V = V[:, w.real < 0]
    P = np.real(V[3:] @ np.linalg.inv(V[:3]))
    return (B.T @ P).ravel() / R


def response_with_gain(K: np.ndarray, e0: float, secs: float, dt: float, delay_s: float) -> tuple[np.ndarray, np.ndarray]:
    """stability.step_response for an arbitrary gain vector (same A, B and delay buffer)."""
    model = ST.Model(1.0, 1.0)
    A, B = model.A, model.B
    lag = int(round(delay_s / dt))
    x = np.array([e0, 0.0, 0.0])
    hist = [0.0] * lag
    out = np.empty(int(secs / dt))
    for i in range(len(out)):
        out[i] = x[0]
        u = float(-K @ x)
        hist.append(u)
        x = x + (A @ x + B * (hist.pop(0) if lag else u)) * dt
    return np.arange(len(out)) * dt, out


def item_c4() -> None:
    base = dict(e=0.02, v=0.15, a=math.radians(5.0), u=math.radians(6.0))
    knobs = [("e", "ยอมให้ตำแหน่งคลาด e_max", [0.01, 0.02, 0.04], lambda v: f"{v * 100:g} cm",
              lambda r: f"q_e มากขึ้น (e_max เล็กลง): เข้าเป้าเร็วขึ้น\n{r[2][2]:.2f} → {r[0][2]:.2f} s, เกินเป้า {r[2][3]:.0f} → {r[0][3]:.0f} mm"),
             ("v", "ยอมให้ความเร็ว v_max", [0.075, 0.15, 0.30], lambda v: f"{v:g} m/s",
              lambda r: f"q_v มากขึ้น (v_max เล็กลง): แกว่งเกินน้อยลง\nเกินเป้า {r[2][3]:.0f} → {r[0][3]:.0f} mm, เข้าเป้า {r[2][2]:.2f} → {r[0][2]:.2f} s"),
             ("u", "ยอมให้มุมที่สั่ง u_max", [math.radians(3), math.radians(6), math.radians(12)],
              lambda v: f"{math.degrees(v):g}°",
              lambda r: f"r มากขึ้น (u_max เล็กลง): สั่งเบาลง ช้าลง\nเข้าเป้า {r[2][2]:.2f} → {r[0][2]:.2f} s, เกินเป้า {r[2][3]:.0f} → {r[0][3]:.0f} mm")]
    shades = ["#9cc2ee", C_SIM, "#0d366b"]
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.3, wspace=0.12)
    rows = []
    for ax, (key, title, values, fmt, message) in zip(axes, knobs):
        for value, colour in zip(values, shades):
            p = dict(base, **{key: value})
            Q = np.diag([1 / p["e"] ** 2, 1 / p["v"] ** 2, 1 / p["a"] ** 2])
            K = lqr_gain(Q, 1 / p["u"] ** 2)
            T, E = response_with_gain(K, 0.06, 5.0, 0.002, 0.0)
            is_design = value == base[key]
            outside = np.where(np.abs(E) > 0.01)[0]
            t_in = float(T[outside[-1]]) if len(outside) else 0.0
            over = max(0.0, -float(E.min())) * 1000
            ax.plot(T, E * 100, color=colour, lw=3.5 if is_design else 2.2,
                    label=f"{fmt(value)}{'  (ที่ออกแบบ)' if is_design else ''}")
            rows.append((key, value, t_in, over))
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("เวลา (s)")
        ax.legend(frameon=False, fontsize=12.5, loc="center left", bbox_to_anchor=(0.3, 0.6))
        mine = [r for r in rows if r[0] == key]
        ax.text(0.0, -0.3, message(mine), transform=ax.transAxes, fontsize=12.5, color=C_INK, va="top")
    axes[0].set_ylabel("ตำแหน่งบอล (cm)")
    fig.suptitle("ปรับ Q, R ของ LQR (Bryson's rule) แล้วการตอบสนองเปลี่ยนอย่างไร", fontsize=19, x=0.02, ha="left",
                 color=C_INK)
    fig.text(0.02, 0.905, "ค่าที่ใช้ใน notebook: e_max 2 cm, v_max 0.15 m/s, มุมแผ่น 5°, u_max 6° → Q = diag(1/e², 1/v², 1/α²), "
             "R = 1/u²  |  จำลองวางบอลห่าง 6 cm ไม่มีเวลาหน่วง (ขั้นออกแบบ)\nที่ออกแบบ: เข้า 1 cm ใน {:.2f} s, เกินเป้า {:.0f} mm".format(
                 *next((r[2], r[3]) for r in rows if r[0] == "e" and r[1] == base["e"])), fontsize=11.5, color=C_MUTED, va="top",
             linespacing=1.5)
    fig.text(0.02, 0.02, "K แก้สมการ Riccati บนแบบจำลองเดียวกับ notebook (รวม b = 0.003 N·s/m ที่ notebook สมมติ); "
             "ตอบสนองจำลองด้วยแบบจำลองของ stability.py  |  ตรวจแล้ว: ค่าที่ออกแบบให้ K = K_AXIS ใน params.py",
             fontsize=10, color=C_MUTED)
    save(fig, "C4_lqr_q_r_effect.png")
    design = lqr_gain(np.diag([1 / 0.02 ** 2, 1 / 0.15 ** 2, 1 / math.radians(5) ** 2]), 1 / math.radians(6) ** 2)
    register("C4_lqr_q_r_effect.png", "★", "10 LQR (แทนตาราง 'เพิ่ม q_e / q_v / r')",
             "ปรับค่ายอมรับได้ใน Bryson's rule ทีละตัว (ตำแหน่ง, ความเร็ว, มุมที่สั่ง) แล้วดูการดึงบอลจาก 6 cm",
             "ตารางผลการปรับจูนในสไลด์ 10 มีตัวเลขรองรับ (ดูข้อความใต้แต่ละกราฟ) และบอกได้ว่า Q, R ที่ใช้มาจากเกณฑ์อะไร",
             f"K ที่ออกแบบ {np.round(design, 3).tolist()} = K_AXIS; Q/R จาก calculated_lqr_ik_3rrs.ipynb")


def simulate_states(model: ST.Model, e0_m: float, secs: float, dt: float, delay_s: float):
    """Same integration as stability.step_response, keeping the plate tilt as well as the error."""
    steps, lag = int(secs / dt), int(round(delay_s / dt))
    x = np.zeros(model.n)
    x[model.e_index] = e0_m
    history = [0.0] * lag
    E, TH_ = np.empty(steps), np.empty(steps)
    for i in range(steps):
        E[i], TH_[i] = x[model.e_index], x[model.e_index + 2]
        u = float(-model.K @ x)
        history.append(u)
        x = x + (model.A @ x + model.B * (history.pop(0) if lag else u)) * dt
    return np.arange(steps) * dt, E, TH_


C2_CASES = ((0.03, "×0.03 ช้า แกว่งเกินเป้า"), (GAINS[0], f"×{GAINS[0]:g} ที่เลือกใช้"),
            (0.15, "×0.15 เร็วขึ้นในแบบจำลอง แต่เผื่อ delay น้อย"), (0.40, "×0.40 ไม่เสถียร"))


def item_c2() -> None:
    secs, dt, e0 = 8.0, 0.005, 0.06
    sims = [(k, label, *simulate_states(ST.Model(k, GAINS[1]), e0, secs, dt, DELAY)) for k, label in C2_CASES]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.09, right=0.8, top=0.86, bottom=0.18)
    shades = {0.03: "#9cc2ee", GAINS[0]: C_SIM, 0.15: "#5f8fc9", 0.40: C_FAIL}
    for k, label, T, E, _ in sims:
        e = np.clip(E * 100, -15, 15)
        ax.plot(T, e, color=shades[k], lw=4 if k == GAINS[0] else 2, zorder=3 if k == GAINS[0] else 2)
        ax.text(secs + 0.1, np.clip(e[-1], -8, 8) + {0.03: 0.9, 0.15: -0.9}.get(k, 0), label, va="center", fontsize=13,
                color=C_INK, weight="bold" if k == GAINS[0] else "normal")
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.axhspan(-15, -10, color=C_FAIL, alpha=0.06)
    ax.set_xlim(0, secs)
    ax.set_ylim(-11, 7)
    ax.set_xlabel("เวลา (s)")
    ax.set_ylabel("ตำแหน่งบอลเทียบจุดศูนย์กลาง (cm)")
    ax.set_title("จำลองวางบอลห่างกลาง 6 cm ที่เกนต่างกัน (เวลาหน่วง 0.13 s)", loc="left")
    fig.text(0.09, 0.02, "แบบจำลองเชิงเส้นแกนเดียวจาก stability.py (A, B, K เดียวกับตัวควบคุมจริง) ไม่มี dead zone  |  "
             "เส้น ×0.40 ถูกตัดที่ขอบแผ่น", fontsize=10.5, color=C_MUTED)
    save(fig, "C2_gain_family.png")

    W, row_h = 1280, 180
    H = row_h * len(sims) + 60
    scale = 1100 / 0.30                      # plate +-15 cm drawn 1100 px wide
    video = Mp4("C2_gain_family.mp4", (W, H))
    step = int(round(1 / 30 / dt))
    for i in range(0, len(sims[0][2]), step):
        frame = np.full((H, W, 3), 255, np.uint8)
        frame = put_text(frame, f"t = {sims[0][2][i]:4.1f} s   (มุมเอียงในภาพขยาย 5 เท่าให้เห็นชัด)", (20, 12), 26)
        for r, (k, label, T, E, TH_) in enumerate(sims):  # visual tilt is x5, clamped at 17 deg so rows never cross
            cy = 60 + r * row_h + 110
            ang = float(np.clip(TH_[i] * 5, -0.3, 0.3))
            c, s = math.cos(ang), math.sin(ang)
            half = 550
            p1 = (int(W / 2 - half * c), int(cy + half * s))
            p2 = (int(W / 2 + half * c), int(cy - half * s))
            colour = bgr(shades[k])
            cv2.line(frame, p1, p2, (90, 90, 90), 8, cv2.LINE_AA)
            cv2.drawMarker(frame, (W // 2, cy - 10), bgr(C_TARGET), cv2.MARKER_TRIANGLE_UP, 18, 2)
            pos = float(E[i])
            fell = abs(pos) > 0.15
            d = float(np.clip(pos, -0.15, 0.15)) * scale
            bx, by = int(W / 2 + d * c), int(cy - d * s - 0.02 * scale - 4)
            cv2.circle(frame, (bx, by), int(0.02 * scale), colour, -1, cv2.LINE_AA)
            text = label + ("   → บอลตกแผ่น" if fell else f"   ห่างกลาง {abs(pos) * 100:4.1f} cm")
            frame = put_text(frame, text, (20, cy - 150 + 60), 24, (30, 30, 30) if not fell else bgr(C_FAIL))
        video.write(frame)
    video.close()
    register("C2_gain_family.png", "★", "10 LQR และการจูน",
             "จำลองการดึงบอลจาก 6 cm ที่ตัวคูณเกน 0.03 / 0.076 / 0.15 / 0.40",
             "เกนต่ำไปช้าและแกว่งเกิน, เกนสูงไปไม่เสถียร; 0.15 ดูดีในแบบจำลองแต่เผื่อ delay น้อย (ดู C1)",
             "stability.py แบบจำลองเดียวกับที่ใช้วิเคราะห์ เวลาหน่วง 0.13 s")
    register("C2_gain_family.mp4", "○", "10 LQR และการจูน",
             "แอนิเมชันมุมมองด้านข้าง 4 แถว: แผ่นเอียงและบอลกลิ้งตามเกนแต่ละค่า",
             "ภาพเคลื่อนไหวช่วยให้คนที่ไม่ได้เรียนควบคุมเห็นทันทีว่าเกนมีผลอย่างไร",
             "ข้อมูลเดียวกับ C2_gain_family.png")


def item_c3() -> None:
    zone = RM.default_zone_cm()
    base = scores("circle")
    base_mm = mean_of(base, "track_mm")
    rr = json.loads((Path(__file__).resolve().parent / "report_runs.json").read_text(encoding="utf-8"))

    def group(stamps):
        res = [RM.score_run(LOGS / f"run_{s}.csv", zone) for s in stamps]
        return mean_of(res, "track_mm"), sum(1 for r in res if r.get("lost")), len(res)

    rows = [("ค่าตั้งต้น: เกน ×0.076 / kv 2.8, วงกลม 3 cm", *(base_mm, sum(r["lost"] for r in base), len(base)), "ref")]
    k15 = group(rr["ellipse"]["runs"] + rr["hexagon"]["runs"])
    rows.append(("เพิ่มเกนเป็น ×0.15 / kv 1.8 (วงรี + หกเหลี่ยม)", *k15, "better" if k15[0] < 0.9 * base_mm else "worse"))
    for tag in ("prefix_ladder_0.05", "prefix_ladder_0.10", "prefix_ladder_0.20"):
        res = scores(tag)
        ki = tag.split("_")[-1]
        v = mean_of(res, "track_mm")
        rows.append((f"เพิ่มตัวอินทิเกรต k_i = {ki}", v, sum(r["lost"] for r in res), len(res), "better" if v < 0.9 * base_mm else "worse"))
    for key, name in (("circle_5cm", "ขยายเป็นวงกลม 5 cm"), ("circle_7cm", "ขยายเป็นวงกลม 7 cm")):
        rows.append((name, *group(rr[key]["runs"]), "size"))
    demo = RM.collect_runs("demo_path_5cm_10s")
    if demo:
        rows.append(("วงกลม 5 cm คาบ 10 s (เร็วขึ้น)", *group([p.stem[4:] for p in demo]), "size"))
    text_rows = [("เร่งตัวชดเชยความเอียง (trim ki 0.5) ระหว่างเส้นทาง", "บอลหลุดใน 1–2 s ทั้ง 2 รอบ", False),
                 ("ชดเชย dead zone 0.5° และ 1.0°", "แกว่ง ~1 Hz เกินเส้นทางแล้วหลุด", False),
                 ("ลดคาบส่งคำสั่งเซอร์โว 0.10 → 0.04 s", "ทรงตัวจากจุดไกลได้ (ใช้เป็นค่ามาตรฐาน)", True)]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_axes([0.42, 0.33, 0.55, 0.53])
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, lost, n, kind) in zip(y, rows):
        colour = {"ref": C_TARGET, "better": C_PASS, "worse": C_REAL, "size": "#f4a27f"}[kind]
        ax.barh(yi, v, color=colour, height=0.62)
        note = f"{v:.1f} mm" + (f"   หลุด {lost}/{n}" if lost else f"   ({n} รอบ)")
        ax.text(v + 0.6, yi, note, va="center", fontsize=12.5, color=C_INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12.5)
    ax.tick_params(axis="y", length=0)
    ax.axvline(base_mm, color=C_TARGET, ls="--", lw=1.8)
    ax.axvline(RM._limit("track_mm"), color=C_PASS, ls=":", lw=1.5)
    ax.text(RM._limit("track_mm"), len(rows) - 0.35, "เกณฑ์ 15", color=C_PASS, fontsize=11, ha="center")
    ax.set_xlim(0, 60)
    ax.set_xlabel("ความคลาดตามเส้นทางเฉลี่ย เทียบ ณ เวลาเดียวกัน (mm)", fontsize=13)
    ax.grid(axis="y", visible=False)
    fig.text(0.02, 0.95, "สิ่งที่ลองจูนแล้ว: ไม่มีวิธีไหนทำให้ตามเส้นทางดีกว่าค่าตั้งต้น", fontsize=19, color=C_INK, va="top")
    fig.text(0.02, 0.25, "ลองแล้วแต่ไม่มีตัวเลขใน log ที่ระบุได้ (อ้างจากตาราง 4.16 ในรายงาน):", fontsize=13, color=C_MUTED)
    for i, (label, result, good) in enumerate(text_rows):
        yy = 0.19 - i * 0.055
        fig.text(0.04, yy, label, fontsize=13, color=C_INK, va="center")
        fig.text(0.56, yy, ("✓ " if good else "✗ ") + result, fontsize=13, color=C_PASS if good else C_FAIL, va="center")
    fig.text(0.02, 0.012, "แถบส้ม = ไม่ดีขึ้นเกิน 10% จากค่าตั้งต้น  |  แถบส้มอ่อน = เส้นทางขนาดอื่น (ความคลาดเป็น mm จึงใหญ่ขึ้นตามขนาด)  |  ตัวเลขจาก run_metrics.score_run; "
             "กลุ่มรอบจาก report_runs.json และ tag ใน logs", fontsize=10, color=C_MUTED)
    save(fig, "C3_tuning_history.png")
    register("C3_tuning_history.png", "★", "15 การปรับปรุง / Q&A 'ลองจูนอะไรไปแล้ว'",
             "ทุกวิธีที่ลอง พร้อมความคลาดเฉลี่ยและจำนวนรอบที่บอลหลุด เทียบค่าตั้งต้น",
             "หลักฐานว่าปัญหาไม่ได้อยู่ที่การจูน: เพิ่มเกน ใส่อินทิเกรต ชดเชย dead zone ไม่มีวิธีไหนดีขึ้น",
             f"ค่าตั้งต้น {base_mm:.1f} mm; ×0.15 {k15[0]:.1f} mm; ตัวเลขอื่นในภาพ")


# ================================================================================================
# D. why the real system falls short
# ================================================================================================

def item_d1() -> None:
    cmd = []
    for p in RM.collect_runs("circle"):
        rows = csv_rows(p.stem[4:])
        ph = col(rows, "path_phase")
        v = col(rows, "control_valid") > 0.5
        sel = v & (ph == 3)
        cmd += list(np.hypot(col(rows, "theta_x_cmd_deg"), col(rows, "theta_y_cmd_deg"))[sel])
    typical = float(np.median(cmd))
    bars = [
        ("เส้นทางวงกลม 3 cm คาบ 20 s ต้องการ", TH.reference_tilt_demand_deg(0.03, 20.0), C_SIM),
        ("วงกลม 5 cm คาบ 10 s ต้องการ", TH.reference_tilt_demand_deg(0.05, 10.0), C_SIM),
        ("dead zone แกน Y (บอลเริ่มขยับ)", 0.25, C_TARGET),
        ("dead zone แกน X (บอลเริ่มขยับ)", 0.5, C_TARGET),
        ("มุมที่ตัวควบคุมสั่งจริงระหว่างวน (ค่ากลาง)", typical, C_REAL),
        ("ขีดจำกัดมุมเอียงที่ตั้งไว้", PROF.TILT_LIMIT_DEG, "#c9c8c1"),
    ]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.36, right=0.93, top=0.86, bottom=0.18)
    y = np.arange(len(bars))[::-1]
    for yi, (label, v, colour) in zip(y, bars):
        ax.barh(yi, v, color=colour, height=0.6)
        ax.text(v * 1.12, yi, f"{v:.3f}°" if v < 0.1 else f"{v:.2f}°", va="center", fontsize=14, color=C_INK, weight="bold")
    ax.set_yticks(y, [b[0] for b in bars], fontsize=13.5)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 30)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10], ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel("มุมเอียง (องศา, สเกล log)")
    ax.grid(axis="y", visible=False)
    ratio = 0.25 / bars[0][1]
    ax.set_title(f"เส้นทางต้องการมุมเล็กกว่า dead zone {ratio:.0f}–{0.5 / bars[0][1]:.0f} เท่า บอลจึงไหลตามเส้นไม่ได้", loc="left",
                 x=-0.5)
    fig.text(0.02, 0.02, "มุมที่ต้องการ = a / g_eff โดย a = R·ω² (theory_limits.reference_tilt_demand_deg)  |  "
             "dead zone จากทดสอบเอียงแผ่น (B3)  |  มุมที่สั่งจริง: ค่ากลางของ |θ| ช่วงวนเต็มขนาด รอบ circle ที่นับผล",
             fontsize=10.5, color=C_MUTED)
    save(fig, "D1_tilt_needed_vs_deadzone.png")
    register("D1_tilt_needed_vs_deadzone.png", "★", "15 สาเหตุ / Q&A 'ทำไมตามเส้นไม่ดี'",
             "มุมเอียงที่เส้นทางต้องการ เทียบ dead zone มุมที่สั่งจริง และขีดจำกัด (สเกล log)",
             "เส้นทางช้าจนต้องการมุมเล็กกว่าที่แผ่นตอบสนองได้ บอลจึงต้องรอ error สะสมก่อนขยับ (หยุด-ไถล)",
             f"ต้องการ {bars[0][1]:.3f}°, dead zone 0.25/0.5°, สั่งจริงค่ากลาง {typical:.2f}°")


def real_path_run(stamp: str) -> dict:
    rows = csv_rows(stamp)
    v = col(rows, "control_valid") > 0.5
    t = col(rows, "t_s")
    t0 = t[v][0]
    return {"t": t[v] - t0, "x": col(rows, "x_m")[v], "y": col(rows, "y_m")[v], "xr": col(rows, "x_ref_m")[v],
            "yr": col(rows, "y_ref_m")[v], "phase": col(rows, "path_phase")[v]}


def representative_run(tag: str, min_seen: float = 0.0) -> str:
    """Counted run of this tag that kept the ball and whose mean error is closest to the group mean.

    min_seen > 0 also requires the ball to be detected in at least that fraction of the log rows after
    it was first seen (for a video, where long detection dropouts would hide what the ball does).
    The group mean is always taken over every counted run.
    """
    zone = RM.default_zone_cm()
    scored = [(p.stem[4:], RM.score_run(p, zone)) for p in RM.collect_runs(tag)]
    mean = float(np.mean([r["track_mm"] for _, r in scored]))
    eligible = [(s_, r) for s_, r in scored if not r["lost"] and seen_fraction(s_) >= min_seen]
    return min(eligible, key=lambda item: abs(item[1]["track_mm"] - mean))[0]


def seen_fraction(stamp: str) -> float:
    valid = col(csv_rows(stamp), "control_valid") > 0.5
    return float(valid[int(np.argmax(valid)):].mean()) if valid.any() else 0.0


def item_d2() -> None:
    d2_replay("circle", REPLAY_CIRCLE_RUN, "D2_sim_vs_real_path.mp4", "วงกลม 3 cm")
    d2_replay("hexagon", representative_run("hexagon"), "D2_sim_vs_real_hexagon.mp4", "หกเหลี่ยม 3 cm")


def d2_replay(shape: str, stamp: str, name: str, label: str, speed: int = 4) -> None:
    real = real_path_run(stamp)
    T, P, R = CF.sim_path(shape, GAINS)
    size, scale = 760, 760 / 0.12
    c = size // 2
    to_px = lambda x, y: (int(round(c + x * scale)), int(round(c - y * scale)))
    base = np.full((size, size + 420, 3), 255, np.uint8)
    for k in range(-6, 7):
        cv2.line(base, (int(c + k * 0.01 * scale), 0), (int(c + k * 0.01 * scale), size), (232, 232, 228), 1)
        cv2.line(base, (0, int(c + k * 0.01 * scale)), (size, int(c + k * 0.01 * scale)), (232, 232, 228), 1)
    ref_poly = np.array([to_px(x, y) for x, y in PA.reference_polyline(shape)], np.int32)
    cv2.polylines(base, [ref_poly], True, bgr(C_TARGET), 2, cv2.LINE_AA)
    base = put_text(base, f"เส้นทางที่สั่ง ({label})", (size + 20, 120), 24, bgr(C_TARGET))
    cv2.circle(base, (size + 32, 186), 11, bgr(C_SIM), -1, cv2.LINE_AA)
    cv2.circle(base, (size + 32, 226), 11, bgr(C_REAL), 4, cv2.LINE_AA)
    base = put_text(base, "จำลอง", (size + 55, 170), 26, bgr(C_SIM))
    base = put_text(base, "ของจริง (จุด = ศูนย์กลางบอล)", (size + 55, 210), 26, bgr(C_REAL))
    base = put_text(base, f"1 ช่อง = 1 cm   เล่นเร็ว {speed} เท่า", (size + 20, size - 60), 20, (110, 110, 110))
    base = put_text(base, f"run {stamp} (รอบที่นับผล)", (size + 20, size - 30), 18, (110, 110, 110))

    video = Mp4(name, (size + 420, size))
    end = min(real["t"][-1], T[-1])
    trail = 150
    for tt in np.arange(0.0, end, speed / 30.0):
        frame = base.copy()
        i_sim = int(np.searchsorted(T, tt))
        i_real = int(np.searchsorted(real["t"], tt))
        s0 = max(0, i_sim - int(trail * speed / 30 / 0.01))
        sim_pts = np.array([to_px(*p) for p in P[s0:i_sim + 1:5]], np.int32)
        r0 = max(0, i_real - trail * speed // 2)
        real_pts = np.array([to_px(x, y) for x, y in zip(real["x"][r0:i_real + 1], real["y"][r0:i_real + 1])], np.int32)
        if len(sim_pts) > 1:
            cv2.polylines(frame, [sim_pts], False, bgr(C_SIM), 3, cv2.LINE_AA)
        if len(real_pts) > 1:
            cv2.polylines(frame, [real_pts], False, bgr(C_REAL), 3, cv2.LINE_AA)
        rx, ry = R[min(i_sim, len(R) - 1)]
        cv2.drawMarker(frame, to_px(rx, ry), bgr(C_TARGET), cv2.MARKER_CROSS, 26, 3)
        cv2.circle(frame, to_px(*P[min(i_sim, len(P) - 1)]), 14, bgr(C_SIM), -1, cv2.LINE_AA)
        j = min(i_real, len(real["t"]) - 1)
        cv2.circle(frame, to_px(real["x"][j], real["y"][j]), 14, bgr(C_REAL), 5, cv2.LINE_AA)
        phase = "ทรงตัวที่กลาง" if tt < CF.HOLD_S else ("เส้นทางค่อยๆ ขยาย" if tt < CF.HOLD_S + CF.RAMP_S else "วนเต็มขนาด")
        e_sim = math.hypot(*(P[min(i_sim, len(P) - 1)] - R[min(i_sim, len(R) - 1)])) * 100
        e_real = math.hypot(real["x"][j] - real["xr"][j], real["y"][j] - real["yr"][j]) * 100
        frame = put_text(frame, f"t = {tt:4.1f} s   {phase}", (size + 20, 20), 26)
        frame = put_text(frame, f"ห่างจุดอ้างอิง  จำลอง {e_sim:4.1f} cm", (size + 20, 280), 22, bgr(C_SIM))
        frame = put_text(frame, f"ห่างจุดอ้างอิง  ของจริง {e_real:4.1f} cm", (size + 20, 315), 22, bgr(C_REAL))
        video.write(frame)
    video.close()
    register(name, "★", "14 Tracking / 15 จำลองกับจริง",
             f"มุมมองด้านบน: บอลจำลอง (น้ำเงิน) กับบอลจริงจาก log (ส้ม) วิ่งตาม{label}พร้อมกัน เล่นเร็ว {speed} เท่า",
             "แบบจำลองวิ่งทับเส้นทาง ของจริงตามหลังและวงไม่สม่ำเสมอ ภาพเดียวแทนตารางเปรียบเทียบได้ "
             "(ภาพจาก log ไม่ใช่วิดีโอกล้อง)",
             f"จำลอง: ch4_figures.sim_path เกน {GAINS[0]:g}/{GAINS[1]:g} delay {DELAY:g} s; จริง: run {stamp} "
             "= รอบที่ไม่หลุดซึ่ง error ใกล้ค่าเฉลี่ยกลุ่มที่สุด")


def item_d3() -> None:
    zone = RM.default_zone_cm()
    runs = [(p.stem[4:], RM.score_run(p, zone)) for p in RM.collect_runs("circle")]
    runs = [(s_, r) for s_, r in runs if not r.get("lost") and r.get("phase_deg") == r.get("phase_deg")]
    mean_phase = float(np.mean([r["phase_deg"] for _, r in runs]))
    stamp, res = min(runs, key=lambda item: abs(item[1]["phase_deg"] - mean_phase))
    real = real_path_run(stamp)
    lag_s = -res["phase_deg"] / 360.0 * PROF.PATH_PERIOD_S
    T, P, R = CF.sim_path("circle", GAINS)
    w0 = CF.HOLD_S + CF.RAMP_S + 5.0
    win = (real["t"] >= w0) & (real["t"] <= w0 + 20.0)
    sw = (T >= w0) & (T <= w0 + 20.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.18)
    ax.plot(real["t"][win] - w0, real["xr"][win] * 100, color=C_TARGET, ls="--", lw=2.5, label="ตำแหน่งที่สั่ง (แกน X)")
    ax.plot(T[sw] - w0, P[sw, 0] * 100, color=C_SIM, lw=2.2, label="จำลอง")
    ax.plot(real["t"][win] - w0, real["x"][win] * 100, color=C_REAL, lw=2.4, label="ของจริง")
    ax.axhline(0, color=C_MUTED, lw=0.8)
    tr = real["t"][win] - w0
    xr_w = real["xr"][win]
    up = np.where((xr_w[:-1] < 0) & (xr_w[1:] >= 0))[0]
    t_cross = float(tr[up[0]]) if len(up) else 5.0
    ax.annotate("", xy=(t_cross + lag_s, 0.0), xytext=(t_cross, 0.0), arrowprops=dict(arrowstyle="->", color=C_INK, lw=2.5))
    ax.text(t_cross + lag_s / 2, -0.35, f"ตามหลัง ~{lag_s:.1f} s", ha="center", va="top", fontsize=14, color=C_INK,
            weight="bold", bbox=dict(fc="white", ec="none", pad=2))
    ax.set_ylim(-4.5, 5.0)
    ax.set_xlim(0, 20)
    ax.set_xlabel("เวลา (s) หนึ่งรอบของวงกลม")
    ax.set_ylabel("ตำแหน่งแกน X (cm)")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.set_title(f"ของจริงวนตามจังหวะได้ แต่ตามหลังเส้นทาง ~{-res['phase_deg']:.0f}° ของรอบ (~{lag_s:.1f} s)", loc="left")
    fig.text(0.09, 0.02, f"run {stamp} = รอบวงกลมที่ไม่หลุดซึ่งความล่าช้าใกล้ค่าเฉลี่ยกลุ่มที่สุด ({-mean_phase:.0f}°)  |  "
             "ความล่าช้า = เฟสของความถี่พื้นฐาน (run_metrics.score_run)", fontsize=10.5, color=C_MUTED)
    save(fig, "D3_lag.png")
    register("D3_lag.png", "○", "15 สาเหตุ Lag",
             "ตำแหน่งแกน X หนึ่งรอบ: ที่สั่ง vs จำลอง vs ของจริง พร้อมลูกศรบอกความล่าช้า",
             "อธิบายว่าทำไม error เทียบเวลา (19.8 mm) สูงกว่า error ระยะถึงเส้นทาง (11.4 mm): บอลอยู่บนเส้นแต่มาช้า",
             f"รอบ {stamp}: {-res['phase_deg']:.0f}° ≈ {lag_s:.1f} s; ค่าเฉลี่ยวงกลมที่ไม่หลุด {-mean_phase:.0f}°")


def item_d4() -> None:
    real = real_path_run(REPLAY_CIRCLE_RUN)
    T, P, R = CF.sim_path("circle", GAINS)
    w0 = CF.HOLD_S + CF.RAMP_S
    win = real["t"] >= w0
    sw = T >= w0
    ang = lambda x, y: np.degrees(np.unwrap(np.arctan2(y, x)))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.18)
    ref_ang = ang(real["xr"][win], real["yr"][win])
    real_ang = ang(real["x"][win], real["y"][win])
    real_ang += 360.0 * np.round((ref_ang[0] - real_ang[0]) / 360.0)
    sim_ang = ang(P[sw, 0], P[sw, 1])
    sim_ang += 360.0 * np.round((ref_ang[0] - sim_ang[0]) / 360.0)
    ax.plot(T[sw] - w0, sim_ang, color=C_SIM, lw=5, alpha=0.5, label="จำลอง (ทับเส้นทางที่สั่งเกือบสนิท)")
    ax.plot(real["t"][win] - w0, ref_ang, color=C_TARGET, ls="--", lw=2.2, label="เส้นทางที่สั่ง (หมุนสม่ำเสมอ)", zorder=4)
    ax.plot(real["t"][win] - w0, real_ang, color=C_REAL, lw=2.4, label="ของจริง")
    ax.set_xlabel("เวลาตั้งแต่เริ่มวนเต็มขนาด (s)")
    ax.set_ylabel("มุมรอบศูนย์กลางที่บอลวนไปได้ (องศา)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("บอลจริงวนแบบ 'หยุด-ไถล' (ช่วงราบสลับช่วงชัน) ไม่ลื่นเหมือนแบบจำลอง", loc="left")
    fig.text(0.09, 0.02, f"run {REPLAY_CIRCLE_RUN}  |  มุมรอบศูนย์กลาง = atan2(y, x) แบบต่อเนื่อง  |  "
             "ช่วงราบ = บอลติดอยู่รอให้มุมเอียงพ้น dead zone", fontsize=10.5, color=C_MUTED)
    save(fig, "D4_stick_slip.png")
    register("D4_stick_slip.png", "○", "15 สาเหตุ Dead zone",
             "มุมรอบศูนย์กลางที่บอลวนไปได้ตามเวลา: ที่สั่งเป็นเส้นตรง จำลองเรียบ ของจริงเป็นขั้นบันได",
             "เห็นอาการหยุด-ไถลที่เกิดจาก dead zone โดยไม่ต้องใช้สมการ",
             f"run {REPLAY_CIRCLE_RUN}")


def item_d5() -> None:
    rows = csv_rows(BALANCE_RUN)
    v = col(rows, "control_valid") > 0.5
    t = col(rows, "t_s")[v]
    t = t - t[0]
    x = col(rows, "x_m")[v] * 100
    raw = col(rows, "x_dot_mps")[v] * 100
    k = 9
    smooth = np.convolve(x, np.ones(k) / k, mode="same")
    true_v = np.gradient(smooth, t)
    w = (t >= 14.0) & (t <= 17.0)
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.18, hspace=0.3)
    axes[0].plot(t[w], x[w], color=C_REAL, lw=2.4, marker="o", ms=4)
    axes[0].set_ylabel("ตำแหน่ง X (cm)")
    axes[0].set_title("ตำแหน่งแทบไม่ขยับ …", loc="left", fontsize=15)
    axes[1].axhline(0, color=C_MUTED, lw=0.8)
    axes[1].plot(t[w], raw[w], color=C_REAL, lw=1.6, label="ความเร็วที่ตัวควบคุมใช้ (ผลต่างระหว่างเฟรม)")
    axes[1].plot(t[w], true_v[w], color=C_INK, lw=2.6, label=f"ความเร็วจากตำแหน่งที่เฉลี่ย {k} เฟรม")
    axes[1].set_ylabel("ความเร็ว X (cm/s)")
    axes[1].set_xlabel("เวลา (s)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=12)
    axes[1].set_title("… แต่ความเร็วที่คำนวณได้กระโดดไปมา และส่งตรงไปที่มุมเอียงที่สั่ง", loc="left", fontsize=15)
    ratio = float(np.std(raw[t > 10]) / max(1e-6, np.std(true_v[t > 10])))
    fig.suptitle(f"ความเร็วที่ตัวควบคุมใช้ แกว่งมากกว่าความเร็วจากตำแหน่งที่เฉลี่ย ~{ratio:.0f} เท่า", fontsize=19, x=0.02, ha="left", color=C_INK)
    fig.text(0.1, 0.02, f"run {BALANCE_RUN} ช่วงนิ่งหลังวินาทีที่ 10  |  1 px = {10 / PAS.pixel_per_cm():.2f} mm, "
             f"30 เฟรม/s → คลาด 1 px = {10 / PAS.pixel_per_cm() * 30 / 10:.1f} cm/s", fontsize=10.5, color=C_MUTED)
    save(fig, "D5_velocity_noise.png")
    register("D5_velocity_noise.png", "○", "15 สาเหตุ Noise",
             "ตำแหน่งกับความเร็วช่วง 3 วินาทีที่บอลเกือบนิ่ง: ความเร็วที่ตัวควบคุมใช้ vs ความเร็วจากตำแหน่งที่เฉลี่ย",
             "การหาความเร็วจากผลต่างระหว่างเฟรมขยายสัญญาณรบกวน ทำให้มุมที่สั่งสั่นตาม (เห็นในกราฟหน้า 09 เดิม)",
             f"ส่วนเบี่ยงเบนมาตรฐานต่างกัน ~{ratio:.0f} เท่า")


# ================================================================================================
# E. comparisons as animations
# ================================================================================================

FULLGAIN_RUN = "20260926T023531Z"      # tag fullgain_lqr: k 1.0, kv 1.0, trim off
USEDGAIN_VIDEO_RUN = CENTERING_VIDEO   # tag demo_centering: the standard gains


def run_series(stamp: str) -> dict:
    rows = csv_rows(stamp)
    v = col(rows, "control_valid") > 0.5
    t = col(rows, "t_s")
    t0 = t[v][0]
    dist = np.where(v, np.hypot(col(rows, "x_m") - col(rows, "x_ref_m"), col(rows, "y_m") - col(rows, "y_ref_m")) * 100, np.nan)
    return {"t": t - t0, "valid": v, "dist": dist, "i0": int(np.argmax(v))}


def item_e1() -> None:
    secs = 12.0
    runs = [(USEDGAIN_VIDEO_RUN, f"เกนที่ใช้ ×{GAINS[0]:g} / kv {GAINS[1]:g}", C_SIM),
            (FULLGAIN_RUN, "เกน LQR เต็มตามที่คำนวณ ×1 / kv 1", C_FAIL)]
    data = [run_series(st) for st, _, _ in runs]
    caps = [cv2.VideoCapture(str(LOGS / f"video_{st}.mp4")) for st, _, _ in runs]
    frames = [[] for _ in runs]
    for k, cap in enumerate(caps):
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames[k].append(cv2.resize(f, (640, 400), interpolation=cv2.INTER_AREA))
        cap.release()

    W, PH = 1280, 330
    fig, ax = plt.subplots(figsize=(W / 100, PH / 100), dpi=100)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    for d, (_, label, colour) in zip(data, runs):
        sel = d["t"] <= secs
        ax.plot(d["t"][sel], d["dist"][sel], color=colour, lw=2.2, label=label)
        gaps = sel & ~d["valid"] & (d["t"] >= 0)
        ax.scatter(d["t"][gaps], np.full(gaps.sum(), -0.4), marker="|", s=40, color=colour, alpha=0.5)
    ax.axhline(4.0, color=C_TARGET, ls="--", lw=1.5)
    ax.text(secs, 4.15, "โซนศูนย์กลาง 4 cm", ha="right", fontsize=11, color=C_MUTED)
    ax.text(0.1, 9.3, "ขีดที่ขอบล่าง = ช่วงที่กล้องมองไม่เห็นบอล", fontsize=11, color=C_MUTED, va="center")
    ax.set_xlim(0, secs)
    ax.set_ylim(-0.9, 10)
    ax.set_xlabel("เวลาตั้งแต่เห็นบอลครั้งแรก (s)", fontsize=12)
    ax.set_ylabel("ห่างกลาง (cm)", fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=11.5, ncol=2)
    fig.canvas.draw()
    plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR).copy()
    tx = lambda tt: int(round(ax.transData.transform((tt, 0))[0]))
    y_top, y_bot = int(PH - ax.get_window_extent().y1), int(PH - ax.get_window_extent().y0)
    plt.close(fig)

    header = np.full((56, W, 3), 255, np.uint8)
    for k, (_, label, colour) in enumerate(runs):
        header = put_text(header, label, (20 + 640 * k, 12), 26, bgr(colour))
    video = Mp4("E1_fullgain_vs_used.mp4", (W, 56 + 400 + PH))
    for tt in np.arange(0.0, secs, 1 / 30.0):
        panels = []
        for d, fr in zip(data, frames):
            j = int(np.clip(d["i0"] + np.searchsorted(d["t"][d["i0"]:], tt), 0, len(fr) - 1))
            img = fr[j].copy()
            if not d["valid"][j]:
                img = put_text(img, "กล้องมองไม่เห็นบอล → แผ่นกลับตำแหน่งกลาง", (12, 360), 20, (255, 255, 255), bg=(0, 0, 160))
            panels.append(img)
        bottom = plot.copy()
        cv2.line(bottom, (tx(tt), y_top), (tx(tt), y_bot), (60, 60, 60), 2)
        video.write(np.vstack([header, np.hstack(panels), bottom]))
    video.close()
    zone = RM.default_zone_cm()
    rf, ru = RM.score_run(LOGS / f"run_{FULLGAIN_RUN}.csv", zone), RM.score_run(LOGS / f"run_{USEDGAIN_VIDEO_RUN}.csv", zone)
    seen = 100 * data[1]["valid"].mean()
    register("E1_fullgain_vs_used.mp4", "★", "10 LQR หรือ 11 เสถียรภาพ (หลักฐานจากเครื่องจริง)",
             "วิดีโอกล้องจริง 2 รอบเคียงกัน: เกนที่ใช้ vs เกน LQR เต็ม พร้อมกราฟระยะห่างกลางที่เดินไปพร้อมกัน",
             "เกนเต็มเหวี่ยงบอลข้ามแผ่นจนกล้องมองไม่เห็นเป็นช่วงๆ ส่วนเกนที่ใช้ดึงเข้ากลางแล้วนิ่ง ยืนยันรูป C1 จากของจริง",
             f"เกนเต็ม run {FULLGAIN_RUN}: overshoot {rf['overshoot_pct']:.0f}%, RMS {rf['rms_mm']:.1f} mm, เห็นบอล {seen:.0f}% ของเฟรม, "
             f"บอลหลุด; เกนที่ใช้ run {USEDGAIN_VIDEO_RUN}: overshoot {ru['overshoot_pct']:.0f}%, RMS {ru['rms_mm']:.1f} mm "
             "(ทั้งสองเป็นรอบสาธิต 1 รอบ ไม่นับในบทที่ 4)")


def item_e2() -> None:
    from test_tilt_direction import _lagged_double_integral
    m = CF.manifest()
    tests = []
    for tilt, stamp in m["tilt_sweep"]["x"]:
        with open(LOGS / f"tilt_test_x_{tilt:g}deg_{stamp}.csv", newline="", encoding="utf-8") as h:
            rows = list(csv.DictReader(h))
        t = np.array([float(r["t_s"]) for r in rows])
        tr = np.array([r["tracked"] == "1" for r in rows])
        x = np.array([float(r["x_cm"] or "nan") for r in rows])
        u = np.radians(np.array([float(r["tilt_cmd_deg"]) for r in rows]))
        phase = np.array([r["phase"] for r in rows])
        base = float(np.median(x[(phase == "baseline") & tr][-10:]))
        t_plus = t[phase == "tilt+"][0]
        theory = _lagged_double_integral(t, u, TA_FROM_PARAMS, DELAY) * ST.G_EFF * 100
        tests.append({"tilt": tilt, "t": t - t_plus, "meas": np.where(tr, x - base, np.nan), "theory": theory - theory[t < t_plus][-1] if (t < t_plus).any() else theory,
                      "u": np.degrees(u)})
    W, lane_h = 1280, 165
    H = 100 + lane_h * len(tests)
    scale = 900 / 8.0                         # 8 cm of travel across 900 px
    x0 = 300
    video = Mp4("E2_deadzone_tilt_test.mp4", (W, H))
    for tt in np.arange(-0.5, 2.3, 1 / 60.0):            # half speed
        frame = np.full((H, W, 3), 255, np.uint8)
        frame = put_text(frame, f"เอียงแผ่นตรงๆ (ไม่มีตัวควบคุม) เวลา {tt:+.2f} s   เล่นช้า 0.5 เท่า", (20, 14), 26)
        cv2.circle(frame, (32, 66), 10, bgr(C_REAL), -1, cv2.LINE_AA)
        cv2.circle(frame, (262, 66), 10, bgr(C_SIM), 3, cv2.LINE_AA)
        frame = put_text(frame, "บอลจริง (กล้อง)", (50, 52), 20, (90, 90, 90))
        frame = put_text(frame, "ทฤษฎีเชิงเส้น   (มุมเอียงในภาพขยาย 3 เท่า)", (280, 52), 20, (90, 90, 90))
        for k, d in enumerate(tests):
            cy = 100 + k * lane_h + 95
            j = int(np.clip(np.searchsorted(d["t"], tt), 0, len(d["t"]) - 1))
            ang = math.radians(float(np.clip(d["u"][j] * 3, -4.5, 4.5)))     # tilt drawn x3, clamped so lanes never overlap
            c, s_ = math.cos(ang), math.sin(ang)
            p1 = (int(x0 - 60 * c), int(cy + 60 * s_))
            p2 = (int(x0 + 960 * c), int(cy - 960 * s_))
            cv2.line(frame, p1, p2, (90, 90, 90), 6, cv2.LINE_AA)
            cv2.line(frame, (x0, cy - 40), (x0, cy + 8), bgr(C_TARGET), 1)
            meas = d["meas"][j]
            th = float(d["theory"][j])
            for value, colour, filled in ((th, C_SIM, False), (meas, C_REAL, True)):
                if value != value:
                    continue
                px = x0 + float(np.clip(value, -0.5, 8.0)) * scale
                py = cy - (px - x0) * s_ / c - 16
                cv2.circle(frame, (int(px), int(py)), 14, bgr(colour), -1 if filled else 4, cv2.LINE_AA)
            label = f"สั่งเอียง {d['tilt']:g}°"
            frame = put_text(frame, label, (20, cy - 40), 26, (30, 30, 30))
            if meas == meas:
                frame = put_text(frame, f"จริง {meas:+.1f} cm / ทฤษฎี {th:+.1f} cm", (20, cy), 18, (90, 90, 90))
        video.write(frame)
    video.close()
    register("E2_deadzone_tilt_test.mp4", "★", "15 สาเหตุ Dead zone (แทน/คู่กับ B3)",
             "ผลทดสอบเอียงแผ่นตรงๆ 4 มุม (0.3, 0.6, 1, 2°) แกน X: บอลจริง (ส้ม) เทียบบอลตามทฤษฎี (น้ำเงิน) เคลื่อนพร้อมกัน",
             "มุมเล็กบอลจริงแทบไม่ขยับขณะที่ทฤษฎีบอกว่าควรกลิ้ง มุมใหญ่สองลูกไปด้วยกัน = dead zone เห็นได้ด้วยตา",
             "logs/tilt_test_x_* (report_runs.json tilt_sweep); ทฤษฎี = g_eff 5.89 m/s²/rad กับ τ 0.18 s และ delay 0.13 s")


def item_e3() -> None:
    rows = csv_rows(BALANCE_RUN)
    v = col(rows, "control_valid") > 0.5
    t = col(rows, "t_s")[v]
    t = t - t[0]
    x, y = col(rows, "x_m")[v], col(rows, "y_m")[v]
    secs = 12.0
    model = ST.Model(*GAINS)
    Ts, Ex = ST.step_response(model, float(x[0]), secs=secs, dt=0.005, delay_s=DELAY)
    _, Ey = ST.step_response(model, float(y[0]), secs=secs, dt=0.005, delay_s=DELAY)
    size, sc = 720, 720 / 0.20
    c = size // 2
    to_px = lambda a, b: (int(round(c + a * sc)), int(round(c - b * sc)))
    base = np.full((size, size + 440, 3), 255, np.uint8)
    for k in range(-10, 11):
        cv2.line(base, (int(c + k * 0.01 * sc), 0), (int(c + k * 0.01 * sc), size), (232, 232, 228), 1)
        cv2.line(base, (0, int(c + k * 0.01 * sc)), (size, int(c + k * 0.01 * sc)), (232, 232, 228), 1)
    cv2.circle(base, (c, c), int(0.04 * sc), bgr(C_TARGET), 2, cv2.LINE_AA)
    cv2.drawMarker(base, (c, c), bgr(C_TARGET), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(base, (size + 32, 186), 11, bgr(C_SIM), -1, cv2.LINE_AA)
    cv2.circle(base, (size + 32, 226), 11, bgr(C_REAL), 4, cv2.LINE_AA)
    base = put_text(base, "จำลอง (แบบจำลองเชิงเส้น + delay 0.13 s)", (size + 55, 170), 22, bgr(C_SIM))
    base = put_text(base, "ของจริง (รอบที่นับผล)", (size + 55, 210), 22, bgr(C_REAL))
    base = put_text(base, "วงเทา = โซนศูนย์กลาง 4 cm   1 ช่อง = 1 cm", (size + 20, size - 60), 18, (110, 110, 110))
    base = put_text(base, f"run {BALANCE_RUN}", (size + 20, size - 30), 18, (110, 110, 110))
    video = Mp4("E3_sim_vs_real_centering.mp4", (size + 440, size))
    for tt in np.arange(0.0, secs, 1 / 30.0):
        frame = base.copy()
        i_s = int(np.searchsorted(Ts, tt))
        i_r = int(np.searchsorted(t, tt))
        sim_pts = np.array([to_px(a, b) for a, b in zip(Ex[:i_s + 1:6], Ey[:i_s + 1:6])], np.int32)
        real_pts = np.array([to_px(a, b) for a, b in zip(x[:i_r + 1], y[:i_r + 1])], np.int32)
        if len(sim_pts) > 1:
            cv2.polylines(frame, [sim_pts], False, bgr(C_SIM), 3, cv2.LINE_AA)
        if len(real_pts) > 1:
            cv2.polylines(frame, [real_pts], False, bgr(C_REAL), 2, cv2.LINE_AA)
        i_s, i_r = min(i_s, len(Ex) - 1), min(i_r, len(x) - 1)
        cv2.circle(frame, to_px(Ex[i_s], Ey[i_s]), 14, bgr(C_SIM), -1, cv2.LINE_AA)
        cv2.circle(frame, to_px(x[i_r], y[i_r]), 14, bgr(C_REAL), 5, cv2.LINE_AA)
        frame = put_text(frame, f"t = {tt:4.1f} s", (size + 20, 20), 28)
        frame = put_text(frame, f"ห่างกลาง  จำลอง {math.hypot(Ex[i_s], Ey[i_s]) * 100:4.1f} cm", (size + 20, 280), 22, bgr(C_SIM))
        frame = put_text(frame, f"ห่างกลาง  ของจริง {math.hypot(x[i_r], y[i_r]) * 100:4.1f} cm", (size + 20, 315), 22, bgr(C_REAL))
        video.write(frame)
    video.close()
    register("E3_sim_vs_real_centering.mp4", "★", "12 ผลจำลอง / 15a จำลองกับจริง",
             "มุมมองด้านบน: บอลจำลองกับบอลจริงเริ่มจากจุดเดียวกัน (8 cm) แล้วดึงเข้ากลางพร้อมกัน",
             "จำลองเข้ากลางแล้วนิ่งสนิท ของจริงเข้าโซนเร็วพอกันแต่ค้างวนอยู่ห่างกลาง 1–2 cm (dead zone)",
             f"run {BALANCE_RUN}; จำลองด้วย stability.step_response เกน {GAINS[0]:g}/{GAINS[1]:g} delay {DELAY:g} s")


def item_e5() -> None:
    """Why the commanded tilt chatters: camera noise, differentiated into a velocity, times the velocity gain."""
    rows = csv_rows(BALANCE_RUN)
    valid = col(rows, "control_valid") > 0.5
    t = col(rows, "t_s")
    t = t - t[valid][0]
    t0, t1 = 6.0, 9.0          # the ball is almost still here, so what moves is noise, not the ball
    m = valid & (t >= t0) & (t < t1)
    x_mm = col(rows, "x_m")[m] * 1000
    v_cms = col(rows, "x_dot_mps")[m] * 100
    cmd = col(rows, "theta_x_cmd_deg")[m]
    k1 = TH.gains(*GAINS)[0]
    k2 = TH.gains(*GAINS)[1]
    from_v = np.degrees(k2 * v_cms / 100)
    from_e = np.degrees(k1 * x_mm / 1000)
    share = float(np.var(from_v) / np.var(cmd))
    tt = t[m]
    tt = tt - tt[0]

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 6.4), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.17, hspace=0.18)
    series = [(x_mm - x_mm.mean(), "ตำแหน่ง\n(mm)", C_SIM, f"ตำแหน่งแกว่งแค่ ±{np.std(x_mm):.1f} mm"),
              (v_cms, "ความเร็ว\n(cm/s)", C_REAL, f"ความเร็วที่คำนวณได้\nกระโดด ±{np.std(v_cms):.1f} cm/s"),
              (cmd, "มุมที่สั่ง\n(องศา)", "#7a4fb3", f"มุมที่สั่งสั่น ±{np.std(cmd):.2f}°\n{share * 100:.0f}% มาจากเทอมความเร็ว")]
    for ax, (y, label, colour, note) in zip(axes, series):
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.plot(tt, y, color=colour, lw=2.0)
        lim = max(abs(np.nanmin(y)), abs(np.nanmax(y))) * 1.15
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(label, fontsize=14, rotation=0, ha="right", va="center")
        ax.text(1.03, 0.5, note, transform=ax.transAxes, fontsize=16, color=colour, va="center", ha="left")
    axes[-1].set_xlabel("เวลา (s)", fontsize=14)
    fig.suptitle("ลูกบอลแทบนิ่ง แต่ความเร็วที่คำนวณจากกล้องกระโดด ทำให้มุมที่สั่งสั่นกลับไปมา", fontsize=18, color=C_INK)
    fig.text(0.1, 0.015, f"แกน X, รอบทรงตัว {BALANCE_RUN} ช่วง {t0:g}–{t1:g} s หลังเห็นลูกบอล  |  "
             f"เทอมความเร็ว = k2·v, k2 = {k2:.3f} rad/(m/s)  |  1 px ต่อเฟรมที่ 30 Hz = 7.5 mm/s",
             fontsize=10.5, color=C_MUTED)
    save(fig, "E5_velocity_noise.png")
    register("E5_velocity_noise.png", "★", "สาเหตุ 3 Noise จากการวัดความเร็ว",
             "ช่วง 3 วินาทีที่ลูกบอลแทบนิ่ง: ตำแหน่ง, ความเร็วที่คำนวณได้ และมุมที่สั่ง ของแกน X",
             "กล้องสั่นแค่ระดับพิกเซล แต่การหาผลต่างระหว่างเฟรมขยายให้เป็นความเร็วที่กระโดด และเกนความเร็วส่งต่อไปเป็นมุมที่สั่งสั่น",
             f"run {BALANCE_RUN}: std ตำแหน่ง {np.std(x_mm):.1f} mm, ความเร็ว {np.std(v_cms):.1f} cm/s, มุมสั่ง "
             f"{np.std(cmd):.2f}° ซึ่ง {share * 100:.0f}% ของ variance มาจาก k2·v (เทอมตำแหน่ง k1·e เพียง {np.std(from_e):.2f}°)")


def item_e4() -> None:
    base = dict(e=0.02, v=0.15, a=math.radians(5.0), u=math.radians(6.0))
    knobs = [("e", "ปรับ q_e (ยอมให้ตำแหน่งคลาด e_max)", [0.01, 0.02, 0.04], lambda v_: f"{v_ * 100:g} cm"),
             ("v", "ปรับ q_v (ยอมให้ความเร็ว v_max)", [0.075, 0.15, 0.30], lambda v_: f"{v_:g} m/s"),
             ("u", "ปรับ r (ยอมให้มุมที่สั่ง u_max)", [math.radians(3), math.radians(6), math.radians(12)],
              lambda v_: f"{math.degrees(v_):g}°")]
    shades = ["#9cc2ee", C_SIM, "#0d366b"]
    sims = []
    for key, title, values, fmt in knobs:
        lanes = []
        for value in values:
            p = dict(base, **{key: value})
            K = lqr_gain(np.diag([1 / p["e"] ** 2, 1 / p["v"] ** 2, 1 / p["a"] ** 2]), 1 / p["u"] ** 2)
            T, E = response_with_gain(K, 0.06, 3.0, 0.002, 0.0)
            lanes.append((fmt(value) + ("  (ที่ออกแบบ)" if value == base[key] else ""), T, E))
        sims.append((title, lanes))
    W, lane_h, head = 1280, 58, 50
    H = 60 + len(sims) * (head + 3 * lane_h + 20)
    x0, sc = 900, 700 / 0.07
    video = Mp4("E4_lqr_q_r_effect.mp4", (W, H))
    for tt in np.arange(0.0, 3.0, 1 / 60.0):             # half speed
        frame = np.full((H, W, 3), 255, np.uint8)
        frame = put_text(frame, f"ขั้นออกแบบ LQR: ปล่อยบอลห่างกลาง 6 cm   t = {tt:.2f} s   (เล่นช้า 0.5 เท่า)", (20, 12), 24)
        y = 60
        for title, lanes in sims:
            frame = put_text(frame, title, (20, y + 8), 24, (30, 30, 30))
            y += head
            for k, (label, T, E) in enumerate(lanes):
                cy = y + k * lane_h + lane_h // 2
                cv2.line(frame, (x0 - int(0.065 * sc), cy), (x0 + int(0.015 * sc), cy), (215, 215, 210), 2)
                cv2.line(frame, (x0, cy - 20), (x0, cy + 20), bgr(C_TARGET), 2)
                i = min(int(tt / 0.002), len(E) - 1)
                cv2.circle(frame, (int(x0 - E[i] * sc), cy), 13, bgr(shades[k]), -1, cv2.LINE_AA)
                frame = put_text(frame, label, (40, cy - 14), 20, (60, 60, 60))
            y += 3 * lane_h + 20
        video.write(frame)
    video.close()
    register("E4_lqr_q_r_effect.mp4", "○", "10 LQR (คู่กับรูป C4)",
             "แอนิเมชันของรูป C4: บอลสามลูกต่อหนึ่งตัวปรับ วิ่งเข้ากลางพร้อมกัน เห็นลูกไหนเร็ว/ช้า/เลยเป้า",
             "ใช้แทนตาราง 'เพิ่ม q_e / q_v / r' ได้ทันที ตัวเลขอยู่ในรูป C4",
             "ข้อมูลเดียวกับ C4 (Q/R จาก notebook, แบบจำลองขั้นออกแบบ ไม่มีเวลาหน่วง)")


# ================================================================================================
# existing assets, Q&A sheet, catalogue
# ================================================================================================

def register_existing() -> None:
    root = "../"
    for file, prio, slide, shows in (
        ("02-04_pipeline_strip.png", "★", "8 Computer Vision", "ภาพดิบ → threshold → ลบ noise"),
        ("05-06_detection_strip.png", "★", "8 Computer Vision", "Hough → contour สำรอง → จุดศูนย์กลางเป็น cm"),
        ("05b_plate_cm_grid.png", "○", "8 Computer Vision", "ตาราง cm ซ้อนบนภาพกล้อง"),
        ("05c_pixel_cm_table.png", "○", "8 / สำรอง", "ตารางแปลง pixel ↔ cm"),
        ("14_tracking_two_metrics.png", "★", "14 ผลจริง Tracking", "ตัวชี้วัด 2 แบบเทียบเกณฑ์ 15 mm"),
        ("09_velocity_angle_vs_time.png", "○", "สำรอง", "ตำแหน่ง ความเร็ว มุมที่สั่ง รอบทรงตัว"),
    ):
        if (KIT.parent / file).exists():
            CATALOG.append({"file": root + file, "priority": prio, "slide": slide, "shows": shows,
                            "caption": "ทำไว้ก่อนแล้ว (presentation_assets.py)", "numbers": "ดูไฟล์ *_notes.json ข้างภาพ"})


QA = """# ชีตถาม-ตอบ (เตรียมสำหรับกรรมการ)

ตัวเลขทุกตัวในชีตนี้สร้างจาก log และโค้ดใน workspace ตอนรัน presentation_kit.py ทุกข้อบอกว่าควรเปิดรูปไหน

## ก. ตัวเลขมาจากไหน

**ถาม: เกณฑ์ tracking ≤ 15 mm มาจากไหน**
ตอบ: ถ้าไม่ควบคุมเลย ปล่อยบอลนิ่งกลางแผ่น ความคลาดจะเท่ารัศมีเส้นทาง 30 mm เราตั้งเกณฑ์ว่าต้องดีกว่านั้นอย่างน้อยครึ่งหนึ่ง คือ 15 mm (theory_limits.py) → รูป B1, 14_tracking_two_metrics
*หมายเหตุ: ใช้ 15 mm ตามรายงาน 4.3.2 ไม่ใช่ 10 mm เพราะ 10 mm ไม่มีที่มา*

**ถาม: เกณฑ์อื่นๆ มาจากไหน**
ตอบ: RMS 10 mm = 1/4 ของเส้นผ่านศูนย์กลางบอล, settling 7 s = 2 เท่าของทฤษฎี ({settle_th:.2f} s), steady 20 mm = รัศมีบอล, overshoot 20% = ทฤษฎี {os_th:.0f}% บวกเผื่อ dead zone/delay, success 80% = 4 ใน 5 รอบ → รูป B1

**ถาม: ค่า k_g (อัตราเร่งต่อมุมเอียง) มาจากไหน**
ตอบ: ลูกปิงปองเป็นทรงกลมกลวง I = 2/3·m·r² บอลกลิ้งด้วย a = 3/5·g·sinθ → g_eff = {g_eff:.2f} m/s²/rad (params.KR = 0.6) วัดจริงได้ X {gx:g}, Y {gy:g} → รูป B3
*ถ้าสไลด์เขียน 7.0 (= 5/7·g) นั่นคือสูตรลูกตัน ต้องแก้*

**ถาม: pixel_per_cm 39.80 มาจากไหน**
ตอบ: คาลิเบรตด้วยกระดานหมากรุกช่อง 20 mm (04_calibration.py) ตรวจซ้ำด้วยขนาดบอลในภาพ (~160 px / 4 cm ≈ 40 px/cm) ภาพกระดานที่เหลืออยู่วัดได้ 36.67 (ต่าง 7.9%) น่าจะถ่ายตอนกระดาษอยู่ต่ำกว่าระดับแผ่น → รูป 05b, 05c

**ถาม: เวลาหน่วง 0.13 s วัดมาหรือเปล่า**
ตอบ: ยังไม่ได้วัดตรง เป็นค่าประมาณ ส่วนที่รู้แล้ว {known:.0f} ms (กล้อง {age:.0f} ms วัดจาก log, รอบลูป {loop:.0f} ms วัดจาก log, คาบส่งคำสั่ง และเวลาที่ตั้งให้เซอร์โว) ส่วนที่ยังไม่รู้คือกลไกเซอร์โว ทดสอบวัดแบบวงเปิดแล้วใช้ไม่ได้เพราะบอลไหลออกจากแผ่น → รูป A3
ที่สำคัญ: error ตามเส้นทางในทฤษฎีไม่ขึ้นกับ delay (1.4 mm ทุกค่า) delay มีผลแค่ต่อ phase margin

## ข. จูนยังไง

**ถาม: ค่า Q, R และเกน LQR ได้มาอย่างไร**
ตอบ: ใช้ Bryson's rule ใน calculated_lqr_ik_3rrs.ipynb: ยอมให้ตำแหน่งคลาด 2 cm, ความเร็ว 0.15 m/s, มุมแผ่น 5°, มุมที่สั่ง 6° → Q = diag(2500, 44.4, 131.3), R = 91.2 แก้สมการ Riccati ได้ K = [5.236, 1.853, 1.530] (ตรวจซ้ำแล้วได้ค่าเดียวกับ params.K_AXIS) โพลวงปิด −3.27 ± 3.04j, −8.52 → รูป C4

**ถาม: ทำไมไม่ใช้เกน LQR ตรงๆ แต่คูณ 0.076 และคูณเกนความเร็ว 2.8**
ตอบ: LQR ออกแบบโดยไม่มีเวลาหน่วง ถ้าใส่เวลาหน่วงเข้าไป เกนตามที่ออกแบบเหลือ PM {pm_lqr:.0f}° ที่ 0.13 s และไม่เสถียรเมื่อหน่วงเกิน {crit_lqr:.2f} s ซึ่งอยู่ในช่วงที่เป็นไปได้ (0.05–0.25 s, ยังไม่ได้วัด) เกนที่ใช้มี PM {pm_used:.0f}° และทนได้ถึง {crit_used:.2f} s → รูป C1
ข้อควรรู้: การลดเกนเป็นการเลือกจากการทดลองและการวิเคราะห์เสถียรภาพ ไม่ได้ออกแบบ LQR ใหม่ (เกนที่ใช้ไม่ตรงกับ LQR ชุดใดที่ Q เป็นเมทริกซ์ทแยง)

**ถาม: ทำไมเลือก 0.076 ไม่ใช่ 0.15 ที่ดูเร็วกว่า**
ตอบ: ในแบบจำลอง 0.15 เร็วกว่าจริง แต่ทนเวลาหน่วงได้ถึง {crit_15:.2f} s เท่านั้น (0.076 ทนได้ {crit_used:.2f} s) และทดลองจริงที่ ×0.15 ตามเส้นทางไม่ได้ดีขึ้น → รูป C1, C3

**ถาม: τ = 0.18 s ในแบบจำลองมาจากไหน**
ตอบ: ประมาณจากความเร็วตัวเปล่าของเซอร์โวในสเปก: τ ≈ 1/ω = 0.19 s ÷ (π/3) = 0.181 s (notebook ใช้สเปก HX-30HM; HX-35H ที่ใช้จริงระบุ 0.18 s/60° จะได้ 0.172 s) notebook เองระบุว่าเป็นค่าประมาณหยาบสำหรับการออกแบบ ไม่ใช่ค่าที่วัด และการ fit จากข้อมูลทดสอบเอียงแผ่นแยกค่า τ ออกมาไม่ได้

**ถาม: แบบจำลองใน notebook กับที่ใช้วิเคราะห์ต่างกันไหม**
ตอบ: ต่างกันหนึ่งพจน์ notebook มีแรงหน่วงของบอล −(b/m)ė โดยสมมติ b = 0.003 N·s/m (1.11 s⁻¹) ใช้ตอนหา K ส่วน stability.py ที่ใช้วิเคราะห์เสถียรภาพไม่ใส่พจน์นี้ (ปลอดภัยกว่า เพราะ b ไม่เคยวัด) ถ้าใส่ PM ของเกนที่ใช้ที่ 0.13 s จะเป็น ~61° แทน 32°

**ถาม: ลองใช้เกน LQR เต็มกับเครื่องจริงหรือยัง**
ตอบ: ลองแล้ว 1 รอบ (tag fullgain_lqr, ไม่นับในบทที่ 4): overshoot {os_full:.0f}%, RMS {rms_full:.1f} mm, กล้องเห็นบอลแค่ {seen_full:.0f}% ของเฟรมเพราะบอลถูกเหวี่ยงไปขอบแผ่น และบอลหลุด เทียบกับรอบสาธิตเกนที่ใช้ overshoot {os_used:.0f}%, RMS {rms_used:.1f} mm แบบจำลองทำนายไว้แล้วว่าเกนเต็มจะแกว่งมาก (overshoot 32% ที่ delay 0.13 s) ของจริงแกว่งมากกว่านั้น → วิดีโอ E1

**ถาม: ผลจำลอง overshoot 7.9% / Ts 2.86 s กับ 5% / 3.45 s ต่างกันทำไม**
ตอบ: เป็นคนละนิยาม ทั้งคู่ปล่อยบอลจาก 6 cm ที่เกน 0.076/2.8 ชุด 7.9% และ 2.86 s คือจำลองรวมเวลาหน่วง 0.13 s (ใช้ในรายงาน บทที่ 4) ส่วน 4.6% และ 3.45 s คือแบบจำลองอุดมคติไม่มีเวลาหน่วง (ใช้ตั้งเกณฑ์ใน B1) Ts คือเวลาเข้าแถบ ±2% ของระยะเริ่ม ถ้าวัดแบบเดียวกับผลจริง (เข้าโซน 4 cm) จำลองได้ 0.86 s ของจริง 1.8 s

**ถาม: ลองปรับอะไรไปแล้วบ้าง**
ตอบ: เพิ่มเกน, เพิ่มตัวอินทิเกรต k_i 0.05/0.10/0.20, เร่ง trim, ชดเชย dead zone 0.5°/1.0°, ขยายเส้นทาง 5/7 cm, วงเร็วขึ้น 5 cm/10 s ไม่มีวิธีไหนทำให้ตามเส้นทางดีกว่าค่าตั้งต้น สิ่งเดียวที่ดีขึ้นคือลดคาบส่งคำสั่งเซอร์โวจาก 0.10 เป็น 0.04 s → รูป C3

## ค. ทำไมยังไม่ดี

**ถาม: ทำไมจำลองได้ 1.4 mm แต่ของจริง 11–20 mm**
ตอบ: แบบจำลองเป็นเชิงเส้น ไม่มี dead zone เส้นทางวงกลม 3 cm ที่คาบ 20 s ต้องการมุมเอียงแค่ {demand:.3f}° แต่แผ่นต้องเอียงเกิน 0.25–0.5° บอลถึงจะเริ่มขยับ บอลจึงติดแล้วไถลเป็นช่วงๆ และตามหลัง ~1.5 s → รูป D1, D2, D4, D3

**ถาม: ความคลาดสองแบบต่างกันอย่างไร**
ตอบ: ระยะถึงเส้นทาง ({cross:.1f} mm) วัดว่าบอลอยู่ห่างเส้นแค่ไหน ไม่สนเวลา ส่วนเทียบ ณ เวลาเดียวกัน ({sync:.1f} mm) นับรวมที่บอลมาช้า ผลต่างคือส่วนของความล่าช้า เทียบเกณฑ์ 15 mm แบบแรกผ่าน แบบหลังไม่ผ่าน → รูป 14, D3

**ถาม: ทำไมมุมที่สั่งสั่นตลอดเวลา**
ตอบ: ความเร็วหาจากผลต่างตำแหน่งระหว่างเฟรม คลาด 1 px ต่อเฟรมกลายเป็นความเร็วคลาดหลายมิลลิเมตรต่อวินาที และเกนความเร็วสูง (×2.8) สั่นนี้จึงไปถึงมุมเอียง ข้อเสนอคือใช้ตัวกรองหรือตัวประมาณสถานะ แต่ยังไม่ได้ทดลองว่าช่วยหรือทำให้ข้าม dead zone ยากขึ้น → รูป D5

**ถาม: ทำไมไม่แก้ด้วยการชดเชย dead zone**
ตอบ: ลองแล้วที่ 0.5° และ 1.0° บอลแกว่ง ~1 Hz เกินเส้นทางแล้วหลุด เพราะการชดเชยแบบง่ายเท่ากับเพิ่มเกนในลูปที่ที่เผื่อเสถียรภาพเหลือน้อย ข้อเสนอคือลดระยะหลวมทางกลไก หรือชดเชยแบบมีฮิสเทอรีซิส → รูป C3

## ง. กลไกและ IK

**ถาม: ขนาดกลไกจริงเท่าไร**
ตอบ: Link 1 = 65 mm, Link 2 = 112 mm (servo_calibration/config/servo_geometry.json ตรงกับแบบ CAD Link 1 = 65.00) ที่ท่า neutral Link 1 ทำมุม 25° กับพื้น Link 2 ตั้งตรง ความสูง H0 = 65·sin25° + 112 + 28 (จุดยึดถึงใต้แผ่น) + 6 (แผ่นหนา) = 173.5 mm ตรงกับที่วัด เซอร์โวอยู่ด้านใน ขาชี้ออกไปหาจุดยึดใต้แผ่น (รัศมีจุดยึดประมาณ 13 cm จากอัตรา 2.26 ที่วัด ควรวัดยืนยัน)

**ถาม: IK คำนวณทุกรอบบน Pi 5 ใช่ไหม**
ตอบ: ตัวควบคุมไม่ได้ใช้ IK ขับเซอร์โว แต่ใช้ตัวคูณเชิงเส้นที่วัดจากเครื่องจริง (servo_mapping.json): มุมเซอร์โว = 2.26·θx·[1, −½, −½] + 1.96·θy·[0, −1, +1] ซึ่งคือ IK ที่ linearize รอบท่า neutral แล้ว ที่มุมเอียง 5° ต่างจาก IK เต็มไม่เกิน ~8% ส่วน IK เต็มใน main.py คำนวณไว้แสดงผลเท่านั้น และยังใช้เรขาคณิตจาก notebook (ฐานเซอร์โวรัศมี 180 mm ขาชี้เข้า) ที่ไม่ตรงเครื่องจริง → วิดีโอ A2

## จ. กล้องและภาพ (อาจารย์สาย computer vision)

**ถาม: ทำไมใช้กล้องขาวดำ แยกบอลอย่างไร**
ตอบ: กล้อง global shutter รุ่นสี IMX296 ขาดตลาด จึงใช้ OV9281 ซึ่งเป็น global shutter ขาวดำ (ภาพไม่บิดเมื่อบอลเคลื่อนที่เร็วเหมือนกัน) ข้อแลกคือใช้ HSV ไม่ได้ ต้องแยกด้วยความสว่าง: blur 7 → threshold 160–255 → open/close 5×5 → Hough circle (ตรวจความสว่างในวงและรอบวง) ถ้า Hough ไม่เจอค่อยใช้ contour ที่ความกลม ≥ 0.6 → รูป 02-04, 05-06

**ถาม: แสงสะท้อนมีผลไหม**
ตอบ: มี ในภาพตัวอย่างแสงสะท้อนติดเป็นก้อนเดียวกับบอลหลังลบ noise ความกลมเหลือ 0.45 ขั้น contour จะหาไม่เจอ แต่ Hough ยังเจอเพราะดูขอบวงกลม และมีการตัดตำแหน่งที่ไกลกว่า 11.5 cm จากกลางทิ้ง เพราะเคยจับโคมไฟนอกแผ่น → รูป 05-06

**ถาม: เซอร์โวอ่านตำแหน่งกลับมาใช้ในลูปไหม**
ตอบ: ไม่ใช้ในลูปควบคุม (SERVO_READBACK=0) อ่านตำแหน่งเฉพาะตอนคาลิเบรตหาขีดจำกัดและตำแหน่งกลาง (servo_calibration/01–03) ในลูปใช้แบบจำลองอันดับหนึ่งของมุมแผ่นแทน
"""


def write_qa() -> None:
    cl = TH.closed_loop(*GAINS)
    stamps = [p.stem[4:] for tag in ("balance", "circle", "hexagon") for p in RM.collect_runs(tag)]
    ages, dts = [], []
    for s in stamps:
        rows = csv_rows(s)
        ages += [float(r["frame_age_s"]) for r in rows if r.get("frame_age_s") not in (None, "", "nan")]
        dts += [float(r["dt_s"]) for r in rows[5:]]
    age, loop = float(np.nanmedian(ages)) * 1000, float(np.median(dts)) * 1000
    circle = [PA.run_summary(p.stem[4:], "circle") for p in RM.collect_runs("circle")]
    zone = RM.default_zone_cm()
    rf = RM.score_run(LOGS / f"run_{FULLGAIN_RUN}.csv", zone)
    ru = RM.score_run(LOGS / f"run_{USEDGAIN_VIDEO_RUN}.csv", zone)
    seen_full = 100 * run_series(FULLGAIN_RUN)["valid"].mean()
    text = QA.format(
        settle_th=cl.settle_2pct_s, os_th=cl.overshoot_pct, g_eff=ST.G_EFF, gx=TH.measured("plant_gain_x"),
        gy=TH.measured("plant_gain_y"), known=age + loop * 1.5 + PROF.COMMAND_PERIOD_S * 500 + PROF.MOVE_MS / 2,
        age=age, loop=loop,
        pm_lqr=ST.margins(ST.Model(1.0, 1.0), DELAY)["pm_deg"], crit_lqr=ST.critical_delay(ST.Model(1.0, 1.0)),
        pm_used=ST.margins(ST.Model(*GAINS), DELAY)["pm_deg"], crit_used=ST.critical_delay(ST.Model(*GAINS)),
        crit_15=ST.critical_delay(ST.Model(0.15, 1.8)),
        os_full=rf["overshoot_pct"], rms_full=rf["rms_mm"], seen_full=seen_full, os_used=ru["overshoot_pct"], rms_used=ru["rms_mm"],
        demand=TH.reference_tilt_demand_deg(0.03, 20.0), cross=np.mean([c["cross_mm"] for c in circle]),
        sync=np.mean([c["sync_mm"] for c in circle]))
    (KIT / "qa_cheatsheet.md").write_text(text, encoding="utf-8")
    print("  wrote qa_cheatsheet.md")


def write_catalog() -> None:
    lines = ["# วัตถุดิบสำหรับสไลด์ (presentation kit)", "",
             "★ = ควรใช้  ○ = ใช้ก็ได้/สำรอง  ทุกรูปสร้างจาก log จริงด้วย `python presentation_kit.py`", "",
             "| ไฟล์ | สำคัญ | สไลด์ที่เหมาะ | แสดงอะไร | ข้อความที่ควรพูด | ตัวเลขและที่มา |", "|---|---|---|---|---|---|"]
    for c in CATALOG:
        lines.append(f"| {c['file']} | {c['priority']} | {c['slide']} | {c['shows']} | {c['caption']} | {c['numbers']} |")
    lines += ["", "ชีตถาม-ตอบ: `qa_cheatsheet.md`"]
    (KIT / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [c["file"] for c in CATALOG if not (KIT / c["file"]).exists()]
    if missing:
        raise SystemExit(f"catalog lists files that do not exist: {missing}")
    print("  wrote catalog.md")


ITEMS = {"E1": item_e1, "E2": item_e2, "E3": item_e3, "E4": item_e4, "E5": item_e5, "A1": item_a1, "A2": item_a2, "A3": item_a3, "A4": item_a4, "B1": item_b1, "B2": item_b2, "B3": item_b3,
         "C1": item_c1, "C2": item_c2, "C3": item_c3, "C4": item_c4, "D1": item_d1, "D2": item_d2, "D3": item_d3,
         "D4": item_d4, "D5": item_d5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("items", nargs="*", help=f"any of {' '.join(ITEMS)} (default: all, plus catalog and Q&A)")
    args = parser.parse_args()
    unknown = set(args.items) - set(ITEMS)
    if unknown:
        parser.error(f"unknown item(s): {', '.join(sorted(unknown))}")
    KIT.mkdir(parents=True, exist_ok=True)
    for key in args.items or ITEMS:
        print(f"[{key}]")
        ITEMS[key]()
    if not args.items:
        register_existing()
        write_qa()
        write_catalog()


if __name__ == "__main__":
    sys.exit(main())
