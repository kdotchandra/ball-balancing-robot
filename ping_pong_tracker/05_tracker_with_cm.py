"""
Step 5: Full Tracker — ตำแหน่งจริง X, Y (cm)
──────────────────────────────────────────────
รวม grayscale Detection + Calibration + แสดงผลครบ:
  - ตำแหน่ง X, Y หน่วย pixel
  - ตำแหน่ง X, Y หน่วย cm จากมุมซ้ายบนของภาพ
  - ตำแหน่ง X, Y หน่วย cm จากจุดกึ่งกลางภาพ (Origin ที่ศูนย์กลาง)
  - FPS
  - บันทึก log ลง CSV (กด L เพื่อเปิด/ปิด)
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import csv
from datetime import datetime

try:
    sys.path.append("/usr/lib/python3/dist-packages")
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# ══════════════════════════════════════════
#  ค่าคงที่
# ══════════════════════════════════════════
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT = 800
DETECTION_CONFIG = os.path.join(os.path.dirname(__file__), "detection_config.json")
CALIB_CONFIG  = os.path.join(os.path.dirname(__file__), "calib_config.json")
LOG_DIR       = os.path.join(os.path.dirname(__file__), "logs")
SHOW_PREVIEW  = os.environ.get("BALL_TRACKER_PREVIEW", "0") == "1"

BALL_R_MIN     = 8
BALL_R_MAX     = 80
CIRCULARITY_MIN = 0.60
HOUGH_MIN_RADIUS = 25
HOUGH_MAX_RADIUS = 60
HOUGH_PARAM2 = 22
MIN_CONFIDENCE = 0.45
MAX_TRACK_JUMP = 180
TRACK_HOLD_FRAMES = 5

# Morphological
MORPH_K        = 5


# ══════════════════════════════════════════
#  โหลด config
# ══════════════════════════════════════════
def load_json(filepath: str, default: dict) -> dict:
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
        print(f"[INFO] โหลด {filepath}")
        return data
    print(f"[WARN] ไม่พบ {filepath} ใช้ค่า default")
    return default


# ══════════════════════════════════════════
#  Detection
# ══════════════════════════════════════════
def get_mask(frame: np.ndarray, detection_cfg: dict) -> np.ndarray:
    """สร้าง intensity mask ของลูกปิงปอง"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur_kernel = int(detection_cfg.get("blur_kernel", 7)) | 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    threshold_type = (
        cv2.THRESH_BINARY
        if detection_cfg.get("threshold_mode", "bright") == "bright"
        else cv2.THRESH_BINARY_INV
    )
    _, mask = cv2.threshold(blurred, int(detection_cfg.get("gray_min", 160)), 255, threshold_type)
    gray_max = int(detection_cfg.get("gray_max", 255))
    if gray_max < 255:
        upper_mask = cv2.threshold(blurred, gray_max, 255, cv2.THRESH_BINARY_INV)[1]
        mask = cv2.bitwise_and(mask, upper_mask)
    morph_size = int(detection_cfg.get("morph_kernel", MORPH_K)) | 1
    k = np.ones((morph_size, morph_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def detect_ball_hough(frame: np.ndarray) -> tuple | None:
    """หาวงกลมสว่างจากภาพ grayscale โดยเทียบกับวงแหวนรอบวงกลม"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=80, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS,
    )
    if circles is None:
        return None
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    best = None
    best_score = 0.0
    for cx, cy, radius in np.round(circles[0]).astype(int):
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner = gray[distance <= radius * 0.65]
        ring = gray[(distance >= radius * 1.15) & (distance <= radius * 1.55)]
        if inner.size == 0 or ring.size == 0:
            continue
        contrast = float(np.mean(inner) - np.mean(ring))
        if contrast <= 12:
            continue
        radius_score = max(0.0, 1.0 - abs(radius - 40) / 40)
        score = contrast * (1.0 + radius_score)
        if score > best_score:
            best_score = score
            radius_score = max(0.0, 1.0 - abs(radius - 40) / 40)
            contrast_score = min(1.0, max(0.0, (contrast - 12) / 80))
            confidence = contrast_score * (0.5 + 0.5 * radius_score)
            best = (int(cx), int(cy), int(radius), confidence)
    return best


def detect_ball(mask: np.ndarray, frame: np.ndarray) -> tuple | None:
    """ใช้ Hough กับภาพจริงก่อน แล้วใช้ contour เป็น fallback"""
    hough_result = detect_ball_hough(frame)
    if hough_result is not None:
        return hough_result

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 200 < area < 22000:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
        if circularity < CIRCULARITY_MIN:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / max(height, 1)
        if not 0.65 <= aspect_ratio <= 1.5:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not BALL_R_MIN <= radius <= BALL_R_MAX:
            continue
        solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1e-5)
        if solidity < 0.75:
            continue
        score = area * circularity
        if score > best_score:
            best_score = score
            confidence = min(1.0, max(0.0, circularity * solidity))
            best = (int(cx), int(cy), int(radius), confidence)
    return best


def _lock_ae_awb(camera) -> None:
    try:
        for _ in range(20):
            camera.capture_metadata()
            time.sleep(0.06)
        meta = camera.capture_metadata()
        controls = {
            "AeEnable": False,
            "ExposureTime": int(meta["ExposureTime"]),
            "AnalogueGain": float(meta["AnalogueGain"]),
        }
        colour_gains = meta.get("ColourGains")
        if colour_gains is not None:
            # Absent on monochrome sensors (e.g. OV9281 mono) -- nothing to white-balance.
            controls["AwbEnable"] = False
            controls["ColourGains"] = colour_gains
        camera.set_controls(controls)
        print(
            f"[Camera] Locked exposure: exposure={meta['ExposureTime']}us "
            f"gain={meta['AnalogueGain']:.2f} colour_gains={colour_gains}"
        )
    except Exception as exc:
        print(f"[Camera] Could not lock AE/AWB, leaving auto-exposure enabled: {exc}")


class Picamera2Adapter:
    def __init__(self):
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": 30},
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)
        _lock_ae_awb(self.camera)

    def isOpened(self):
        return True

    def read(self):
        try:
            return True, self.camera.capture_array("main")
        except Exception:
            return False, None

    def release(self):
        self.camera.stop()
        self.camera.close()


def open_camera(index: int):
    if Picamera2 is not None:
        return Picamera2Adapter()
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


# ══════════════════════════════════════════
#  พิกเซล → cm
# ══════════════════════════════════════════
def px_to_cm(px: int, py: int,
             pixel_per_cm: float,
             frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """
    แปลงพิกเซลเป็น cm
    คืนค่า:
      (x_from_topleft_cm, y_from_topleft_cm,
       x_from_center_cm,  y_from_center_cm)
    """
    x_tl = px / pixel_per_cm
    y_tl = py / pixel_per_cm

    cx_px = frame_w / 2
    cy_px = frame_h / 2
    x_c   = (px - cx_px) / pixel_per_cm
    y_c   = (cy_px - py) / pixel_per_cm   # Y บวก = ขึ้น

    return x_tl, y_tl, x_c, y_c


# ══════════════════════════════════════════
#  วาดผลลัพธ์
# ══════════════════════════════════════════
def draw_crosshair_center(frame: np.ndarray) -> None:
    """วาด crosshair ที่ศูนย์กลางภาพ (origin)"""
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (255, 255, 0), 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (255, 255, 0), 1)
    cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
    cv2.putText(frame, "ORIGIN (0,0)", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_ball(frame: np.ndarray,
              cx: int, cy: int, r: int,
              x_c: float, y_c: float,
              has_calib: bool) -> None:
    """วาดวงกลมและตำแหน่งบอล"""
    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
    # เส้นจาก origin → ball
    cv2.line(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
             (cx, cy), (255, 100, 0), 1)
    unit = "cm" if has_calib else "px"
    label = f"({x_c:+.1f}, {y_c:+.1f}) {unit}"
    cv2.putText(frame, label, (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def draw_info_panel(frame: np.ndarray,
                    detected: bool,
                    cx: int, cy: int,
                    x_tl: float, y_tl: float,
                    x_c: float,  y_c: float,
                    fps: float,
                    logging: bool,
                    has_calib: bool) -> None:
    """แสดงข้อมูลทั้งหมดที่มุมซ้ายบน"""
    cv2.rectangle(frame, (0, 0), (420, 185), (0, 0, 0), -1)

    status_col = (0, 255, 0) if detected else (0, 0, 255)
    status_txt = "DETECTED ✓" if detected else "NOT FOUND ✗"
    cv2.putText(frame, f"Ball   : {status_txt}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

    if detected:
        cv2.putText(frame, f"Pixel  : X={cx:4d}  Y={cy:4d} px",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        if has_calib:
            cv2.putText(frame, f"From TL: X={x_tl:6.1f}  Y={y_tl:6.1f} cm",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Center : X={x_c:+6.1f}  Y={y_c:+6.1f} cm",
                        (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "ยังไม่ได้ calibrate (ดู Step 4)",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    log_col = (0, 255, 100) if logging else (150, 150, 150)
    log_txt = "ON (L=หยุด)" if logging else "OFF (L=เริ่ม)"
    cv2.putText(frame, f"Log    : {log_txt}",
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, log_col, 1)
    cv2.putText(frame, f"FPS    : {fps:.1f}   Q=ออก",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ══════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════
class CSVLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(LOG_DIR, f"track_{ts}.csv")
        self.file   = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["timestamp", "cx_px", "cy_px",
             "x_tl_cm", "y_tl_cm", "x_center_cm", "y_center_cm"]
        )
        print(f"[LOG] บันทึกลง {filepath}")

    def write(self, cx, cy, x_tl, y_tl, x_c, y_c):
        self.writer.writerow(
            [f"{time.time():.4f}", cx, cy,
             f"{x_tl:.3f}", f"{y_tl:.3f}",
             f"{x_c:.3f}", f"{y_c:.3f}"]
        )

    def close(self):
        self.file.close()
        print("[LOG] ปิด CSV logger")


# ══════════════════════════════════════════
#  Main
# ══════════════════════════════════════════
def main():
    print("=" * 50)
    print("  Step 5: Full Tracker with CM Position")
    print("=" * 50)
    print("  L = เปิด/ปิด CSV logging")
    print("  Q = ออก")
    print("=" * 50)

    # โหลด config
    detection_cfg = load_json(DETECTION_CONFIG, {
        "threshold_mode": "bright", "gray_min": 160,
        "gray_max": 255, "blur_kernel": 7, "morph_kernel": 5
    })
    calib_cfg = load_json(CALIB_CONFIG, {})
    ppc       = calib_cfg.get("pixel_per_cm")   # pixel per cm
    has_calib = ppc is not None

    if has_calib:
        print(f"[CALIB] pixel_per_cm = {ppc:.4f}  (1px = {1/ppc*10:.3f} mm)")
    else:
        print("[WARN] ไม่มีค่า calibration — จะแสดงเฉพาะ pixel")

    cap = open_camera(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] เปิดกล้อง index={CAMERA_INDEX} ไม่ได้")
        return

    fps         = 0.0
    frame_count = 0
    start_time  = time.time()
    logging     = False
    logger: CSVLogger | None = None
    tracked_result = None
    missed_frames = 0

    win = "Step 5 - Tracker (L=Log, Q=Quit)"
    if SHOW_PREVIEW:
        cv2.namedWindow(win)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror กล้องให้ภาพเหมือนกระจก
        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)

        # ── ตรวจจับ ────────────────────────
        mask   = get_mask(frame, detection_cfg)
        result = detect_ball(mask, frame)
        if result is not None and result[3] >= MIN_CONFIDENCE:
            if tracked_result is None:
                tracked_result = result
                missed_frames = 0
            else:
                jump = np.hypot(result[0] - tracked_result[0], result[1] - tracked_result[1])
                if jump <= MAX_TRACK_JUMP:
                    tracked_result = result
                    missed_frames = 0
                else:
                    missed_frames += 1
        else:
            missed_frames += 1
        if missed_frames > TRACK_HOLD_FRAMES:
            tracked_result = None
        result = tracked_result

        # ── แปลงหน่วย + วาดผล ──────────────
        x_tl = y_tl = x_c = y_c = 0.0

        if result:
            cx, cy, r, confidence = result
            if has_calib:
                x_tl, y_tl, x_c, y_c = px_to_cm(cx, cy, ppc,
                                                  FRAME_WIDTH, FRAME_HEIGHT)
            else:
                # Fallback: still show meaningful center offsets in pixels.
                x_c = float(cx - (FRAME_WIDTH / 2))
                y_c = float((FRAME_HEIGHT / 2) - cy)
            draw_crosshair_center(frame)
            draw_ball(frame, cx, cy, r, x_c, y_c, has_calib)
            draw_info_panel(frame, True, cx, cy,
                            x_tl, y_tl, x_c, y_c,
                            fps, logging, has_calib)

            # log ข้อมูล
            if logging and logger and has_calib:
                logger.write(cx, cy, x_tl, y_tl, x_c, y_c)

            if has_calib:
                print(f"\r[TRACK] px=({cx:4d},{cy:4d}) "
                    f"cm_center=({x_c:+6.1f},{y_c:+6.1f})  FPS={fps:.1f}  ",
                    end="", flush=True)
            else:
                print(f"\r[TRACK] px=({cx:4d},{cy:4d}) "
                    f"px_center=({x_c:+7.1f},{y_c:+7.1f})  FPS={fps:.1f}  ",
                    end="", flush=True)
        else:
            draw_crosshair_center(frame)
            draw_info_panel(frame, False, 0, 0, 0, 0, 0, 0,
                            fps, logging, has_calib)

        # แสดง mask มุมขวาบน
        if SHOW_PREVIEW:
            mask_small = cv2.resize(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (320, 180)
            )
            frame[0:180, FRAME_WIDTH - 320:FRAME_WIDTH] = mask_small
            cv2.imshow(win, frame)

            # ── keyboard ───────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                if not logging:
                    logger  = CSVLogger()
                    logging = True
                    print("\n[LOG] เริ่ม logging...")
                else:
                    if logger:
                        logger.close()
                    logging = False
                    print("\n[LOG] หยุด logging")

    print()
    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    if logger:
        logger.close()
    print("[DONE] ปิด Tracker แล้ว")


if __name__ == "__main__":
    main()