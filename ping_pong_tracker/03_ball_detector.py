"""
Step 3: Ball Detector
- โหลดค่า grayscale จาก detection_config.json
- ตรวจจับลูกปิงปองด้วย intensity mask + Contour
- Resize ภาพลงก่อนประมวลผล เพื่อเพิ่ม FPS
- แสดงตำแหน่ง X, Y (pixel) real-time บนภาพ Full Resolution
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import freeze_support

try:
    sys.path.append("/usr/lib/python3/dist-packages")
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# ══════════════════════════════════════════
#  ค่าคงที่
# ══════════════════════════════════════════
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "detection_config.json")
CAMERA_FPS = 30
LOG_INTERVAL = 0.2  # วินาที ระหว่างการพิมพ์สถานะลง console
FPS_SMOOTHING_ALPHA = 0.12
MAX_WORKERS = max(2, min((os.cpu_count() or 2) - 1, 4))
MAX_IN_FLIGHT = MAX_WORKERS * 3
SHOW_PREVIEW = os.environ.get("BALL_DETECTOR_PREVIEW", "0") == "1"
CAPTURE_SECONDS = float(os.environ.get("BALL_DETECTOR_CAPTURE_SECONDS", "0"))
CAPTURE_DIR = Path(__file__).resolve().parent / "logs" / "detector_capture"

# ── Resize สำหรับการประมวลผล ──
PROC_WIDTH = 640
PROC_HEIGHT = 400
SCALE_X = FRAME_WIDTH / PROC_WIDTH
SCALE_Y = FRAME_HEIGHT / PROC_HEIGHT

# ── ขนาด radius ลูกปิงปองบนภาพ PROC size ──
BALL_RADIUS_MIN = 8
BALL_RADIUS_MAX = 80

# ── Morphological ──
MORPH_KERNEL_SIZE = 5

# ── Circularity threshold ──
CIRCULARITY_MIN = 0.60
HOUGH_MIN_RADIUS = 25
HOUGH_MAX_RADIUS = 60
HOUGH_PARAM2 = 22
MIN_CONFIDENCE = 0.45
MAX_TRACK_JUMP = 90
TRACK_HOLD_FRAMES = 5


# ══════════════════════════════════════════
#  โหลด Config
# ══════════════════════════════════════════
def load_detection_config() -> dict:
    """โหลดค่า grayscale detection configuration"""
    default = {
        "threshold_mode": "bright",
        "gray_min": 160,
        "gray_max": 255,
        "blur_kernel": 7,
        "morph_kernel": 5,
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        print(f"[INFO] โหลด grayscale: [{config['gray_min']},{config['gray_max']}]")
        return config
    print(f"[WARN] ไม่พบ {CONFIG_FILE} — ใช้ค่า default")
    return default


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
    """รับภาพ CSI ผ่าน libcamera และคืน RGB frame ให้ detector"""

    def __init__(self):
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": CAMERA_FPS},
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)
        _lock_ae_awb(self.camera)

    def isOpened(self) -> bool:
        return True

    def read(self):
        try:
            return True, self.camera.capture_array("main")
        except Exception:
            return False, None

    def release(self) -> None:
        self.camera.stop()
        self.camera.close()


def open_camera(index: int):
    """ใช้ Picamera2 กับ CSI; fallback เป็น V4L2 สำหรับ webcam"""
    if Picamera2 is not None:
        return Picamera2Adapter()

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap


class CameraStream:
    """อ่านกล้องใน thread แยก แล้วเก็บเฟรมล่าสุดไว้ให้ main loop"""

    def __init__(self, index: int):
        self.cap = open_camera(index)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> "CameraStream":
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self) -> None:
        while self.running:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.001)
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self) -> tuple[bool, int, np.ndarray | None]:
        with self.lock:
            if self.frame is None:
                return False, 0, None
            return True, self.frame_id, self.frame

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# ══════════════════════════════════════════
#  Image Processing
# ══════════════════════════════════════════
def preprocess(small_frame: np.ndarray, config: dict) -> np.ndarray:
    """สร้าง intensity mask จากภาพ PROC size"""
    gray = small_frame if small_frame.ndim == 2 else cv2.cvtColor(
        small_frame, cv2.COLOR_RGB2GRAY
    )
    blur_kernel = int(config.get("blur_kernel", 7)) | 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    threshold_type = cv2.THRESH_BINARY if config.get("threshold_mode", "bright") == "bright" else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(blurred, int(config["gray_min"]), 255, threshold_type)
    if config.get("gray_max", 255) < 255:
        max_mask = cv2.threshold(blurred, int(config["gray_max"]), 255, cv2.THRESH_BINARY_INV)[1]
        mask = cv2.bitwise_and(mask, max_mask)

    morph_size = int(config.get("morph_kernel", MORPH_KERNEL_SIZE)) | 1
    kernel = np.ones((morph_size, morph_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# ══════════════════════════════════════════
#  Detection
# ══════════════════════════════════════════
def detect_ball_contour(mask: np.ndarray) -> tuple | None:
    """ตรวจจับลูกปิงปองด้วย Contour บนภาพ PROC size"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_result = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (200 < area < 22000):
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
        if circularity < CIRCULARITY_MIN:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if not (BALL_RADIUS_MIN <= radius <= BALL_RADIUS_MAX):
            continue

        x, y, width, height = cv2.boundingRect(cnt)
        aspect_ratio = width / max(height, 1)
        if not (0.65 <= aspect_ratio <= 1.5):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / max(hull_area, 1e-5)
        if solidity < 0.75:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            confidence = min(1.0, max(0.0, circularity * solidity))
            best_result = (int(cx), int(cy), int(radius), confidence)

    return best_result


def detect_ball_hough(small_frame: np.ndarray, config: dict) -> tuple | None:
    """หาวงกลมสว่างที่เป็นก้อนทึบ ไม่ใช่เส้นสายหรือขอบเงา"""
    gray = small_frame if small_frame.ndim == 2 else cv2.cvtColor(
        small_frame, cv2.COLOR_RGB2GRAY
    )
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=80,
        param2=HOUGH_PARAM2,
        minRadius=int(config.get("hough_min_radius", HOUGH_MIN_RADIUS)),
        maxRadius=int(config.get("hough_max_radius", HOUGH_MAX_RADIUS)),
    )
    if circles is None:
        return None

    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    bright_mask = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY)[1]
    best_result = None
    best_score = 0.0
    for cx, cy, radius in np.round(circles[0]).astype(int):
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner = gray[distance <= radius * 0.65]
        ring = gray[(distance >= radius * 1.15) & (distance <= radius * 1.55)]
        if inner.size == 0 or ring.size == 0:
            continue
        contrast = float(np.mean(inner) - np.mean(ring))
        inner_fill = float(np.mean(bright_mask[distance <= radius * 0.65]) / 255.0)
        ring_fill = float(np.mean(bright_mask[(distance >= radius * 1.15) & (distance <= radius * 1.55)]) / 255.0)
        min_inner_fill = float(config.get("min_inner_fill", 0.55))
        min_fill_contrast = float(config.get("min_fill_contrast", 0.18))
        if contrast <= 8 or inner_fill < min_inner_fill or inner_fill - ring_fill < min_fill_contrast:
            continue
        expected_radius = float(config.get("expected_radius", 40))
        radius_score = max(0.0, 1.0 - abs(radius - expected_radius) / expected_radius)
        fill_score = min(1.0, max(0.0, (inner_fill - ring_fill) / 0.7))
        score = contrast * (1.0 + radius_score) * (0.5 + 0.5 * fill_score)
        if score > best_score:
            best_score = score
            radius_score = max(0.0, 1.0 - abs(radius - expected_radius) / expected_radius)
            contrast_score = min(1.0, max(0.0, (contrast - 8) / 80))
            confidence = contrast_score * (0.5 + 0.5 * radius_score) * (0.5 + 0.5 * fill_score)
            best_result = (int(cx), int(cy), int(radius), confidence)
    return best_result


def process_small_frame(small_frame: np.ndarray, config: dict) -> tuple[np.ndarray, tuple | None]:
    """ประมวลผลเฟรมย่อแบบแยก worker: preprocess + contour detection"""
    mask = preprocess(small_frame, config)
    result = detect_ball_hough(small_frame, config)
    if result is None:
        result = detect_ball_contour(mask)
    return mask, result


def scale_to_full(cx_s: int, cy_s: int, r_s: int) -> tuple:
    """แปลง pixel จาก PROC size → Full resolution"""
    cx_f = int(cx_s * SCALE_X)
    cy_f = int(cy_s * SCALE_Y)
    r_f = int(r_s * SCALE_X)
    return cx_f, cy_f, r_f


# ══════════════════════════════════════════
#  Drawing
# ══════════════════════════════════════════
def draw_detection(frame: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """วาดผลการตรวจจับบนภาพ Full resolution"""
    cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 0, 255), 2)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 0, 255), 2)
    label = f"({cx}, {cy}) px  r={radius}"
    cv2.putText(
        frame,
        label,
        (cx + 12, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def draw_status(frame: np.ndarray, detected: bool, cx: int = 0, cy: int = 0,
                fps: float = 0.0, confidence: float = 0.0) -> None:
    """แสดงสถานะที่มุมซ้ายบน"""
    cv2.rectangle(frame, (0, 0), (370, 122), (0, 0, 0), -1)

    status_color = (0, 255, 0) if detected else (0, 0, 255)
    status_text = "DETECTED ✓" if detected else "NOT FOUND ✗"
    cv2.putText(frame, f"Ball : {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if detected:
        cv2.putText(frame, f"Pos  : X={cx:4d} px  Y={cy:4d} px", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Conf : {confidence:.2f}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS  : {fps:.1f}", (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)


def draw_mask_preview(mask: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= 100:
            cv2.drawContours(preview, [contour], -1, (0, 0, 255), 2)
    return preview


def create_capture_writer(path: Path) -> cv2.VideoWriter:
    """สร้าง MP4 writer สำหรับภาพ BGR ที่บันทึกจาก headless mode"""
    return cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        CAMERA_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )


def draw_roi(frame: np.ndarray, config: dict) -> None:
    roi = config.get("roi", {})
    if not roi.get("enabled", False):
        return
    height, width = frame.shape[:2]
    x = int(float(roi.get("x", 0.0)) * width)
    y = int(float(roi.get("y", 0.0)) * height)
    right = int((float(roi.get("x", 0.0)) + float(roi.get("width", 1.0))) * width)
    bottom = int((float(roi.get("y", 0.0)) + float(roi.get("height", 1.0))) * height)
    cv2.rectangle(frame, (x, y), (right, bottom), (0, 255, 255), 3)
    cv2.putText(frame, "ROI", (x + 8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


# ══════════════════════════════════════════
#  Main
# ══════════════════════════════════════════
def main():
    print("=" * 45)
    print("  Step 3: Ball Detector (High FPS)")
    print("=" * 45)
    print(f"  Process size : {PROC_WIDTH} x {PROC_HEIGHT}")
    print(f"  Display size : {FRAME_WIDTH} x {FRAME_HEIGHT}")
    print(f"  Scale factor : {SCALE_X:.1f}x")
    print("  Q = ออก")
    if CAPTURE_SECONDS > 0:
        print(f"  Capture      : {CAPTURE_SECONDS:.1f}s -> {CAPTURE_DIR}")
    print("=" * 45)

    config = load_detection_config()
    cv2.setNumThreads(1)

    stream = CameraStream(CAMERA_INDEX)
    if not stream.cap.isOpened():
        print(f"[ERROR] เปิดกล้อง index={CAMERA_INDEX} ไม่ได้")
        stream.stop()
        return

    stream.start()
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

    process_fps = 0.0
    last_frame_id = 0
    last_completed_frame_id = None
    last_process_time = None
    last_print_time = 0.0
    pending_tasks: list[tuple[int, Future]] = []
    latest_mask = np.zeros((PROC_HEIGHT, PROC_WIDTH), dtype=np.uint8)
    latest_result: tuple | None = None
    tracked_result: tuple | None = None
    missed_frames = 0
    capture_start = time.perf_counter() if CAPTURE_SECONDS > 0 else None
    capture_frame_count = 0
    capture_writer = None
    if CAPTURE_SECONDS > 0:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        capture_timestamp = time.strftime("%Y%m%d_%H%M%S")
        capture_writer = create_capture_writer(
            CAPTURE_DIR / f"detector_{capture_timestamp}.mp4"
        )
        if not capture_writer.isOpened():
            print("[WARN] เปิด MP4 writer ไม่ได้ จะบันทึกเป็นภาพแทน")
            capture_writer = None

    win = "Step 3 - Ball Detector (Q=Quit)"
    if SHOW_PREVIEW:
        cv2.namedWindow(win)

    try:
        while True:
            ret, frame_id, frame = stream.read()
            if not ret:
                time.sleep(0.001)
                continue

            if frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame_id

            small = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)

            if len(pending_tasks) < MAX_IN_FLIGHT:
                future = executor.submit(process_small_frame, small, config)
                pending_tasks.append((frame_id, future))

            still_pending: list[tuple[int, Future]] = []
            for task_frame_id, future in pending_tasks:
                if future.done():
                    try:
                        mask, result = future.result()
                        latest_mask = mask
                        if result is not None and result[3] >= MIN_CONFIDENCE:
                            if tracked_result is None:
                                tracked_result = result
                                missed_frames = 0
                            else:
                                jump = np.hypot(
                                    result[0] - tracked_result[0],
                                    result[1] - tracked_result[1],
                                )
                                if jump <= MAX_TRACK_JUMP:
                                    tracked_result = result
                                    missed_frames = 0
                                else:
                                    missed_frames += 1
                        else:
                            missed_frames += 1
                        if missed_frames > TRACK_HOLD_FRAMES:
                            tracked_result = None
                        latest_result = tracked_result

                        completed_now = time.perf_counter()
                        if last_process_time is not None and last_completed_frame_id is not None:
                            dt = completed_now - last_process_time
                            if dt > 0:
                                frame_delta = max(1, task_frame_id - last_completed_frame_id)
                                instant_fps = frame_delta / dt
                                instant_fps = min(instant_fps, CAMERA_FPS)
                                if process_fps == 0.0:
                                    process_fps = instant_fps
                                else:
                                    process_fps = (
                                        FPS_SMOOTHING_ALPHA * instant_fps
                                        + (1.0 - FPS_SMOOTHING_ALPHA) * process_fps
                                    )
                        last_process_time = completed_now
                        last_completed_frame_id = task_frame_id
                    except Exception as exc:
                        print(f"[ERROR] worker ล้มเหลว frame={task_frame_id}: {exc}")
                else:
                    still_pending.append((task_frame_id, future))
            pending_tasks = still_pending

            if latest_result:
                cx_s, cy_s, r_s, confidence = latest_result
                cx, cy, r = scale_to_full(cx_s, cy_s, r_s)
                draw_detection(frame, cx, cy, r)
                draw_status(frame, detected=True, cx=cx, cy=cy, fps=process_fps,
                            confidence=confidence)

                now = time.time()
                if now - last_print_time >= LOG_INTERVAL:
                    print(
                        f"\r[BALL] X={cx:4d} Y={cy:4d} px  r={r:3d}  FPS={process_fps:.1f}   ",
                        end="",
                        flush=True,
                    )
                    last_print_time = now
            else:
                draw_status(frame, detected=False, fps=process_fps)

            if CAPTURE_SECONDS > 0:
                capture_frame_count += 1
                output_frame = frame.copy()
                draw_roi(output_frame, config)
                if latest_result:
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                else:
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                if capture_writer is not None:
                    capture_writer.write(output_frame)
                if capture_frame_count % max(1, int(CAMERA_FPS)) == 0:
                    image_path = CAPTURE_DIR / f"frame_{capture_frame_count:05d}.jpg"
                    mask_path = CAPTURE_DIR / f"mask_{capture_frame_count:05d}.png"
                    cv2.imwrite(str(image_path), output_frame)
                    cv2.imwrite(str(mask_path), latest_mask)

            if SHOW_PREVIEW:
                mask_small = cv2.resize(
                    draw_mask_preview(latest_mask),
                    (320, 180),
                    interpolation=cv2.INTER_NEAREST,
                )
                frame[0:180, FRAME_WIDTH - 320:FRAME_WIDTH] = mask_small
                cv2.imshow(win, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if capture_start is not None and time.perf_counter() - capture_start >= CAPTURE_SECONDS:
                break

        print()
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        stream.stop()
        if capture_writer is not None:
            capture_writer.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        print("[DONE] ปิด Ball Detector แล้ว")


if __name__ == "__main__":
    freeze_support()
    main()
