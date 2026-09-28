"""
Step 1: ทดสอบกล้องและบันทึกคุณสมบัติจริง

ผลลัพธ์จะถูกบันทึกใน logs/camera_test_<timestamp>.json และ .csv
เพื่อใช้ตรวจสอบ format, resolution, FPS, timing และ intensity ของภาพ
ก่อนปรับ detector สำหรับกล้อง monochrome
"""

import cv2
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.path.append("/usr/lib/python3/dist-packages")
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# ══════════════════════════════════════════
#  ค่าคงที่ตั้งต้น
# ══════════════════════════════════════════
CAMERA_INDEX = 0           # OV9281 ถูกลงทะเบียนเป็น /dev/video0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 800
TARGET_FPS   = 30
TEST_DURATION_SECONDS = 60
MAX_CONSECUTIVE_READ_FAILURES = 30
LOG_DIR = Path(__file__).resolve().parent / "logs"
SHOW_PREVIEW = os.environ.get("CAMERA_TEST_PREVIEW", "0") == "1"


def fourcc_text(value: float) -> str:
    """แปลงค่า FOURCC ของ OpenCV เป็นข้อความที่อ่านได้"""
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def backend_name(cap: cv2.VideoCapture) -> str:
    """คืนชื่อ backend ที่ OpenCV ใช้งานจริง"""
    try:
        return cap.getBackendName()
    except cv2.error:
        return "unknown"


def select_backend() -> int:
    """เลือก backend ตามระบบปฏิบัติการ โดยให้ OpenCV default เป็น fallback"""
    if platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class Picamera2Adapter:
    """ทำให้ Picamera2 ใช้ interface เดียวกับ VideoCapture ในสคริปต์นี้"""

    def __init__(self):
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": TARGET_FPS},
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: FRAME_HEIGHT,
            cv2.CAP_PROP_FPS: TARGET_FPS,
            cv2.CAP_PROP_BACKEND: 0,
            cv2.CAP_PROP_FOURCC: 0,
        }
        return float(values.get(property_id, -1))

    def getBackendName(self) -> str:
        return "Picamera2/libcamera"

    def read(self):
        try:
            return True, self.camera.capture_array("main")
        except Exception:
            return False, None

    def release(self) -> None:
        self.camera.stop()
        self.camera.close()


def open_camera(index: int):
    """เปิด CSI ผ่าน Picamera2 หรือ fallback เป็น VideoCapture สำหรับ webcam"""
    if Picamera2 is not None:
        return Picamera2Adapter()

    cap = cv2.VideoCapture(index, select_backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


def draw_info(frame, fps: float, width: int, height: int,
              elapsed: float, read_failures: int) -> None:
    """วาดข้อมูล FPS และขนาดภาพบนหน้าจอ"""
    # กล่องพื้นหลังสำหรับข้อความ
    cv2.rectangle(frame, (0, 0), (440, 140), (0, 0, 0), -1)

    cv2.putText(frame, f"FPS    : {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Size   : {width} x {height}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Time   : {elapsed:.1f}/{TEST_DURATION_SECONDS}s",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Read failures: {read_failures}",
                (10, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "Press Q to quit",
                (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)


def frame_statistics(frame: np.ndarray) -> dict:
    """คำนวณสถิติที่ช่วยเลือก threshold สำหรับภาพ monochrome"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    percentiles = np.percentile(gray, [1, 5, 25, 50, 75, 95, 99])
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "ndim": int(frame.ndim),
        "channels": 1 if frame.ndim == 2 else int(frame.shape[2]),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def camera_properties(cap: cv2.VideoCapture) -> dict:
    """อ่านค่าที่ driver รายงานหลังจากตั้งค่าแล้ว"""
    property_ids = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "backend": cv2.CAP_PROP_BACKEND,
        "buffersize": cv2.CAP_PROP_BUFFERSIZE,
    }
    properties = {name: cap.get(prop_id) for name, prop_id in property_ids.items()}
    properties["backend_name"] = backend_name(cap)
    properties["fourcc"] = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
    properties["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties["requested_width"] = FRAME_WIDTH
    properties["requested_height"] = FRAME_HEIGHT
    properties["requested_fps"] = TARGET_FPS
    return properties


def main():
    print("=" * 40)
    print("  Step 1: Camera Test")
    print("=" * 40)

    # ── เปิดกล้อง ──────────────────────────
    cap = open_camera(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] ไม่สามารถเปิดกล้อง index={CAMERA_INDEX} ได้")
        print("        ตรวจ /dev/videoX หรือรัน rpicam-hello --list-cameras")
        return

    properties = camera_properties(cap)
    actual_w = properties["width"]
    actual_h = properties["height"]
    print(f"[OK]  เปิดกล้องสำเร็จ: {actual_w} x {actual_h}")
    print(f"[INFO] Backend: {properties['backend_name']} | FOURCC: {properties['fourcc']}")
    preview_status = "เปิด" if SHOW_PREVIEW else "ปิด"
    print(f"[INFO] เริ่มเก็บข้อมูล {TEST_DURATION_SECONDS} วินาที... Preview: {preview_status}")
    if SHOW_PREVIEW:
        print("[INFO] กด Q เพื่อออกก่อนเวลา")

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOG_DIR / f"camera_test_{timestamp}.json"
    csv_path = LOG_DIR / f"camera_test_{timestamp}.csv"
    frame_records = []
    fps = 0.0
    frame_count = 0
    read_failures = 0
    consecutive_read_failures = 0
    start_time = time.perf_counter()
    previous_frame_time = None

    while time.perf_counter() - start_time < TEST_DURATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            consecutive_read_failures += 1
            if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print(
                    f"[ERROR] อ่านเฟรมไม่ได้ต่อเนื่อง {consecutive_read_failures} ครั้ง"
                )
                print("        ตรวจว่า CAMERA_INDEX เป็น camera capture node ไม่ใช่ decoder/ISP node")
                print("        สำหรับกล้อง CSI ให้รัน: rpicam-hello --list-cameras")
                break
            continue

        consecutive_read_failures = 0
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - start_time
        interval_ms = None if previous_frame_time is None else (now - previous_frame_time) * 1000
        previous_frame_time = now
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        stats = frame_statistics(frame)
        stats.update({
            "frame_number": frame_count,
            "elapsed_seconds": elapsed,
            "interval_ms": interval_ms,
        })
        frame_records.append(stats)

        if SHOW_PREVIEW:
            display_frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 3 and isinstance(cap, Picamera2Adapter):
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            display_frame = cv2.flip(display_frame, 1)
            draw_info(display_frame, fps, actual_w, actual_h, elapsed, read_failures)
            cv2.imshow("Step 1 - Camera Test (Q=Quit)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    intervals = [record["interval_ms"] for record in frame_records
                 if record["interval_ms"] is not None]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "camera_index": CAMERA_INDEX,
        "test_duration_seconds": elapsed,
        "target_duration_seconds": TEST_DURATION_SECONDS,
        "frames_read": frame_count,
        "read_failures": read_failures,
        "average_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "interval_ms_min": min(intervals) if intervals else None,
        "interval_ms_median": float(np.median(intervals)) if intervals else None,
        "interval_ms_max": max(intervals) if intervals else None,
        "camera_properties": properties,
        "first_frame": frame_records[0] if frame_records else None,
        "last_frame": frame_records[-1] if frame_records else None,
        "notes": [
            "Frame statistics are grayscale-converted for multi-channel input.",
            "Displayed frames are mirrored only for preview; recorded statistics use the raw frame.",
            "Check shape, dtype, channels and FOURCC before selecting the detector conversion.",
        ],
    }
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    fieldnames = list(frame_records[0].keys()) if frame_records else []
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(frame_records)

    print(f"[DONE] ปิดกล้องแล้ว | FPS เฉลี่ย: {summary['average_fps']:.2f}")
    print(f"[INFO] Frames: {frame_count} | Read failures: {read_failures}")
    print(f"[SAVED] Summary: {json_path}")
    print(f"[SAVED] Per-frame data: {csv_path}")


if __name__ == "__main__":
    main()