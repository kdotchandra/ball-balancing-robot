"""
Step 4: Camera Calibration (Pixel per CM)
─────────────────────────────────────────
วิธีใช้:
  1. วางวัตถุอ้างอิงที่รู้ขนาดจริงบนพื้น (เช่น กระดาษ A4 = 21 x 29.7 cm)
  2. กด 'M' เพื่อเข้าโหมดวัด แล้วคลิก 2 จุดบนภาพ
  3. กรอกระยะจริง (cm) ที่ console
  4. โปรแกรมคำนวณ pixel_per_cm และบันทึกลง calib_config.json
  5. กด 'S' เพื่อบันทึก / 'R' เพื่อรีเซ็ต / 'Q' เพื่อออก
"""

import cv2
import numpy as np
import json
import os
import math
import sys
import time
import argparse

sys.path.append("/usr/lib/python3/dist-packages")
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


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

# ══════════════════════════════════════════
#  ค่าคงที่
# ══════════════════════════════════════════
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
CALIB_FILE    = os.path.join(os.path.dirname(__file__), "calib_config.json")


# ══════════════════════════════════════════
#  Global state สำหรับ mouse callback
# ══════════════════════════════════════════
points: list       = []   # จุดที่คลิก [(x1,y1), (x2,y2)]
measuring: bool    = False


def mouse_callback(event, x, y, flags, param):
    """รับการคลิกเมาส์เพื่อเลือกจุดวัด"""
    global points, measuring
    if not measuring:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"  [คลิก {len(points)}] ({x}, {y})")


def pixel_distance(p1: tuple, p2: tuple) -> float:
    """คำนวณระยะห่างระหว่าง 2 จุด (pixel)"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def draw_measurement(frame: np.ndarray, pts: list,
                     ppc: float | None) -> None:
    """วาดเส้นวัดและแสดงค่าบนภาพ"""
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 8, (0, 255, 255), -1)
        cv2.putText(frame, f"P{i+1}", (p[0] + 10, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(pts) == 2:
        cv2.line(frame, pts[0], pts[1], (0, 255, 0), 2)
        mid = ((pts[0][0] + pts[1][0]) // 2,
               (pts[0][1] + pts[1][1]) // 2)
        dist_px = pixel_distance(pts[0], pts[1])
        label   = f"{dist_px:.1f} px"
        if ppc and ppc > 0:
            dist_cm  = dist_px / ppc
            label   += f" = {dist_cm:.2f} cm"
        cv2.putText(frame, label, (mid[0] + 10, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_instructions(frame: np.ndarray, measuring: bool,
                      ppc: float | None) -> None:
    """แสดงคำแนะนำและค่า pixel_per_cm"""
    cv2.rectangle(frame, (0, 0), (430, 130), (0, 0, 0), -1)
    mode = "MEASURING MODE" if measuring else "VIEW MODE"
    cv2.putText(frame, mode, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if measuring else (150, 150, 150), 2)
    cv2.putText(frame, "M = เริ่มวัด | R = รีเซ็ต | S = บันทึก | Q = ออก",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if ppc:
        cv2.putText(frame, f"pixel/cm = {ppc:.4f}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"1 cm = {ppc:.1f} px  |  1 px = {1/ppc*10:.3f} mm",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)
    else:
        cv2.putText(frame, "ยังไม่ได้ calibrate",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)


def save_calibration(ppc: float) -> None:
    """บันทึกค่า calibration"""
    config = {"pixel_per_cm": ppc}
    with open(CALIB_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[SAVED] บันทึก calibration ไปที่ {CALIB_FILE}")
    print(f"        pixel_per_cm = {ppc:.4f}")
    print(f"        1 px = {1/ppc*10:.3f} mm")


def calibrate_from_checkerboard(frame: np.ndarray) -> float:
    """Estimate pixel/cm from a 7x9-square board with 20 mm squares."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    patterns = ((7, 9), (9, 7), (8, 6), (6, 8))
    found = False
    corners = None
    pattern = patterns[0]
    for candidate in patterns:
        found, corners = cv2.findChessboardCornersSB(gray, candidate)
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray,
                candidate,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        if found:
            pattern = candidate
            break
    if not found or corners is None:
        raise RuntimeError("Checkerboard not detected; keep the full board visible and well lit")

    points = corners.reshape(pattern[1], pattern[0], 2)
    horizontal = np.linalg.norm(np.diff(points, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(points, axis=0), axis=2)
    square_px = float(np.mean(np.concatenate((horizontal.ravel(), vertical.ravel()))))
    pixel_per_cm = square_px / 2.0
    print(f"[INFO] checkerboard square spacing={square_px:.3f} px")
    print(f"[INFO] pixel_per_cm={pixel_per_cm:.4f}")
    return pixel_per_cm


def main():
    global points, measuring

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless-checkerboard",
        action="store_true",
        help="Detect a visible 7x9-square, 20 mm checkerboard without a GUI",
    )
    parser.add_argument(
        "--capture",
        metavar="PATH",
        help="Save one camera frame to PATH without opening a GUI",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 4: Camera Calibration")
    print("=" * 50)
    print("  1. วางวัตถุอ้างอิงบนพื้น (เช่น กระดาษ A4)")
    print("  2. กด M แล้วคลิก 2 จุดที่รู้ระยะจริง")
    print("  3. กรอกระยะจริง (cm) ที่ keyboard")
    print("  4. กด S เพื่อบันทึก")
    print("=" * 50)

    # โหลดค่าเดิมถ้ามี
    pixel_per_cm: float | None = None
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            cfg = json.load(f)
        pixel_per_cm = cfg.get("pixel_per_cm")
        print(f"[INFO] โหลดค่าเดิม: pixel_per_cm = {pixel_per_cm:.4f}")

    backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_DSHOW
    cap = None
    camera = None
    if Picamera2 is not None:
        camera = Picamera2()
        config = camera.create_video_configuration(
            main={"size": (FRAME_WIDTH, 800), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(0.5)
        _lock_ae_awb(camera)
        print("[INFO] เปิดกล้องด้วย Picamera2/libcamera")
    else:
        backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            print(f"[ERROR] เปิดกล้อง index={CAMERA_INDEX} ไม่ได้")
            return

    if args.capture or args.headless_checkerboard:
        try:
            frame = None
            for _ in range(10):
                frame = camera.capture_array() if camera is not None else cap.read()[1]
                if frame is not None:
                    break
            if frame is None:
                raise RuntimeError("Camera returned no frame")

            if args.capture:
                output_path = os.path.abspath(args.capture)
                if not cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Could not save camera frame to {output_path}")
                print(f"[SAVED] camera frame: {output_path}")

            if args.headless_checkerboard:
                pixel_per_cm = None
                for _ in range(30):
                    frame = camera.capture_array() if camera is not None else cap.read()[1]
                    if frame is None:
                        continue
                    try:
                        pixel_per_cm = calibrate_from_checkerboard(frame)
                        break
                    except RuntimeError:
                        continue
                try:
                    if pixel_per_cm is None:
                        raise RuntimeError("Checkerboard not detected after 30 frames; keep the full board visible and well lit")
                    save_calibration(pixel_per_cm)
                except RuntimeError:
                    raise
        finally:
            if camera is not None:
                camera.stop()
                camera.close()
            else:
                cap.release()
        return

    win = "Step 4 - Calibration (M=Measure, S=Save, Q=Quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        if camera is not None:
            frame = camera.capture_array()
            ret = frame is not None
        else:
            ret, frame = cap.read()
        if not ret:
            break

        # Mirror กล้องให้ภาพเหมือนกระจก
        frame = cv2.flip(frame, 1)

        # ตรวจว่าคลิกครบ 2 จุดในโหมดวัดหรือยัง
        if measuring and len(points) == 2:
            dist_px = pixel_distance(points[0], points[1])
            print(f"\n[วัดได้] {dist_px:.1f} pixel ระหว่าง {points[0]} → {points[1]}")
            real_cm_str = input("  กรอกระยะจริง (cm): ").strip()
            try:
                real_cm      = float(real_cm_str)
                pixel_per_cm = dist_px / real_cm
                print(f"  ✅ pixel_per_cm = {pixel_per_cm:.4f}")
            except ValueError:
                print("  [ERROR] ค่าไม่ถูกต้อง ลองใหม่")
            measuring = False
            points    = []

        draw_measurement(frame, points, pixel_per_cm)
        draw_instructions(frame, measuring, pixel_per_cm)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            measuring = True
            points    = []
            print("\n[MODE] Measuring — คลิก 2 จุดบนภาพ")
        elif key == ord('r'):
            measuring    = False
            points       = []
            pixel_per_cm = None
            print("[RESET] รีเซ็ตค่าทั้งหมด")
        elif key == ord('s'):
            if pixel_per_cm:
                save_calibration(pixel_per_cm)
            else:
                print("[WARN] ยังไม่มีค่า calibration")
        elif key == ord('q'):
            break

    if camera is not None:
        camera.stop()
        camera.close()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("[DONE] ปิด Calibration แล้ว")


if __name__ == "__main__":
    main()