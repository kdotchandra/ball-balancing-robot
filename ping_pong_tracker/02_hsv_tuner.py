"""
Step 2: HSV Color Tuner
- ปรับค่า HSV แบบ real-time ด้วย Trackbar
- แสดง Original / HSV Mask / Result พร้อมกัน
- บันทึกค่า HSV ที่ดีที่สุดลงไฟล์ hsv_config.json
"""

import cv2
import numpy as np
import json
import os

# ══════════════════════════════════════════
#  ค่าคงที่
# ══════════════════════════════════════════
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
CONFIG_FILE   = "hsv_config.json"

# ค่า HSV เริ่มต้นสำหรับสีส้ม (ลูกปิงปอง)
DEFAULT_HSV = {
    "h_min":  5,  "h_max": 25,
    "s_min": 150, "s_max": 255,
    "v_min": 150, "v_max": 255
}


def load_hsv_config() -> dict:
    """โหลดค่า HSV จากไฟล์ถ้ามี"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        print(f"[INFO] โหลดค่า HSV จาก {CONFIG_FILE}")
        return config
    return DEFAULT_HSV.copy()


def save_hsv_config(config: dict) -> None:
    """บันทึกค่า HSV ลงไฟล์"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] บันทึกค่า HSV ไปที่ {CONFIG_FILE}")
    print(f"        H: [{config['h_min']}, {config['h_max']}]")
    print(f"        S: [{config['s_min']}, {config['s_max']}]")
    print(f"        V: [{config['v_min']}, {config['v_max']}]")


def create_trackbars(window_name: str, config: dict) -> None:
    """สร้าง Trackbar ทั้งหมด"""
    cv2.createTrackbar("H Min", window_name, config["h_min"], 179, lambda x: None)
    cv2.createTrackbar("H Max", window_name, config["h_max"], 179, lambda x: None)
    cv2.createTrackbar("S Min", window_name, config["s_min"], 255, lambda x: None)
    cv2.createTrackbar("S Max", window_name, config["s_max"], 255, lambda x: None)
    cv2.createTrackbar("V Min", window_name, config["v_min"], 255, lambda x: None)
    cv2.createTrackbar("V Max", window_name, config["v_max"], 255, lambda x: None)


def get_trackbar_values(window_name: str) -> dict:
    """อ่านค่าจาก Trackbar"""
    return {
        "h_min": cv2.getTrackbarPos("H Min", window_name),
        "h_max": cv2.getTrackbarPos("H Max", window_name),
        "s_min": cv2.getTrackbarPos("S Min", window_name),
        "s_max": cv2.getTrackbarPos("S Max", window_name),
        "v_min": cv2.getTrackbarPos("V Min", window_name),
        "v_max": cv2.getTrackbarPos("V Max", window_name),
    }


def apply_hsv_mask(frame: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    แปลง BGR → HSV แล้วสร้าง Mask ตามค่า HSV
    คืนค่า: (mask, result)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([config["h_min"], config["s_min"], config["v_min"]])
    upper = np.array([config["h_max"], config["s_max"], config["v_max"]])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations เพื่อลด noise
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.erode(mask,  kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=2)

    # ใส่ mask กลับบนภาพต้นฉบับ
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return mask, result


def draw_hsv_info(frame: np.ndarray, config: dict) -> None:
    """แสดงค่า HSV ปัจจุบันบนภาพ"""
    cv2.rectangle(frame, (0, 0), (340, 110), (0, 0, 0), -1)
    cv2.putText(frame,
                f"H: [{config['h_min']:3d}, {config['h_max']:3d}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame,
                f"S: [{config['s_min']:3d}, {config['s_max']:3d}]",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame,
                f"V: [{config['v_min']:3d}, {config['v_max']:3d}]",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame,
                "S=Save  Q=Quit",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)


def build_display(original: np.ndarray,
                  mask: np.ndarray,
                  result: np.ndarray) -> np.ndarray:
    """
    รวม 3 ภาพแสดงพร้อมกัน:
    [Original | Mask | Result]
    """
    h, w = original.shape[:2]
    small_w, small_h = w // 2, h // 2

    # ย่อภาพให้เล็กลงครึ่งหนึ่ง
    orig_small   = cv2.resize(original, (small_w, small_h))
    mask_bgr     = cv2.cvtColor(cv2.resize(mask,   (small_w, small_h)), cv2.COLOR_GRAY2BGR)
    result_small = cv2.resize(result,   (small_w, small_h))

    # Label แต่ละภาพ
    for img, label in zip([orig_small, mask_bgr, result_small],
                          ["Original", "HSV Mask", "Result"]):
        cv2.putText(img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ต่อภาพแนวนอน (Original บนซ้าย, Mask บนขวา, Result ล่างกลาง)
    top_row    = np.hstack([orig_small, mask_bgr])
    # Result วางตรงกลาง
    pad_left   = small_w // 2
    pad_right  = small_w - pad_left  # กัน rounding error
    bottom_row = np.hstack([
        np.zeros((small_h, pad_left,  3), dtype=np.uint8),
        result_small,
        np.zeros((small_h, pad_right, 3), dtype=np.uint8)
    ])

    return np.vstack([top_row, bottom_row])


def main():
    print("=" * 40)
    print("  Step 2: HSV Color Tuner")
    print("=" * 40)
    print("  S = บันทึกค่า HSV")
    print("  Q = ออก")
    print("=" * 40)

    config  = load_hsv_config()
    cap     = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] เปิดกล้อง index={CAMERA_INDEX} ไม่ได้")
        return

    # สร้าง window และ trackbar
    win = "Step 2 - HSV Tuner (S=Save, Q=Quit)"
    cv2.namedWindow(win)
    create_trackbars(win, config)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror กล้องให้ภาพเหมือนกระจก
        frame = cv2.flip(frame, 1)

        # อ่านค่า HSV จาก trackbar
        config = get_trackbar_values(win)

        # สร้าง mask และผลลัพธ์
        mask, result = apply_hsv_mask(frame, config)

        # วาดข้อมูลบนภาพต้นฉบับ
        draw_hsv_info(frame, config)

        # รวมและแสดงภาพ
        display = build_display(frame, mask, result)
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_hsv_config(config)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] ปิด HSV Tuner แล้ว")


if __name__ == "__main__":
    main()