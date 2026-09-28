# Ping Pong Ball Tracker — POC

ตรวจจับลูกปิงปองสีส้มและหาตำแหน่ง X, Y บนพื้น

## โครงสร้างไฟล์

```
ping_pong_tracker/
├── 01_test_camera.py       # ทดสอบกล้อง
├── 02_hsv_tuner.py         # ปรับค่าสี HSV
├── 03_ball_detector.py     # ตรวจจับบอล (pixel)
├── 04_calibration.py       # Calibrate pixel → cm
├── 05_tracker_with_cm.py   # Tracker ครบ + บันทึก CSV
├── hsv_config.json         # (สร้างอัตโนมัติจาก Step 2)
├── detection_config.json   # ค่า grayscale threshold สำหรับ OV9281
├── calib_config.json       # (สร้างอัตโนมัติจาก Step 4)
├── logs/                   # CSV log files
└── requirements.txt
```

## วิธีติดตั้ง

```bash
cd /home/rpi5/capstone_design
python3 -m venv .venv
.venv/bin/python -m pip install -r ping_pong_tracker/requirements.txt
```

บน Linux ให้รันโปรแกรมด้วย `.venv/bin/python` เพื่อไม่ติดข้อจำกัด PEP 668 ของ system Python

## วิธีใช้งาน (เรียงตามลำดับ)

### Step 1 — ทดสอบกล้อง
```bash
.venv/bin/python ping_pong_tracker/01_test_camera.py
```
- ถ้ารันจาก desktop ที่มีหน้าต่าง ให้เปิดภาพ preview ด้วย:
	```bash
	CAMERA_TEST_PREVIEW=1 .venv/bin/python ping_pong_tracker/01_test_camera.py
	```
- หน้าต่างจะแสดงภาพ mirror ที่ความละเอียด 1280×800; กด `Q` เพื่อออก
- ถ้ารันผ่าน SSH/TTY ที่ไม่มี `DISPLAY` ให้ใช้ `rpicam-hello -t 0` เพื่อดูภาพผ่าน camera stack แทน หรือเปิดคำสั่งจาก terminal บน desktop ของ Pi
- ถ้าเป็นกล้อง CSI ให้ตรวจพบ sensor ก่อนด้วย `rpicam-hello --list-cameras`
- ถ้าคำสั่งรายงาน `No cameras available!` ต้องตรวจสาย FFC, ไฟเลี้ยง, camera overlay/driver และการเชื่อมต่อก่อน เพราะ OpenCV จะเปิดกล้องไม่ได้
- โปรแกรมจะเก็บข้อมูลประมาณ 60 วินาทีแล้วหยุดเอง หรือกด `Q` เพื่อออกก่อนเวลา
- บันทึก summary ลง `logs/camera_test_<timestamp>.json`
- บันทึกสถิติรายเฟรมลง `logs/camera_test_<timestamp>.csv`
- ตรวจสอบ `shape`, `dtype`, `channels`, `FOURCC`, resolution จริง, FPS, read failures และค่า intensity percentile จากไฟล์ JSON
- ถ้ากล้องไม่ขึ้น ให้เปลี่ยน `CAMERA_INDEX` ใน `01_test_camera.py` ให้ตรงกับ `/dev/videoX`
- ค่า resolution ในไฟล์เป็นค่าที่ร้องขอเท่านั้น ต้องยึด `camera_properties.width/height` และ `first_frame.shape` เป็นค่าจริง

---

### Step 2 — ปรับค่า HSV (deprecated — ไม่ถูกใช้งานจริง)
```bash
python 02_hsv_tuner.py
```
- ปรับ Trackbar จนเห็นเฉพาะลูกปิงปองใน Mask
- กด `S` เพื่อบันทึก → สร้าง `hsv_config.json`
- กด `Q` เพื่อออก
- **หมายเหตุ:** `hsv_config.json` ที่ได้จาก step นี้ไม่ถูกโหลดใช้งานที่ไหนอีกแล้ว (ทั้ง `03_ball_detector.py`, `05_tracker_with_cm.py` และ `test_integrated_system/ball_tracker.py` อ่านค่าจาก `detection_config.json` เท่านั้น) — เก็บไว้เป็นข้อมูลเก่า ให้ปรับค่าจริงที่ `detection_config.json` (`gray_min`/`gray_max`) แทน

---

### Step 3 — ทดสอบ Detection
```bash
.venv/bin/python ping_pong_tracker/03_ball_detector.py
```
- ใช้ grayscale threshold + morphology + contour filtering แทน HSV
- ค่าเริ่มต้นอยู่ใน `detection_config.json` (`gray_min=160`)
- รันผ่าน SSH ได้โดยไม่เปิด GUI และพิมพ์ตำแหน่ง pixel ทาง console
- ถ้ามี desktop ให้เปิด preview ด้วย `BALL_DETECTOR_PREVIEW=1`

### Web GUI — ดูภาพผ่าน SSH จาก PC

รันบน Raspberry Pi:

```bash
cd /home/rpi5/capstone_design/ping_pong_tracker
/home/rpi5/capstone_design/.venv/bin/python web_stream.py --host 0.0.0.0 --port 8080
```

จาก browser บน PC เปิด:

```text
http://192.168.137.166:8080
```

หน้าเว็บจะแสดงภาพ live พร้อมกรอบลูกบอล, ROI, mask, confidence และ FPS

หยุด server ด้วย `Ctrl+C`

---

### Step 4 — Calibration
```bash
python 04_calibration.py
```
1. วางวัตถุอ้างอิงบนพื้น เช่น กระดาษ A4 (21 × 29.7 cm)
2. กด `M` แล้วคลิก 2 จุดที่รู้ระยะจริง
3. กรอกระยะ cm ที่ keyboard
4. กด `S` เพื่อบันทึก → สร้าง `calib_config.json`

---

### Step 5 — Full Tracker
```bash
.venv/bin/python ping_pong_tracker/05_tracker_with_cm.py
```
- ใช้ detector แบบ grayscale เดียวกับ Step 3
- ผ่าน SSH จะทำงานแบบ headless; ใช้ `BALL_TRACKER_PREVIEW=1` เมื่อมี GUI
- แสดงตำแหน่ง X, Y ทั้ง pixel และ cm
- กด `L` เพื่อเริ่ม/หยุดบันทึก CSV
- กด `Q` เพื่อออก

---

## Output ที่ได้

| ค่า | ความหมาย |
|-----|----------|
| `cx_px, cy_px` | ตำแหน่ง pixel จากมุมซ้ายบน |
| `x_tl_cm, y_tl_cm` | ตำแหน่ง cm จากมุมซ้ายบน |
| `x_center_cm, y_center_cm` | ตำแหน่ง cm จากจุดกึ่งกลางภาพ |

> **หมายเหตุ:** `y_center_cm` เป็นบวก = ขึ้น, ลบ = ลง

## Tips

- ⚠️ ทำ Step 2 ในสภาพแสงเดียวกับที่ใช้งานจริง
- ⚠️ กล้องต้องมองตรงลงมาบนพื้น (Top-down) เพื่อความแม่นยำสูงสุด
- ⚠️ ทำ Step 4 ซ้ำทุกครั้งที่ขยับกล้อง