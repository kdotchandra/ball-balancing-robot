# Test Integrated System — LQR + IK + OpenCV บนเครื่องจริง

ระบบควบคุม ball-and-plate แบบ 3RRS: ตรวจจับลูกปิงปองจากกล้อง → LQR → servo HX-35H จริง

> **เริ่มที่ [../docs/HANDOFF.md](../docs/HANDOFF.md)** (ภาพรวม, ติดตั้ง, calibrate, flag ทั้งหมด, ข้อค้นพบ, ปัญหาที่รู้แล้ว)
> และ **[../docs/DEMO_RUN.md](../docs/DEMO_RUN.md)** (วิธีรัน demo) — README นี้เก็บรายละเอียดเดิมของการรัน `main.py` และการทดลองบทที่ 4

สถานะ (2026-09-28): สั่ง servo จริงผ่าน `run_real_balance.sh --confirm` (ต้องกด Enter ยืนยัน); ถ้ารัน `main.py` ตรงๆ
โดยไม่ตั้ง `SERVO_OUTPUT=1` จะเป็น dry-run (คำนวณและพิมพ์ แต่ไม่ส่งคำสั่ง)

## โครงสร้างไฟล์หลัก

```text
test_integrated_system/
├── main.py                 # loop ควบคุม real-time (flag ทั้งหมด: main.py --help)
├── run_real_balance.sh     # ตัวเรียกที่ปลอดภัย ใช้ตัวนี้สั่ง servo จริง (--profile=NAME ได้)
├── demo_*.sh               # demo balance / circle / hexagon (เรียก profile ที่จูนแล้ว)
├── run_experiment.sh       # การทดลองมาตรฐานบทที่ 4 (N รอบต่อ tag)
├── experiment_profile.py   # ค่า standard + PROFILES ที่จูนแล้ว (ที่เดียวที่เก็บค่า)
├── params.py               # พารามิเตอร์และ gain จาก state_space_control/calculated_lqr_ik_3rrs.ipynb
├── ball_tracker.py         # ตัวตรวจจับ + กล้อง (อ่าน config จาก ping_pong_tracker)
├── lqr_controller.py       # LQR controller
├── kinematics.py           # Linearized IK + Exact 2-link Leg IK
├── actuator.py             # ส่งคำสั่ง/อ่านตำแหน่ง servo (bus HX-35H)
├── velocity_filter.py, friction_comp.py, path_tuning.py, dither.py   # ส่วนเสริมของตัวควบคุม
├── run_metrics.py, analyze_runs.py, theory_limits.py                 # ให้คะแนน / วิเคราะห์ / ขีดจำกัด
├── servo_calibration/      # calibrate servo 01–06
├── tests/                  # pytest (ไม่ใช้ฮาร์ดแวร์)
└── logs/                   # log ทุกรอบ
```

## Dependency

ติดตั้งทั้งโปรเจกต์จาก root (ดู docs/HANDOFF.md ข้อ 4):

```bash
cd /home/rpi5/capstone_design && ./setup_pi.sh --install
```

ถ้ายังไม่ได้สร้างไฟล์ Calibration ให้ทำในโฟลเดอร์ ping_pong_tracker ก่อน

- รัน 04_calibration.py เพื่อสร้าง calib_config.json
- ระบบนี้จะอ่าน detection_config.json และ calib_config.json จากโฟลเดอร์ ping_pong_tracker โดยอัตโนมัติ

## วิธีรัน

จากโฟลเดอร์ test_integrated_system

```bash
/home/rpi5/capstone_design/.venv/bin/python main.py --headless
```

หมายเหตุ: ค่า default ใช้ camera index 0, Picamera2/libcamera และ grayscale detector จาก `ping_pong_tracker/detection_config.json` โดยเปิด mirror ภาพไว้

ต้องสร้าง `ping_pong_tracker/calib_config.json` ก่อนรัน integrated system ห้ามใช้ค่า pixel scale เดา เพราะจะทำให้ feedback ผิดหน่วย

`--headless` ใช้สำหรับ SSH; ถ้ามี desktop ให้ละ option นี้เพื่อเปิด preview

Actuator เริ่มต้นเป็น dry-run และมี confidence gate, position limits และ maximum step ต่อรอบ

การเปิด output ไปยัง gateway ทำได้เฉพาะหลังตรวจ neutral/limits/sign แล้ว:

```bash
SERVO_OUTPUT=1 SERVO_PORT=/dev/serial0 \
/home/rpi5/capstone_design/.venv/bin/python main.py --headless
```

ถ้าต้องการเลือกกล้อง

```bash
python main.py --camera 1
```

ถ้าต้องการปิด mirror

```bash
python main.py --camera 1 --no-mirror
```

ถ้าต้องการดูรายการกล้องที่เปิดได้ก่อน

```bash
python main.py --list-cameras
```

ปุ่มควบคุม

- Q: ออกจากโปรแกรม
- M: สลับโหมดระหว่าง Balance และ Path Tracking

## โหมดการทำงาน

### โหมด 1: Balance Control

Reference ถูกตั้งไว้ที่จุดศูนย์กลางเสมอ

- x_ref = 0
- y_ref = 0

สิ่งที่พิมพ์ออกมาในแต่ละเฟรม

- ตำแหน่งลูกบอล x, y (m)
- มุมเอียงแท่นที่ LQR สั่ง tx, ty (deg)
- มุม Servo แบบ Linearized q_lin ของทั้ง 3 ขา (deg)
- มุม q1 จาก Exact Leg IK ของทั้ง 3 ขา (deg)
- Observation สั้นๆ ว่าทิศทางควบคุมสมเหตุสมผลหรือไม่

ความคาดหวังเบื้องต้น

- ลูกอยู่ด้านขวา x > 0 ควรได้คำสั่งเอียงให้ขวาต่ำเพื่อดึงกลับ
- q_lin ควรเปลี่ยนทิศทางสอดคล้องกับมุมเอียง
- q1_exact ควรใกล้ q_lin เมื่อมุมเอียงยังเล็ก

### โหมด 2: Path Tracking

Reference เป็นเส้นวงรีตามเวลา

- x_ref = 0.03 cos(wt)
- y_ref = 0.02 sin(wt)

พิมพ์ค่าเหมือนโหมด Balance และเพิ่ม

- x_ref, y_ref
- tracking error

ความคาดหวังเบื้องต้น

- มุมสั่งของ Servo ควรเปลี่ยนตามทิศทาง reference
- tracking error ควรลดลงเมื่อบอลอยู่ใกล้เส้นทาง

## การทดสอบประเมินผลด้วยค่ามาตรฐาน (บทที่ 4 ของรายงาน)

ทุกการทดสอบใช้ค่าชุดเดียวกันจาก `experiment_profile.py` (เกนชุดทรงตัว k-scale 0.076, kv-scale 2.8, trim-ki 0.125,
tilt limit 8°, ส่งคำสั่งเซอร์โวทุก 0.04 s / 50 ms) ทั้ง `main.py` และ `run_real_balance.sh` อ่านค่าจากไฟล์นี้ที่เดียว
ถ้าใส่ flag ทับ launcher จะแสดง `Overrides:` และบันทึกไว้ในไฟล์ `.json` ข้างไฟล์ log ของรอบนั้น

```bash
./run_experiment.sh balance      # ทรงตัว 10 ครั้ง วางลูกบอลอิสระ 30 วินาที/ครั้ง
./run_experiment.sh circle       # วงกลมรัศมี 3 cm 10 ครั้ง 60 วินาที/ครั้ง
./run_experiment.sh hexagon      # หกเหลี่ยมที่ครอบวงกลมรัศมี 3 cm 10 ครั้ง
./run_experiment.sh circle --dry-run   # แสดงคำสั่งที่จะรัน ไม่รันจริง
```

- ทุกรอบยังผ่าน `run_real_balance.sh --confirm` และต้องกด Enter ทุกครั้ง (ปิด stdin = ยกเลิก ไม่ขยับ servo)
- นับผลจากป้ายกำกับใน `.json` (`ended_reason`): `completed` และ `ball_lost` นับเป็นผล, `never_tracked` (ไม่เห็นลูกบอลเลย)
  และ `interrupted` (กด Ctrl+C) เก็บ log ไว้แต่ไม่นับ สคริปต์รันต่อจากรอบที่ค้างไว้ได้
- ตัวอินทิเกรตช่วงเส้นทาง (`--path-ki`, LQI) ปิดไว้ (0) จนกว่าจะเลือกค่าจากการทดลอง:
  `python run_metrics.py --ladder ladder_0.05 ladder_0.10 ladder_0.20` ใช้เกณฑ์ที่กำหนดไว้ล่วงหน้าเลือกค่า
- สรุปผลกลุ่มหนึ่ง: `python run_metrics.py --tag circle` (มีอัตราความสำเร็จพร้อมช่วงเชื่อมั่น 95%)
- วัดเวลาหน่วงของลูปด้วย `test_freq_response.py`

## Pipeline ที่ระบบทำในแต่ละเฟรม

1. อ่านภาพจากกล้อง
2. ตรวจจับตำแหน่งลูกบอลด้วย OpenCV
3. แปลงพิกัดจากพิกเซลเป็นเมตร (อ้างอิงจุดกึ่งกลางภาพ)
4. ประมาณความเร็ว x_dot, y_dot ด้วย finite difference
5. คำนวณ LQR เพื่อหามุมเอียงแท่นสั่ง
6. แปลงผ่าน Linearized IK เพื่อหา z_ref และ q_lin
7. แปลงผ่าน Exact Leg IK เพื่อหา q1, q2 ของแต่ละขา
8. พิมพ์ผลแบบ Real-time

## หมายเหตุสำคัญ

- โหมดนี้เป็นการทดสอบความสมเหตุสมผลของ Pipeline เท่านั้น ยังไม่มีการส่งคำสั่งไปฮาร์ดแวร์
- ถ้าไม่ detect ลูกบอลได้ ระบบจะ hold ค่าตำแหน่งล่าสุดไว้ชั่วคราวและ actuator จะไม่รับ confidence ต่ำ
- มีการจำกัดความเร็วเชิงตัวเลขเพื่อกันสัญญาณกระชากจาก noise
- ถ้าทิศทางเอียงดูผิด ให้ตรวจระบบแกนกล้องและการตั้งแกนของแพลตฟอร์มก่อน

## Troubleshooting

1. ยังไม่มี calibration

- รัน `04_calibration.py` ใน `ping_pong_tracker` ก่อน
- ตรวจว่ามี `ping_pong_tracker/calib_config.json`

2. เปิดกล้องไม่ได้

- ลองเปลี่ยน --camera เป็น 1 หรือ 2
- ปิดโปรแกรมอื่นที่ล็อกกล้องอยู่
- ถ้า OBS Virtual Camera โผล่ในรายการ ให้เลือก index ของ webcam จริง หรือใช้ `--camera N` แบบระบุชัดเจน

3. Detect ลูกบอลไม่เจอ / ไม่เสถียร

- กล้องจะล็อก auto-exposure/auto-white-balance เองหลังเปิดกล้องไม่กี่วินาที (ดู log `[Camera] Locked AE/AWB: ...`) เพื่อลดปัญหาความสว่างเปลี่ยนตามแสง — ถ้าแสงในห้องเปลี่ยนมากระหว่างใช้งาน ให้รีสตาร์ทโปรแกรมใหม่เพื่อให้ล็อกค่าใหม่
- ปรับค่า `gray_min`/`gray_max` (ไม่ใช่ HSV — ระบบเปลี่ยนมาใช้ grayscale threshold แล้ว) ใน `ping_pong_tracker/detection_config.json`
- ตรวจแสงให้ใกล้เคียงตอนใช้งานจริง

4. ค่าพิกัด cm หรือ m แปลก

- ทำ Calibration ใหม่ใน ping_pong_tracker
- ตรวจว่ากล้องอยู่มุม top-down มากที่สุด

5. ค่า command แกว่งมาก

- เกิดจากตำแหน่งกระโดด ให้ตรวจ mask และการสะท้อนแสง
- ลด noise ที่ภาพก่อนตรวจจับหรือเพิ่ม filtering ในอนาคต
