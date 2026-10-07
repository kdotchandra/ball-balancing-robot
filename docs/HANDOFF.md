# Handoff: Ball-and-Plate (3RRS) Capstone

เอกสารสำหรับคนที่รับโปรเจกต์ไปพัฒนาต่อ อ่านไฟล์นี้ก่อน แล้วค่อยเปิด [DEMO_RUN.md](DEMO_RUN.md) ตอนจะรันจริง
สถานะ ณ 2026-09-28: balance, เดินตามเส้นวงกลม และหกเหลี่ยม ทำงานได้บนเครื่องจริง (ค่าที่จูนแล้วอยู่ใน `experiment_profile.PROFILES`)

---

## 1. ระบบทำงานอย่างไร

```
กล้อง OV9281 (30 fps) → BallTracker ตรวจจับบอล (threshold ความสว่าง + Hough circle) → ตำแหน่ง x, y (cm)
→ ความเร็ว (alpha-beta filter) → LQR (+ bias trim + friction compensation + gain scheduling ตอนเดินตามเส้น)
→ มุมเอียงแผ่น θx, θy → slew limit → servo_mapping (วัดจาก calibrate) → ตำแหน่ง servo → HX-35H ×3 ผ่าน /dev/serial0
(IK แบบ linearized และ exact คำนวณทุกรอบเพื่อแสดง/เทียบเท่านั้น ไม่ได้ใช้สั่ง servo)
```

- loop หลัก: `test_integrated_system/main.py` (1 รอบต่อ 1 เฟรมกล้อง, ส่งคำสั่ง servo ทุก ≥ 40 ms)
- ทุกรอบบันทึก log `logs/run_<เวลา UTC>.csv` (ทุกแถว = 1 รอบ loop) + `run_<เวลา>.json` (ค่าที่ใช้, tag, profile, เหตุที่จบ)

## 2. ฮาร์ดแวร์

| ส่วน | รายละเอียด |
|---|---|
| คอมพิวเตอร์ | Raspberry Pi 5, Raspberry Pi OS (Debian 13 trixie), Python 3.13 |
| กล้อง | OV9281 mono (global shutter) ผ่าน picamera2, 1280×800 @ 30 fps, ติดด้านบนมองลงแผ่น สูงจากแผ่นประมาณ 25 cm, ภาพถูก mirror |
| Servo | Hiwonder HX-35H bus servo ×3, UART half-duplex 115200 บน `/dev/serial0` (ttyAMA0) |
| ต่อ Pi กับ servo bus | Pi TX → RX ของ Hiwonder BusLinker v3.0, Pi RX → TX ของ BusLinker (ไขว้กัน) → bus servo; ground ของ Pi และ BusLinker ต่อร่วมกัน (common GND) |
| ไฟเลี้ยง | Servo: AC to DC power supply 12 V 16.7 A (ผ่าน BusLinker); Pi ใช้แหล่งจ่ายแยกต่างหาก |
| กลไก | 3RRS, L1 65 mm, L2 112 mm, จุดยึดบนแผ่นรัศมี 120 mm, ฐาน 180 mm, neutral link-1 25° (`params.py`) |
| แผ่น / บอล | อะคริลิก, ลูกปิงปอง 40 mm / 2.7 g |
| ระยะที่กล้องเห็น | แนวตั้งประมาณ ±8 cm จากกลาง (บอลเริ่มถูกขอบภาพตัดที่ ~8 cm), แนวนอนกว้างกว่า |

Arduino sketch ใน `servo_floor_threshold_reader/` และ `servo_manual_position_test/` ใช้ตอนทดสอบ servo ช่วงแรก

### เช็กลิสต์ก่อนต่อวงจรและเปิดเครื่อง (สำหรับคนที่ยังไม่เคยต่อระบบแบบนี้)

ข้อเหล่านี้ต้องเป็นแบบนี้ ระบบถึงจะทำงาน คนที่เคยทำมาแล้วอาจมองว่าเป็นเรื่องพื้นฐาน แต่ถ้าพลาดข้อใดข้อหนึ่ง อาการที่เห็นมักไม่บอกตรงๆ ว่าผิดที่ตรงไหน

**การต่อสาย**
- [ ] **TX/RX ต้องต่อไขว้**: Pi TX → RX ของ BusLinker, Pi RX → TX ของ BusLinker ถ้าต่อ TX กับ TX ระบบจะไม่ขึ้น error แต่ servo จะไม่ตอบ
- [ ] **GND ร่วม**: ground ของ Pi กับ BusLinker ต้องต่อถึงกัน ถ้าไม่ต่อ สัญญาณ UART ไม่มีจุดอ้างอิงร่วม การสื่อสารจะเพี้ยนหรือไม่ทำงาน
- [ ] **ห้ามต่อสาย bus ของ servo เข้า GPIO ของ Pi ตรงๆ**: servo ใช้สายสัญญาณเส้นเดียวรับส่งสลับกัน (half-duplex) ต้องผ่าน BusLinker เสมอ

**ไฟเลี้ยง**
- [ ] **ไฟ servo 12 V เข้าที่ BusLinker ห้ามดึงไฟจาก Pi**: HX-35H ใช้ไฟ 9–12.6 V ห้ามจ่ายเกิน 12.6 V
- [ ] **แหล่งจ่ายต้องจ่ายกระแสพอ**: servo กินกระแสได้ถึง 3 A ต่อตัวตอนออกแรงเต็มที่ (stall) 3 ตัวเท่ากับ 9 A ตัวที่ใช้อยู่จ่ายได้ 16.7 A (`servo_specifications.txt`)
- [ ] **Pi ใช้แหล่งจ่ายแยก**: ตอน servo ดึงกระแสสูง ไฟจะตก ถ้าใช้ไฟร่วมกัน Pi อาจรีบูตเองหรือ log เสีย (ดูข้อ 10)

**Servo**
- [ ] **ID ต้องไม่ซ้ำกัน และอยู่ถูกตำแหน่ง**: ID 1 ซ้ายบน, ID 2 ขวาบน, ID 3 ด้านล่าง (`servo_calibration/README.md`) servo ใหม่จากโรงงานมี ID 1 ทุกตัว ถ้ามี ID ซ้ำกันบน bus เดียวกัน จะสั่งทีละตัวไม่ได้ ต้องตั้ง ID ทีละตัวโดยต่อ servo ไว้ตัวเดียวตอนตั้ง repo นี้ไม่มีเครื่องมือตั้ง ID ต้องใช้เครื่องมือภายนอก เช่น โปรแกรมของ Hiwonder
- [ ] **เปลี่ยน servo หรือถอดประกอบใหม่ ต้อง calibrate ใหม่**: ทำขั้น 01→06 ตามข้อ 5 แล้วรัน `test_tilt_direction.py` ค่า neutral และทิศทางของแต่ละตัวต่างกัน
- [ ] **พินและข้อต่อต้องแน่นทุกตัว**: ถ้าพินหลวม ระบบยังทำงานได้ แต่ balance แย่ลงชัดเจน และดูเหมือนจูนค่าผิด

**Raspberry Pi**
- [ ] **เปิด serial ให้ถูก**: `raspi-config` → Interface Options → Serial Port: login shell **No**, hardware **Yes** แล้ว reboot และ user ต้องอยู่ในกลุ่ม `dialout` ถ้าเปิด login shell ไว้ Linux จะใช้ serial port เดียวกันนี้ และรบกวนการสื่อสารกับ servo
- [ ] **ใช้พอร์ต `/dev/serial0`**: ตัวอย่างใน `servo_calibration/README.md` เขียน `--port /dev/ttyUSB0` แต่การต่อแบบ TX/RX ของเครื่องนี้ใช้ `/dev/serial0`
- [ ] **กล้อง**: สายแพต้องเสียบแน่น และต้องมี `dtoverlay=ov9281` ใน `/boot/firmware/config.txt` เช็กว่ากล้องขึ้นด้วย `rpicam-hello --list-cameras` กล้องเปิดได้ทีละโปรแกรม ต้องปิด `web_stream.py` ก่อนรัน demo
- [ ] **โปรเจกต์ต้องอยู่ที่ `/home/rpi5/capstone_design`**: path นี้เขียนตายตัวไว้ในสคริปต์ (ดูข้อ 4)
- [ ] **รัน `./setup_pi.sh` ก่อนทุกครั้งที่ตั้งเครื่องใหม่**: สคริปต์เช็กให้อัตโนมัติว่ามี serial, กลุ่ม dialout, กล้อง, แพ็กเกจ และไฟล์ calibrate ครบหรือยัง

**กล้องและแสง**
- [ ] **ความสูงกล้องประมาณ 25 cm**: ถ้าเปลี่ยนความสูง ต้องรัน `04_calibration.py` ใหม่ ไม่อย่างนั้นค่าที่แปลงเป็น cm จะผิด และค่า gain ที่จูนไว้จะใช้ไม่ได้
- [ ] **แสงต้องสม่ำเสมอทั่วแผ่น**: ถ้าบางมุมมืด ตัวตรวจจับจะหาบอลไม่เจอตรงนั้น (ดูข้อ 10)

**ตอนรัน**
- [ ] วางมือไว้ใกล้สวิตช์ไฟ servo ทุกครั้งที่รันกับเครื่องจริง
- [ ] หยุดด้วย **Ctrl+C ครั้งเดียว** ห้ามกด Ctrl+Z เพราะ process จะค้างอยู่และยังถือกล้องกับ servo ไว้

## 3. แผนที่ไฟล์

```
capstone_design/
├── README.md                      หน้าแรกของ repo (ภาษาอังกฤษ)
├── docs/                          HANDOFF.md (ไฟล์นี้), DEMO_RUN.md (วิธีรัน demo)
├── setup_pi.sh                    ตรวจ/ติดตั้งเครื่อง
├── requirements.txt, -dev.txt     แพ็กเกจ Python
├── ping_pong_tracker/             งานกล้องช่วงแรก + config ที่ยังใช้อยู่
│   ├── calib_config.json          pixel_per_cm (ต้องมีก่อนรัน)
│   ├── detection_config.json      ค่าตัวตรวจจับ (threshold, Hough band, min_confidence)
│   ├── 04_calibration.py          สร้าง calib_config.json
│   └── web_stream.py              ดูภาพกล้อง + ผลตรวจจับผ่าน browser (port 8080)
└── test_integrated_system/        ระบบหลัก
    ├── main.py                    loop ควบคุม (ทุก flag ดูได้จาก --help)
    ├── run_real_balance.sh        ตัวเรียกที่ปลอดภัย (ต้อง --confirm + กด Enter) — ใช้ตัวนี้เสมอ
    ├── demo_balance.sh / demo_circle.sh / demo_hexagon.sh   demo (เรียก profile)
    ├── run_experiment.sh          รันการทดลองมาตรฐาน N รอบต่อ tag (บทที่ 4)
    ├── experiment_profile.py      ค่า standard (รายงาน) + PROFILES ที่จูนแล้ว — ที่เดียวที่เก็บค่า
    ├── params.py                  พารามิเตอร์ทางกายภาพ + K_AXIS จาก notebook
    ├── ball_tracker.py            ตัวตรวจจับ + กล้อง
    ├── lqr_controller.py, kinematics.py, actuator.py, trajectory.py
    ├── velocity_filter.py, friction_comp.py, path_tuning.py (gain scheduling, dz-lead), dither.py
    ├── servo_calibration/         ขั้นตอน calibrate servo 01–06 + config/*.json
    ├── state_space_control/       notebooks ที่มาของโมเดลและ LQR (calculated_lqr_ik_3rrs.ipynb)
    ├── run_metrics.py             ให้คะแนนรอบเทียบเป้าหมายรายงาน
    ├── analyze_runs.py            เทียบกลุ่มรอบตาม tag (ใช้ตอนจูน)
    ├── theory_limits.py, stability.py   ขีดจำกัดทางทฤษฎี + ค่าที่วัดได้ (MEASURED)
    ├── presentation_kit.py, presentation_assets.py, ch4_*.py   รูป/วิดีโอ/รายงาน
    ├── tests/                     pytest (ไม่ต้องใช้ฮาร์ดแวร์)
    └── logs/                      log ทุกรอบ (git เก็บเฉพาะ run_*.csv/json)
```

เครื่องมือที่เลิกใช้แล้วแต่ยังเก็บไว้: `calibrate_flat_field.py`, `flat_field_ab_test.py` (แก้แสงไม่สม่ำเสมอ — ปิดอยู่), `capture_raw_frames.py`, `test_detection.py`

## 4. ติดตั้งบนเครื่องใหม่

```bash
cd /home/rpi5/capstone_design
./setup_pi.sh            # ตรวจอย่างเดียว บอกว่าขาดอะไร
./setup_pi.sh --install  # ติดตั้ง apt (python3-picamera2, ffmpeg) + สร้าง .venv + pip install
.venv/bin/python -m pytest test_integrated_system/tests   # ต้องผ่านทั้งหมด
```

- notebooks ใน `state_space_control/` ต้องใช้ `scipy` และ `control` เพิ่ม (ไม่อยู่ใน requirements): `.venv/bin/pip install scipy control`
- picamera2 มาจาก apt ไม่ใช่ pip; `ball_tracker.py` เติม `/usr/lib/python3/dist-packages` ให้เอง
- เปิด serial: `raspi-config` → Interface Options → Serial Port: login shell **No**, hardware **Yes** แล้ว reboot; user ต้องอยู่ในกลุ่ม `dialout`
- path ของโปรเจกต์ถูกเขียนตายตัวใน `run_real_balance.sh` (`/home/rpi5/capstone_design`) ถ้าย้ายที่ต้องแก้บรรทัด `PROJECT_ROOT`

## 5. Calibrate (ทำตามลำดับ เมื่อประกอบใหม่หรือเปลี่ยนชิ้นส่วน)

1. **Servo** — `test_integrated_system/servo_calibration/` ขั้น 01→06 ตาม `servo_calibration/README.md`
   (floor limit → mechanical limits → neutral 25° → mapping/ทิศ → dry-run → real tilt test) ผลอยู่ใน `servo_calibration/config/*.json`
   ⚠ README นั้นเขียน `--port /dev/ttyUSB0` แต่เครื่องนี้ใช้ **`/dev/serial0`**
2. **กล้อง** — `ping_pong_tracker/04_calibration.py` → `calib_config.json` (pixel_per_cm; ตอนนี้ 39.8 px/cm) ต้องทำใหม่ถ้าความสูงกล้องเปลี่ยน
3. **ตัวตรวจจับ** — `ping_pong_tracker/detection_config.json` (`gray_min` 160, Hough radius 32–48 px ที่ภาพย่อ 640 px, `min_confidence` 0.45) เช็กด้วย `web_stream.py` โดยกลิ้งบอลไปตามขอบทั้ง 4 ด้าน ไม่ใช่แค่วางกลางแผ่น
4. ตรวจทิศการเอียง: `test_tilt_direction.py`

## 6. รัน

- Demo: [DEMO_RUN.md](DEMO_RUN.md) (`./demo_balance.sh`, `./demo_circle.sh`, `./demo_hexagon.sh` — ทำงานจนกด Ctrl+C)
- รันเอง: `./run_real_balance.sh --confirm [--profile=NAME] [--path=circle|hexagon] [--duration-s=S] [--tag=NAME] [flag อื่น]`
  - flag ที่ใส่หลัง `--profile` จะ override ค่าเดียวใน profile นั้น
  - `python experiment_profile.py --list` ดู profile ทั้งหมด
- ดูภาพกล้อง: `ping_pong_tracker/web_stream.py --host 0.0.0.0 --port 8080` (ใช้พร้อม demo ไม่ได้ กล้องใช้ได้ทีละโปรแกรม)
- หยุดด้วย Ctrl+C ครั้งเดียว ห้าม Ctrl+Z

## 7. วิธีจูน/ทดลองที่ได้ผล

1. ตั้ง tag ต่อชุดค่า (`--tag=...`) และใส่ `--duration-s=30` (balance) / `50` (path) + `--log-servo-feedback`
2. **รันสลับชุดเป็นรอบ** (A, B, A, B, …) — สภาพแผ่น (ไฟฟ้าสถิต, แรงยึดของบอล) เปลี่ยนตามเวลามากพอจะกลบผลของค่าที่ทดลอง
3. อย่างน้อย ~5 รอบต่อชุด **ต่อรูปทรง** — ค่าที่ดีกับวงกลมไม่จำเป็นต้องดีกับหกเหลี่ยม (dz-lead เป็นตัวอย่าง)
4. วิเคราะห์: `python analyze_runs.py TAG_A TAG_B [--after STAMP]` (error, p95, limit cycle, servo jitter, จุดที่บอลหลุด, permutation test) และ `python run_metrics.py --tag TAG` (เทียบเป้ารายงาน)
5. ตรวจ limit cycle ทุกครั้ง (peak 0.6–0.8 Hz ใน analyze_runs) ก่อนเชื่อค่า error เฉลี่ย
6. ค่าที่ชนะแล้วค่อยใส่เป็น profile ใน `experiment_profile.py` พร้อม comment หลักฐาน แล้วรัน `pytest`

## 8. Flag ทั้งหมด (main.py / run_real_balance.sh)

| Flag | สถานะ | หมายเหตุ / หลักฐาน |
|---|---|---|
| `--profile=NAME` | แนะนำ | ชุดค่าที่จูนแล้ว (`tuned_balance/circle/hexagon`) |
| `--k-scale`, `--kv-scale` | แนะนำ | 0.143 / 1.80 = LQR ออกแบบใหม่จากโมเดลที่วัดได้ |
| `--path-k-full=K1,K2,K3` | แนะนำ (path) | gain scheduling: K ช่วง ramp/path เท่านั้น (1.047,0.571,0.474) |
| `--k-full=K1,K2,K3` | ทดลองได้ | ใส่ K ครบ 3 ตัวแทน k-scale (B-full ใน balance แข็งน้อยกว่า B เพราะ K3 ใหญ่) |
| `--ta=SEC` | แนะนำ | 0.11 s ที่วัดได้ (datasheet 0.181) ใช้ในตัวประมาณมุมแผ่น |
| `--vel-ab=A,B` | แนะนำ | 0.7,0.35 ลด noise ความเร็ว 3 เท่า; **0.5,0.15 หน่วงเกิน → limit cycle 0.6 Hz** |
| `--friction-comp=U,V0` | แนะนำ | 0.6,2 ชดเชย stiction; หกเหลี่ยม −19 % (p 0.03), วงกลมไม่ต่าง |
| `--trim-radius-cm=CM` | แนะนำ (balance) | 2 cm กัน trim wind-up ตอนดึงบอลจากไกล; **ใน path ทำให้แย่ลง ใช้ 10 (ค่าเดิม)** |
| `--dz-lead-deg=DEG` | ระวัง | วงกลม 0.25 ดีขึ้น 20 %; วงกลม 0.70 และหกเหลี่ยม 0.25–0.40 **แกว่งค้างจนบอลหลุด** |
| `--log-servo-feedback` | แนะนำตอนทดลอง | อ่านตำแหน่ง servo กลับมาลง log (`servoN_fb`) คำสั่งช้าลง ≤ 8 ms |
| `--record-video` | ใช้ได้ | อัดวิดีโอพร้อม overlay (`logs/video_*.mp4`) |
| `--trim-ki`, `--path-trim-ki`, `--path-ki` | ค่าเดิม | ตาม standard; `--path-ki` (LQI) เคยทดลองใน ladder ผลคือ 0 |
| `--dither=AMP,HZ` | ไม่แนะนำ | ไม่ช่วยความแม่น, servo ขยับเพิ่ม ~40 % (โครงสร้างสั่น) |
| `--trim-init-deg=X,Y` | ไม่แนะนำ | ช่วยได้เฉพาะเมื่อบอลเริ่มฝั่งเดียวกับ offset |
| `--command-period=0.03` | ไม่ช่วย | ส่งถี่ขึ้นแต่ไม่ได้ทุกเฟรม, ผลไม่ต่าง |
| `--detector=bgsub` | ไม่ได้ทดสอบในรอบจูนนี้ | ต้องเอาบอลออกตอนเริ่มเพื่อถ่าย background; ทุก profile ใช้ `old` |
| flat-field (`BALL_FLAT_FIELD`, `calibrate_flat_field.py`) | ปิดอยู่ | แก้แสงไม่สม่ำเสมอ แต่สร้างบอลปลอมที่มุม; ใช้ได้ถ้าจะทำต่อ |

## 9. ค่าที่วัดได้และข้อค้นพบ (รายละเอียด + แหล่งที่มา: `theory_limits.MEASURED`)

- **Delay** ต่อ loop ~170 ms: กล้อง 11 + ตรวจจับ 14 + รอส่งคำสั่ง 33 + servo ~110 ms; servo กลับทิศเพิ่มอีก ~71 ms (dead band ~3 หน่วย ≈ 0.3°)
- **Plant**: servo time constant 0.11 s; ความเร่งบอลต่อมุม 0.113 m/s²/° (โมเดล 0.103) → โมเดลเชิงเส้นใช้ได้
- **ข้อจำกัดหลักคือ stick-slip**: บอลติดแผ่นจนมุมเกิน 0.5–2° (บางครั้ง 3.5°) แล้วหลุดพุ่ง — ขึ้นกับไฟฟ้าสถิต/ความชื้น/ผิวแผ่น, เปลี่ยนตามเวลา
  - ความแข็งต่อแรงรบกวนคงที่คือ K1/(1+K3) ไม่ใช่ K1 (B ที่ K3 เล็กชนะ B-full ใน balance)
  - error ของการเดินตามเส้น (~1.3–2 cm) เป็นการส่ายจริงของบอล ไม่ใช่ noise การวัด (noise ~1–1.5 mm)
- **noise ความเร็วดิบ** 31 mm/s → มุมแผ่นกระตุก 0.4–1.1° ถ้าไม่กรอง
- **dz-lead** เพิ่ม gain ที่คำสั่งเล็ก (≈ 4·lead/(π·A)) → เกินขอบแล้วแกว่งค้าง 0.6–0.8 Hz
- phase margin ตามโมเดล (Ta 0.11, delay 58 ms, + dead zone): standard 75°/65°, tuned balance 66°/53°, path K 71°/56°
- **ข้อมูลดิบ system identification ไม่อยู่ใน repo**: อยู่ใน microSD card ที่ส่งให้อาจารย์ที่ปรึกษาแล้ว ค่าใน `theory_limits.MEASURED` แต่ละค่ามี comment บอกแหล่งที่มา ถ้าต้องการข้อมูลดิบให้ติดต่ออาจารย์ที่ปรึกษา

## 10. ปัญหาที่รู้แล้ว

| ปัญหา | อาการ | ทำอย่างไร |
|---|---|---|
| พิน/ข้อต่อหลวม | balance แย่ลงชัดทุกค่า, บอลค้างห่างกลาง | เช็กก่อนทุกครั้ง |
| ไฟ Pi อาจตก | log เสีย (null byte / JSON ว่าง) 4 ครั้ง, อาจรีบูต | เช็กแหล่งจ่ายไฟ, แยกไฟ servo |
| แสงไม่สม่ำเสมอ | ห้องผู้พัฒนา: บริเวณซ้าย (x < −4.5 cm) มืด บอลหลุดการตรวจจับ | ห้องแสงสม่ำเสมอไม่เจอ; งานค้างข้อ 1 |
| ขอบกล้อง | บอลเกิน ~8 cm แนวตั้งถูกตัด/หลุด | อย่าขยายเส้นเกิน ~3.5 cm |
| IK สองแบบไม่ตรงกัน | `linearized_ik` ให้มุมห่างจาก exact IK / คำสั่ง servo ~10 เท่า (บรรทัด `servo_err` ในหน้าจอ) — ไม่กระทบการควบคุมเพราะ servo ใช้ `servo_mapping` | ตรวจ `K_ACT` ใน `params.py` เทียบ notebook (notebook 0.093, params 0.055) |
| หกเหลี่ยม margin ต่ำกว่าวงกลม | มุมหกเหลี่ยมกระตุ้นการแกว่ง | ทดสอบแยกรูปทรงเสมอ |

## 11. งานค้าง / แนะนำให้ทำต่อ

1. **Adaptive brightness threshold** ในตัวตรวจจับ (threshold เทียบพื้นรอบบอลแทนค่าคงที่ 145/160) — แก้ปัญหาแสง
2. **ลด stick-slip ที่ต้นเหตุ**: บอลหนักขึ้น (เติมซิลิโคนในลูกปิงปอง), สเปรย์กันไฟฟ้าสถิต, ผิวแผ่นสะอาด — ทดสอบว่ามุม breakaway ลดลงไหม
3. **System identification เต็มรูปแบบ** (step/chirp ของ servo+แผ่น, วัด breakaway ต่อช่วงเวลา) แล้วออกแบบ LQR ที่รวม delay/ความแข็งขั้นต่ำเป็นเงื่อนไข
4. กล้อง fps สูงขึ้น (OV9281 ได้ ~120 fps ที่ความละเอียดต่ำ) — ลด delay 11–33 ms แต่ต้อง calibrate ใหม่
5. replay test ของ `BallTracker` (ต้องเก็บภาพดิบไม่มี overlay ไว้ชุดหนึ่ง)
6. ตรวจ `K_ACT` / linearized IK (ตารางข้อ 10)

## 12. Git

- repo บน GitHub: https://github.com/kdotchandra/ball-balancing-robot (branch `main`); บนเครื่อง Pi อยู่ที่ `/home/rpi5/capstone_design`
- ไม่เก็บ: `.venv/`, ภาพ debug `logs/frames_*`, วิดีโอ, `logs/presentation/` (อยู่บนดิสก์เครื่องนี้เท่านั้น — สำรองแยกถ้าต้องการ)
- clone ลงเครื่องใหม่:
  ```bash
  git clone https://github.com/kdotchandra/ball-balancing-robot.git /home/rpi5/capstone_design
  ```
- ก่อน commit ทุกครั้ง: `.venv/bin/python -m pytest test_integrated_system/tests`
