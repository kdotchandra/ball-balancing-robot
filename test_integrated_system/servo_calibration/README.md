# HX-35H Calibration Workflow

เครื่องมือชุดนี้รันบน Raspberry Pi และคุยกับ HX-35H โดยตรงผ่าน binary UART protocol ไม่ใช้ Arduino gateway

เซอร์โวตามรูป:

- ID 1: ซ้ายบน
- ID 2: ขวาบน
- ID 3: ด้านล่าง

## Hardware safety

ต้องมี bus interface ที่ถูกต้องสำหรับ UART TTL/half-duplex, แหล่งจ่าย servo 9-12.6 V และ common ground กับ Pi ห้ามต่อ bus signal เข้า GPIO โดยตรงจนกว่ายืนยันระดับสัญญาณและวงจร direction control แล้ว

ค่าเริ่มต้นทุกสคริปต์เป็นอ่าน/ถามยืนยัน และสคริปต์ที่ขยับต้องใส่ `--confirm`

## Run order

รันจากโฟลเดอร์นี้:

```bash
cd /home/rpi5/capstone_design/test_integrated_system/servo_calibration
```

### 1. Floor limit

วาง Link 1 ทั้งสามในท่า floor-parallel แล้วอ่าน position:

```bash
/home/rpi5/capstone_design/.venv/bin/python 01_floor_limit.py --port /dev/ttyUSB0
```

บันทึกลง `config/servo_limits.json` ใน `floor_threshold` และแสดง `SAVED` path

### 2. Mechanical soft limits

ทำทีละ servo เริ่มจาก floor threshold ใช้ step เล็กและหยุดก่อนกลไกชน:

```bash
/home/rpi5/capstone_design/.venv/bin/python 02_mechanical_limits.py --port /dev/ttyUSB0 --servo 1 --confirm
/home/rpi5/capstone_design/.venv/bin/python 02_mechanical_limits.py --port /dev/ttyUSB0 --servo 2 --confirm
/home/rpi5/capstone_design/.venv/bin/python 02_mechanical_limits.py --port /dev/ttyUSB0 --servo 3 --confirm
```

ตาม observation ปัจจุบัน script ปฏิเสธ command ที่ `target > floor_threshold` เพราะทิศนั้นทำให้ Link 1 ลงไปชนพื้น ตรวจ behavior จริงทุกครั้งก่อน sweep

กรอก safe min/max หลังหยุด โดยถอยจากจุดชนอย่างน้อย 10-20 counts

### 3. Neutral 25 degrees

```bash
/home/rpi5/capstone_design/.venv/bin/python 03_neutral_25deg.py --port /dev/ttyUSB0 --confirm
```

ปรับจน Link 1 ทำมุม 25 degrees กับพื้นและ acrylic plate level จากนั้นกรอกตำแหน่งสุดท้ายและ H0 ตาม datum ที่วัดจริง บันทึกลง `config/servo_geometry.json`

### 4. Servo direction/mapping

```bash
/home/rpi5/capstone_design/.venv/bin/python 04_servo_mapping.py --port /dev/ttyUSB0 --confirm
```

สคริปต์ขยับทีละ ID ไป neutral +/- step และให้ผู้ใช้บันทึกผล tilt จากนั้นกรอก position sign และ coefficients ของ tilt_x/tilt_y บันทึกลง `config/servo_mapping.json`

### 5. Dry-run feedback

ต้องมี limits, neutral และ mapping ครบก่อน:

```bash
/home/rpi5/capstone_design/.venv/bin/python 05_dry_run_feedback.py --tilt-x-deg 0.5 --tilt-y-deg -0.5 --confidence 0.9
```

ไม่มีคำสั่ง serial ถูกส่ง

### 6. Real tilt test

เริ่มที่ tilt limit ต่ำมากและทำ emergency power cutoff ให้พร้อม:

```bash
/home/rpi5/capstone_design/.venv/bin/python 06_real_tilt_test.py \
  --port /dev/ttyUSB0 \
  --tilt-x-deg 0.5 --tilt-y-deg 0 \
  --tilt-limit-deg 0.5 \
  --confidence 0.9 --confirm --enable
```

สคริปต์จะถามยืนยันอีกครั้งก่อนส่ง, clamp ตาม mechanical limits และอ่าน feedback กลับหลังสั่ง

## Config files

- `config/servo_limits.json`: floor threshold และ mechanical soft limits
- `config/servo_geometry.json`: L1/L2, acrylic 6 mm, offset 28 mm, neutral 25 degrees, H0
- `config/servo_mapping.json`: ตัวคูณ tilt → มุม servo จาก IK ที่ linearize รอบท่า neutral (ขนาดของตัวคูณกรอกตามโมเดล ไม่ได้ fit จากการวัด; ขั้น 04 ตรวจทิศทางของแต่ละ servo บนหุ่นจริง)

หลังขั้น 1-3 เสร็จ `test_integrated_system/main.py` จะโหลด neutral และ mechanical limits จาก config เหล่านี้โดยตรง ไม่ใช้ค่า default ใน `params.py`

ห้ามแก้ status เป็น measured เอง และห้ามข้ามลำดับ เพราะขั้นถัดไปโหลดค่าจาก config ของขั้นก่อนหน้า
