# ping_pong_tracker path: E:\New folder\ENG51_4000_CAPSTONE DESIGN\ping_pong_tracker

# calculated_lqr_ik_3rrs.ipynb path: E:\New folder\ENG51_4000_CAPSTONE DESIGN\state_space_control\calculated_lqr_ik_3rrs.ipynb

เขียนโค้ด Python ที่นำผลการคำนวณจากไฟล์ calculated_lqr_ik_3rrs.ipynb
calculated_lqr_ik_3rrs.ipynb
มาใช้งานร่วมกับ OpenCV เพื่อทดสอบว่า Pipeline ทั้งหมดทำงานสมเหตุสมผลหรือไม่

สิ่งที่มีอยู่แล้วคือโค้ด OpenCV ที่ตรวจจับตำแหน่งลูกปิงปอง (x, y) จากกล้องได้แล้ว และไฟล์ calculated_lqr_ik_3rrs.ipynb
calculated_lqr_ik_3rrs.ipynb
ที่มีพารามิเตอร์และ Pipeline ครบดังนี้ ลูกบอลมวล 2.7g เส้นผ่านศูนย์กลาง 40mm แท่น 300mm ขา L1=100mm L2=148.2mm Rolling factor kr=0.6 Servo time constant Ta=0.181437s Damping b=0.003 N·s/m Jacobian matrix J และ Jinv สำหรับ control-level kinematics LQR gain matrix K_axis = [5.235988, 1.853465, 1.529578] Actuator gain k_act=0.093232 m/rad Exact Leg IK แบบ 2-link สำหรับคำนวณมุมขาจริง

Pipeline ที่ต้องการคือ รับค่าตำแหน่ง (x, y) จากกล้อง Real-time แล้วประมาณค่าความเร็ว (x_dot, y_dot) จากตำแหน่งที่ได้จากกล้อง จากนั้นคำนวณผ่าน LQR Controller เพื่อได้มุมเอียง Platform (θx, θy) แล้วแปลงผ่าน Linearized IK เพื่อได้ความสูงขา z_ref และมุม Servo แบบ Linearized แล้วแปลงผ่าน Exact Leg IK เพื่อได้มุม q1, q2 ของแต่ละขาจริง สุดท้าย Print ค่าทั้งหมดออกมาแบบ Real-time

เงื่อนไขการทดสอบในขั้นตอนนี้คือลูกบอลวางอยู่บนพื้นปกติยังไม่ได้อยู่บน Platform จริง ยังไม่ขยับ Servo จริง แค่ Print ค่าออกมาดูเท่านั้น

สิ่งที่ต้องการ Print ออกมาคือสองโหมด

โหมดที่ 1 Balance Control ให้ Print ค่าต่อไปนี้ออกมาพร้อมกัน ได้แก่ ตำแหน่งลูกบอลจากกล้อง (x, y) มุมเอียง Platform ที่ LQR สั่ง (θx, θy) เป็น degree มุม Servo แบบ Linearized ของแต่ละขา (q1_lin, q2_lin, q3_lin) เป็น degree มุม q1 ของแต่ละขาจาก Exact Leg IK (q1_exact_leg1, q1_exact_leg2, q1_exact_leg3) เป็น degree ข้อสังเกตว่าค่าใดสมเหตุสมผล โดยเมื่อลูกบอลอยู่ทางขวาของศูนย์กลาง Platform ควรเอียงให้ขวาต่ำเพื่อดึงลูกกลับมา มุม Servo แต่ละขาควรเปลี่ยนไปสอดคล้องกับทิศทางการเอียง และมุมจาก Exact IK ควรใกล้เคียงกับมุม Linearized เมื่อมุมเอียงยังน้อยอยู่

โหมดที่ 2 Path Tracking ให้ Print ค่าเดียวกันกับโหมดที่ 1 แต่เพิ่ม Reference Position (x_ref, y_ref) และ Tracking Error ออกมาด้วย ข้อสังเกตว่าค่าใดสมเหตุสมผล โดยมุม Servo ควรเปลี่ยนทิศทางไปตามตำแหน่งของ Reference Path และ Tracking Error ควรลดลงเมื่อลูกบอลอยู่ใกล้เส้นทาง

สิ่งที่ต้องการจากโค้ดคือ ดึงค่าพารามิเตอร์และ LQR Gain จาก Notebook มาใช้ตรงๆ ไม่ต้องคำนวณใหม่ รับค่า (x, y) จากกล้อง Real-time ประมาณความเร็วด้วย Finite Difference จากตำแหน่งก่อนหน้า คำนวณ LQR แล้วส่งผ่าน Linearized IK และ Exact Leg IK ครบทั้งสองแบบ Print ค่าทั้งหมดออกมาในรูปแบบที่อ่านง่ายแบบ Real-time โค้ดแบบ Modular ที่ต่อกับ Servo จริงได้ในภายหลัง และอธิบาย Model ของ Inverse Kinematics และ State Space ที่ใช้ด้วย
