# Computer Vision Engineer Intern Assignment - Answers

## สรุปการทดลองที่ใช้ตอบคำถาม

ผม fine-tune YOLO26s สำหรับงาน single-class object detection (`car_front`) โดยจัด dataset ให้อยู่ในรูปแบบที่ Ultralytics รองรับ แล้วแบ่ง train/validation ด้วย `prepare_dataset.py`

Environment ที่ใช้จริง:

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, VRAM 4GB
- PyTorch: `2.11.0+cu128`
- Ultralytics: `8.4.51`
- Final training: `epochs=80`, `imgsz=640`, `batch=4`, `device=0`
- Training time: ประมาณ 0.791 ชั่วโมง

Final validation metrics จาก `best.pt`:

- Precision: 0.986
- Recall: 0.978
- mAP50: 0.994
- mAP50-95: 0.867
- Inference speed บน validation: ประมาณ 6.4 ms/image

ผมใช้ผล final run นี้ในการตอบคำถามด้านล่าง ไม่ใช่ smoke run รอบแรก

## 1) การแบ่ง Train/Validation

ผมเลือก split แบบ 80/20:

- Train: 640 images
- Validation: 160 images
- ทั้ง dataset มี 800 images, positive labels 726 images และ empty/background labels 74 images
- ใน split ที่ได้ train มี background 59 images และ validation มี background 15 images

เหตุผลที่เลือก 80/20 คือ dataset มีขนาดไม่ใหญ่มากสำหรับ object detection ถ้าแบ่ง validation ใหญ่เกินไป โมเดลจะเสีย training examples ที่จำเป็นต่อการเรียนรู้ variation ของหน้ารถ เช่น sedan, MPV, มุมกล้อง, สีรถ, scale และ lighting แต่ถ้า validation เล็กเกินไป metric จะไม่น่าเชื่อถือและ failure analysis จะเห็นเคสน้อยเกินไป ดังนั้น 20% หรือ 160 รูปเป็นจุดสมดุลที่ยังให้ validation ใหญ่พอสำหรับดู precision/recall, mAP และตัวอย่าง false positive/false negative

ผมตั้ง seed คงที่ (`seed=42`) เพราะต้องการให้ผล reproduce ได้ หากผู้ประเมินหรือผมรันซ้ำจะได้ split เดิม ทำให้เปรียบเทียบ experiment ได้ยุติธรรมกว่า

อีกเหตุผลที่ผมไม่ตัด background/empty labels ทิ้ง คือในงาน production false positive มีต้นทุนจริง ถ้าโมเดลเห็นแต่ภาพที่มีรถ อาจเรียนรู้ว่าแทบทุกภาพต้องมี `car_front` แล้วทำนายผิดบนฉากที่ไม่มีรถได้ง่าย การเก็บ negative samples ไว้ทั้ง train และ validation จึงช่วยวัดและลด false positive

เรื่อง data leakage: ถ้า dataset มีรถคันเดียวกันหลายเฟรม เช่น frame ต่อเนื่องจากวิดีโอ ผมจะไม่สุ่มแยกรายภาพแบบ naive เพราะเฟรมใกล้กันมักมี background, object size, lighting และตำแหน่งรถคล้ายกันมาก ถ้าเฟรมหนึ่งอยู่ train และเฟรมข้าง ๆ อยู่ validation ค่า validation จะดูดีเกินจริง แต่ไม่ได้สะท้อน generalization

แนวทางที่ใช้คือ group split: ภาพที่น่าจะมาจาก source เดียวกันต้องอยู่ฝั่งเดียวกันทั้งหมด ไม่ว่าจะ train หรือ validation ใน `prepare_dataset.py` ผม group จาก filename pattern เช่น suffix ของ Roboflow, ชื่อ `train_car_XXXX`, และกลุ่ม `mpv` เพื่อลดโอกาส leakage เท่าที่ทำได้จากข้อมูล filename ที่มี

## 2) การเลือก Augmentation

หลักคิดของผมคือ augmentation ต้องช่วยให้โมเดลทนต่อความหลากหลายที่เกิดจริง แต่ไม่ควรสร้างภาพที่ผิดธรรมชาติสำหรับงาน frontal car detection เพราะ target คือ "พื้นที่ด้านหน้ารถ" ซึ่งมี geometry สำคัญ เช่น ไฟหน้า กระจังหน้า กันชน และแนว symmetry ของรถ

ค่าที่ใช้จริงใน `train.py`:

- `hsv_h=0.015`, `hsv_s=0.5`, `hsv_v=0.35`: เปิดระดับ moderate เพราะสีรถและสภาพแสงเปลี่ยนได้มากในงาน DOOH เช่น แสงแดด เงา กลางคืน หรือกล้องคนละตัว แต่ไม่เพิ่มแรงเกินไปเพราะสี/contrast ที่ผิดธรรมชาติอาจทำให้โมเดลเรียนรู้ feature ที่ไม่ตรงกับภาพจริง
- `degrees=3.0`: เปิดน้อย เพราะหน้ารถใน production ส่วนใหญ่ไม่ควรหมุนเอียงมาก การหมุนแรงจะสร้างตัวอย่างที่ไม่ตรง distribution จริง และอาจทำให้ box localization แย่ลง
- `translate=0.08`: เปิดเล็กน้อย เพราะรถในภาพจริงอาจอยู่ไม่กลางเฟรม หรือถูก crop บางส่วน การ translate ช่วยให้โมเดลไม่ผูกกับตำแหน่งเดิมมากเกินไป
- `scale=0.4`: เปิด เพราะ scale เป็น variation สำคัญ รถอยู่ใกล้/ไกลกล้องได้ และในวิดีโอแนวตั้งขนาด object อาจต่างจากภาพ train
- `shear=0.0`: ปิด เพราะ shear บิดรูปทรงหน้ารถให้ผิด geometry จริง สำหรับงานที่สนใจ frontal area การบิดแบบนี้อาจทำร้าย localization มากกว่าช่วย generalization
- `perspective=0.0005`: เปิดน้อยมาก เพราะมุมกล้องจริงอาจมี perspective เล็กน้อย แต่ถ้าแรงเกินไปจะทำให้ frontal car กลายเป็นมุมแปลกที่ไม่ใช่ target หลัก
- `flipud=0.0`: ปิด เพราะรถกลับหัวไม่ใช่เหตุการณ์จริงใน production ปกติ ถ้าเปิดจะเพิ่ม noise ให้ training
- `fliplr=0.5`: เปิด เพราะการกลับซ้ายขวายังสมเหตุสมผลสำหรับหน้ารถ ช่วยให้โมเดลไม่ bias กับทิศแสงหรือตำแหน่งเลนด้านใดด้านหนึ่ง
- `mosaic=0.6`, `close_mosaic=10`: เปิดแต่ไม่สูงสุด เพราะ mosaic ช่วย dataset ขนาดเล็กโดยเพิ่ม context และ scale variation แต่ถ้าใช้แรงเกินไปภาพจะไม่เหมือน inference จริง ผมจึงใช้ `close_mosaic=10` เพื่อปิด mosaic ช่วงท้าย ให้โมเดล fine-tune กับภาพธรรมชาติก่อนจบ training
- `mixup=0.0`: ปิด เพราะการผสมภาพสองภาพอาจทำให้หน้ารถโปร่ง/ซ้อนกัน ซึ่งไม่ใช่ visual pattern ปกติของ object นี้
- `copy_paste=0.0`: ปิด เพราะการ copy หน้ารถไปวางบน background ใหม่แบบไม่ระวังจะสร้าง shadow, scale และ boundary ที่ไม่สมจริง
- `erasing=0.2`: เปิดเบา ๆ เพราะในภาพจริงอาจมี occlusion เงา ป้าย หรือวัตถุบังบางส่วน แต่ใช้ไม่แรงเพื่อไม่ลบ feature สำคัญอย่างไฟหน้า/กระจังหน้ามากเกินไป
- `auto_augment='randaugment'`: เปิดเพื่อเพิ่ม robustness ตาม config ของ Ultralytics แต่ถ้า full training พบว่า localization แย่หรือ validation fluctuation สูง ผมจะทำ ablation โดยลด/ปิด auto augment

Trade-off หลักคือ augmentation มากช่วยลด overfitting แต่ถ้ามากเกินไปจะทำให้ distribution ของ train ห่างจาก validation/video จริง โดยเฉพาะงานนี้ที่ box boundary สำคัญ ผมจึงเลือก augmentation แบบ conservative มากกว่าสร้างภาพหลากหลายสุดโต่ง

## 3) Failure Analysis

หลัง train 80 epochs แล้ว validation metrics ดีขึ้นมาก:

- Precision: 0.986
- Recall: 0.978
- mAP50: 0.994
- mAP50-95: 0.867

ผมไม่ได้ดูแค่ mAP เพราะ mAP บอกภาพรวม แต่ไม่บอกว่า failure มาจากอะไร จึงรัน prediction บน validation set ด้วย `conf=0.25` แล้วเลือกตัวอย่างที่แย่จาก false positive, false negative, IoU ต่ำ และ duplicate boxes เก็บไว้ใน `results/failure_examples/`

ผลหลัง final run ไม่พบ false negative หรือ IoU ต่ำรุนแรงใน top failure cases แล้ว ปัญหาหลักที่เหลือคือ duplicate boxes หรือ extra prediction บนภาพที่มี GT เพียง 1 กล่อง:

- `failure_1_train_car (1584).jpg`: best IoU 0.850, มี GT 1 กล่อง แต่โมเดล predict 2 กล่อง จึงเป็น localization ที่จับวัตถุหลักได้ดีแต่มี extra box เพิ่ม
- `failure_2_1b40031a-train_car_1649.jpg`: best IoU 0.862, มี duplicate prediction เช่นกัน สะท้อนว่าโมเดลมั่นใจในหลายส่วนของ frontal area
- `failure_3_train_car (1709).jpg`: best IoU 0.879, ปัญหาไม่ใช่ missed detection แต่เป็น duplicate/FP เพิ่ม
- `failure_4_train_car (2378).jpg`: best IoU 0.906, prediction หลักดี แต่ยังมี extra box
- `failure_5_train_car (2429).jpg`: best IoU 0.917, prediction หลักดี แต่ยังมี extra box

วิธีคิดจาก failure รอบสุดท้ายคือโมเดลเรียนรู้ class ได้ดีแล้ว จึงไม่ควรแก้ด้วยการเพิ่ม augmentation หนัก ๆ หรือเปลี่ยน architecture ทันที ปัญหาที่เหลืออยู่ใกล้ post-processing และ confidence calibration มากกว่า เช่น ปรับ confidence threshold, NMS IoU, minimum box size หรือใช้ temporal tracking ในวิดีโอเพื่อรวม box ที่ซ้ำกัน

ถ้ามีเวลาเพิ่ม ผมจะแก้ตามชนิด failure:

- Duplicate boxes: ทำ threshold sweep, ปรับ NMS IoU และดูว่ากล่องซ้ำเกิดจากส่วนไหนของรถ
- False positive บน video: เพิ่ม hard negatives จาก frame จริงที่โมเดล trigger ผิด
- Label ambiguity: audit guideline ว่าหน้ารถควรครอบถึงส่วนใด เพื่อให้โมเดลเรียน boundary ชัดขึ้น
- Production video: ใช้ temporal consistency เพราะ detection บนวิดีโอไม่ควรตัดสินจาก frame เดี่ยว

## 4) Confidence Threshold สำหรับ Inference

ผมรัน inference บน `Video.mp4` ซึ่งมี 346 frames, 30 FPS, resolution 1080x1920 และเปรียบเทียบ threshold สองค่าโดยใช้ final `best.pt`:

- `conf=0.25`: detect 113/346 frames, รวม 119 boxes, max confidence 0.907, average confidence 0.590
- `conf=0.5`: detect 80/346 frames, รวม 82 boxes, max confidence 0.907, average confidence 0.697

ผมเลือก `conf=0.25` สำหรับ `output_video.mp4` เพราะโจทย์ต้องการ output วิดีโอที่วาด bounding box และในบริบท detection เพื่อส่งต่อ downstream การพลาดรถมีต้นทุนสูงกว่าในช่วงแรก ถ้าใช้ 0.5 จำนวน detected frames ลดจาก 113 เหลือ 80 เฟรม แปลว่า recall บนวิดีโอลดลงชัดเจน แม้ precision น่าจะดีขึ้น

เหตุผลเชิง trade-off:

- Threshold ต่ำ เช่น 0.25 เพิ่ม recall เหมาะเมื่อการพลาดรถมีต้นทุนสูง แต่ต้องยอมรับ false positive และ duplicate boxes มากขึ้น
- Threshold สูง เช่น 0.5 เพิ่ม precision และ average confidence ของ box สูงขึ้น เหมาะเมื่อ false positive มีต้นทุนสูง แต่เสี่ยงไม่ detect รถหลายเฟรม
- สำหรับวิดีโอ การตัดสินใจไม่ควรพึ่ง frame เดียวเท่านั้น เพราะรถควรปรากฏต่อเนื่องหลายเฟรม จึงสามารถใช้ temporal consistency ช่วย เช่น detect ต่อเนื่อง N เฟรม หรือ track box ก่อน trigger downstream

ถ้าโมเดลนี้ใช้ trigger License Plate Recognition (LPR) ผมจะเลือก threshold ค่อนข้างต่ำถึงกลาง เช่น 0.25-0.35 ในช่วงแรก เพราะ false negative หมายถึงไม่ส่งภาพเข้า LPR เลย ทำให้เสียโอกาสอ่านป้ายทันที ขณะที่ false positive ยังอาจถูกกรองต่อด้วย LPR confidence, tracking, ROI, minimum box size หรือ rule ว่าต้องเห็นรถต่อเนื่องหลายเฟรม

แต่ถ้าระบบ downstream มีต้นทุนสูงมาก เช่น ทุก trigger ต้องเรียก API ราคาแพง หรือทำให้ผู้ใช้เห็นผลลัพธ์ผิด ผมจะขยับ threshold สูงขึ้นหลัง calibrate จาก validation video ที่ใกล้ production จริง สรุปคือ threshold ที่ดีไม่ได้เป็นค่าคงที่จาก mAP อย่างเดียว แต่ต้องเลือกตาม cost ของ false positive และ false negative ในระบบจริง

## 5) ถ้ามีเวลาเพิ่ม 1 เดือน และข้อมูลเพิ่ม 10 เท่า

ผมจะไม่เริ่มจากการเปลี่ยนโมเดลใหญ่ขึ้นทันที แต่จะเริ่มจาก data quality และ evaluation design ก่อน เพราะงานนี้เป็น single-class detector ที่ performance มักถูกจำกัดด้วย distribution gap และ label consistency มากกว่าความซับซ้อนของ architecture

ข้อมูลที่อยากเพิ่ม:

- ภาพจากกล้อง production จริงหลาย location เพื่อให้ครอบคลุม domain จริงของ DOOH
- ช่วงเวลาและสภาพแสงหลากหลาย เช่น เช้า กลางวัน กลางคืน ย้อนแสง ฝน เงาแรง
- รถหลายประเภท เช่น sedan, pickup, MPV, van, truck, bus และรถที่แต่งไฟหรือกระจังหน้าแตกต่าง
- ภาพที่ยากโดยตั้งใจ เช่น รถเล็กมาก รถถูกบัง motion blur ภาพมืด หรือรถอยู่ขอบเฟรม
- hard negatives เช่น ป้ายโฆษณา แสงสะท้อน กระจก ไฟถนน วัตถุที่มีรูปทรงคล้ายไฟหน้า/กระจังหน้า
- วิดีโอ sequence จริง แต่ต้อง split ตาม video/session เพื่อกัน leakage

การปรับ labeling:

- เขียน guideline ให้ชัดว่า `car_front` รวมส่วนไหน เช่น ไฟหน้า กระจังหน้า กันชนหน้า และตัดส่วน hood/กระจกหน้าแค่ไหน
- ทำ label audit โดยสุ่มทั้ง easy case และ hard case ไม่ใช่สุ่มเฉพาะภาพทั่วไป
- ตรวจ outlier เช่น box เล็ก/ใหญ่ผิดปกติ aspect ratio แปลก หรือ empty label ที่จริงควรมีรถ
- ใช้ double-check บางส่วนระหว่าง annotators เพื่อดู consistency

Experiment ที่จะรันเพิ่ม:

- Train 80-150 epochs ที่ `imgsz=640` บน GPU และใช้ early stopping
- Ablation augmentation เช่น mosaic 0.0/0.3/0.6, hsv strength, erasing on/off
- เทียบ batch size เท่าที่ VRAM 4GB รับได้ และใช้ gradient accumulation ถ้าต้องการ effective batch ใหญ่ขึ้น
- ทดลอง freeze backbone ช่วงแรกเทียบกับ fine-tune ทั้งโมเดล
- Threshold sweep ที่ 0.1-0.7 เพื่อหา operating point ที่เหมาะกับ LPR
- ทดสอบ inference บนวิดีโอจริงพร้อม temporal smoothing/tracking

วิธีวัดว่าโมเดลดีขึ้นจริง:

- ใช้ mAP50 และ mAP50-95 บน validation/test ที่แยกตาม source ไม่รั่วจาก train
- ดู precision/recall ที่ threshold ใช้งานจริง ไม่ดู mAP อย่างเดียว
- วัด false positive per minute บนวิดีโอ เพราะ production ใช้เป็น stream ไม่ใช่ภาพนิ่งอย่างเดียว
- วัด recall ของรถที่ควรถูกส่งเข้า LPR เพราะนี่สัมพันธ์กับ business objective มากกว่า metric detector ล้วน ๆ
- วัด latency/FPS และ VRAM usage บน hardware เป้าหมาย เพราะโมเดลที่แม่นขึ้นแต่ช้าเกินไปอาจใช้จริงไม่ได้

สรุป approach ของผมคือเริ่มจาก split ที่กัน leakage, augmentation ที่สอดคล้องกับ physics ของภาพจริง, failure analysis เพื่อรู้ว่าควรแก้อะไร และเลือก threshold ตาม cost ของ downstream system ไม่ใช่เลือกจาก confidence ที่ดูสวยที่สุดเพียงอย่างเดียว
