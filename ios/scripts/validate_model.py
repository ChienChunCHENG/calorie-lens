"""Run Ultralytics predict with an .mlpackage (macOS only).

Usage: validate_model.py PKG TASK IMGSZ IMAGE_URL
"""
import sys

from ultralytics import YOLO

pkg, task, imgsz, src = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
model = YOLO(pkg, task="segment" if task == "seg" else "detect")
res = model.predict(src, imgsz=imgsz, conf=0.25, verbose=False)[0]
print("image shape:", res.orig_shape)
n = 0 if res.boxes is None else len(res.boxes)
print("detections:", n)
for i in range(n):
    b = res.boxes[i]
    print(f"  {res.names[int(b.cls)]:<16} conf={float(b.conf):.2f} "
          f"xyxy={[round(v) for v in b.xyxy[0].tolist()]}")
if task == "seg":
    print("masks:", None if res.masks is None else tuple(res.masks.data.shape))
if n == 0:
    sys.exit("no detections on sample image")
