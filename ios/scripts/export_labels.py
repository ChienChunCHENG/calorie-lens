"""Write a labels json from a .pt file.

Usage: export_labels.py WEIGHTS TASK IMGSZ OUT_JSON
"""
import json
import sys

from ultralytics import YOLO

pt, task, imgsz, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
names = YOLO(pt).names
with open(out, "w", encoding="utf-8") as f:
    json.dump({"task": task, "imgsz": imgsz, "names": {str(k): v for k, v in names.items()}},
              f, ensure_ascii=False, indent=2)
print(f"{len(names)} classes -> {out}")
