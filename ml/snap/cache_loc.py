"""Run the superset locator once per eval image and cache the thresholded raw anchors (class scores floor 0.02): python cache_loc.py [imgsz] [set,set]"""
import sys, hashlib, numpy as np, onnxruntime as ort, time
from pathlib import Path
from common import SP
from snaplib import load_img, letterbox
from evalsets import sets
imgsz = int(sys.argv[1]) if len(sys.argv) > 1 else 640
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
out = SP / "snap" / f"loc{imgsz}"; out.mkdir(parents=True, exist_ok=True)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(SP / "snap" / f"superset_{imgsz}.onnx"), so, providers=["CPUExecutionProvider"])
nc = len(__import__("json").load(open(SP / "snap/superset.json")))
t0 = time.time(); n = 0
for name, items in sets().items():
    if only and name not in only: continue
    for it in items:
        key = hashlib.md5(it["path"].encode()).hexdigest()[:12]; f = out / f"{key}.npz"
        if f.exists(): continue
        im = load_img(it["path"]); x, r, px, py = letterbox(im, imgsz)
        raw = sess.run(None, {sess.get_inputs()[0].name: x})[0][0][:4 + nc]
        m = raw[4:].max(0) >= 0.02
        np.savez(f, raw=raw[:, m].astype(np.float16), meta=np.array([r, px, py, im.width, im.height], dtype=np.float32)); n += 1
    print(name, n, round(time.time() - t0), flush=True)
