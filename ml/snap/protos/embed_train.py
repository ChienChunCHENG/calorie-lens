"""Embed the prototype training images with the SHIPPED image encoder (web/public/models/clip_image.onnx, same preprocessing as the app:
resize shortest side to 256 bilinear + centre crop 256, fp32, L2-normalised). Food-101 train photos are embedded whole; Open Images train boxes
are embedded as the crop the app would take (box + 20 % padding on each side).
Caches: $SP/research/protos/emb_f101.npz  (emb fp16, concept idx, path)   emb_oi.npz (same, + oi class)
usage: python embed_train.py [f101|oi|ho]   (one heavy job at a time; 3 threads)"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SP, MODELS
from snaplib import load_img, clip_prep, crop_box
import class_map as cm

P = SP / "research/protos"
CONCEPTS = list(cm.C)


def session():
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 3
    return ort.InferenceSession(str(MODELS / "clip_image.onnx"), so, providers=["CPUExecutionProvider"])


def embed_batches(sess, make, n, bs=16, tag=""):
    """make(i) -> PIL image or None. returns (n, 512) float32 (zeros where None) and ok mask"""
    out = np.zeros((n, 512), np.float32); ok = np.zeros(n, bool); t0 = time.time(); buf = []
    def flush():
        if not buf: return
        e = sess.run(None, {"image": np.stack([a for _, a in buf]).astype(np.float32)})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        for (i, _), v in zip(buf, e): out[i] = v; ok[i] = True
        buf.clear()
    for i in range(n):
        try:
            im = make(i)
            if im is not None: buf.append((i, clip_prep(im, 256, "shortest", "center")))
        except Exception as ex: print("skip", tag, i, ex)
        if len(buf) >= bs: flush()
        if i % 200 == 0: print(tag, i, n, f"{time.time() - t0:.0f}s", flush=True)
    flush(); return out, ok


def f101():
    root = P / "f101"; items = []
    src = {}  # f101 class -> concept index
    for ci, k in enumerate(CONCEPTS):
        for f in cm.C[k]["f101"]: src[f] = ci
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name in src:
            for p in sorted(d.glob("*.jpg")): items.append((str(p), src[d.name], d.name))
    print("food101 train images", len(items))
    sess = session()
    E, ok = embed_batches(sess, lambda i: load_img(items[i][0]), len(items), tag="f101")
    np.savez(P / "emb_f101.npz", emb=E[ok].astype(np.float16), concept=np.array([items[i][1] for i in range(len(items)) if ok[i]]),
             cls=np.array([items[i][2] for i in range(len(items)) if ok[i]]), path=np.array([items[i][0] for i in range(len(items)) if ok[i]]))


def oi():
    boxes = json.load(open(P / "oi/train_boxes.json")); c2i = {k: i for i, k in enumerate(CONCEPTS)}
    boxes = [b for b in boxes if (P / "oi/img" / f"{b['id']}.jpg").exists()]
    print("oi train boxes", len(boxes)); sess = session(); cache = {}

    def mk(i):
        b = boxes[i]
        if b["id"] not in cache: cache.clear(); cache[b["id"]] = load_img(P / "oi/img" / f"{b['id']}.jpg")
        im = cache[b["id"]]; x1, y1, x2, y2 = b["box"]
        return crop_box(im, [x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height], 0.2)
    E, ok = embed_batches(sess, mk, len(boxes), tag="oi")
    np.savez(P / "emb_oi.npz", emb=E[ok].astype(np.float16), concept=np.array([c2i[boxes[i]["concept"]] for i in range(len(boxes)) if ok[i]]),
             cls=np.array([boxes[i]["oi"] for i in range(len(boxes)) if ok[i]]), path=np.array([boxes[i]["id"] for i in range(len(boxes)) if ok[i]]))


if __name__ == "__main__":
    {"f101": f101, "oi": oi}[sys.argv[1]]()
