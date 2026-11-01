"""Stage 1 (research cache): for every image run (1) FastSAM-s class-agnostic masks, (2) YOLOE container prompts (plate/bowl/tray),
(3) MobileCLIP image embeddings of every kept mask (masked crop + plain crop). Output pickle per set.
usage: python cache_seg.py <name> <dir> [ids.json]"""
import sys, os, glob, json, pickle, time
from pathlib import Path
import numpy as np, torch, cv2, onnxruntime as ort
torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import MODELS, SP
from snaplib import load_img, clip_prep
from ultralytics import FastSAM, YOLOE
W_ = SP / "research/ingr/w"
fs = FastSAM(str(W_ / "FastSAM-s.pt"))
ye = YOLOE(str(SP / "export/yoloe-11s-seg.pt"))
CN = ["plate", "bowl", "tray"]
ye.set_classes(CN, ye.get_text_pe(CN))
clip = ort.InferenceSession(str(MODELS / "clip_image.onnx"), providers=["CPUExecutionProvider"])
MS = 160  # stored mask width (height = round(160*H/W))


def square(im, box, pad=0.1, fill=(124, 124, 124), mask=None):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1; s = max(w, h) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l, t = int(round(cx - s / 2)), int(round(cy - s / 2)); S = int(round(s))
    a = np.asarray(im)
    can = np.full((S, S, 3), fill, np.uint8)
    sx0, sy0 = max(0, l), max(0, t); sx1, sy1 = min(im.width, l + S), min(im.height, t + S)
    src = a[sy0:sy1, sx0:sx1].copy()
    if mask is not None: src[~mask[sy0:sy1, sx0:sx1]] = fill
    can[sy0 - t:sy1 - t, sx0 - l:sx1 - l] = src
    from PIL import Image
    return Image.fromarray(can)


def hull(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    if cs: cv2.fillPoly(out, [cv2.convexHull(np.concatenate(cs))], 1)
    return out > 0


def unletter(d, W, H, size=640):
    """model masks come at the letterboxed inference shape (padding centred); crop the padding and resize to the image."""
    h, w = d.shape; r = size / max(W, H); nh, nw = round(H * r), round(W * r)
    top, left = (h - nh) // 2, (w - nw) // 2
    return cv2.resize(d[top:top + nh, left:left + nw], (W, H), interpolation=cv2.INTER_LINEAR)


def emb(crops):
    out = []
    for i in range(0, len(crops), 16):
        x = np.stack([clip_prep(c) for c in crops[i:i + 16]]).astype(np.float32)
        e = clip.run(None, {"image": x})[0]; out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def run(path, maxm=14):
    im = load_img(path, 1280); W, H = im.size
    r = fs(np.asarray(im)[..., ::-1].copy(), imgsz=640, conf=0.3, iou=0.8, retina_masks=False, verbose=False)[0]
    ms = r.masks.data.cpu().numpy() if r.masks is not None else np.zeros((0, 1, 1))
    # container (YOLOE)
    y = ye(np.asarray(im)[..., ::-1].copy(), imgsz=640, conf=0.1, iou=0.5, verbose=False)[0]
    cont = []
    if y.masks is not None:
        for k in range(len(y.boxes)):
            m = unletter(y.masks.data[k].cpu().numpy(), W, H) > 0.5
            m = hull(m)
            cont.append(dict(box=y.boxes.xyxy[k].tolist(), score=float(y.boxes.conf[k]), cls=int(y.boxes.cls[k]), mask=cv2.resize(m.astype(np.uint8), (MS, round(MS * H / W)), interpolation=cv2.INTER_AREA)))
    pm = None
    if cont:
        c0 = max(cont, key=lambda c: c["score"]); pm = cv2.resize(c0["mask"], (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if pm.mean() < 0.02: pm = None
    items = []
    for k in range(len(ms)):
        m = unletter(ms[k], W, H) > 0.5
        a = m.mean()
        if a < 0.0025 or a > 0.7: continue
        if pm is not None and ((m & pm).sum() / m.sum() < 0.5 or m.sum() > 0.92 * pm.sum()): continue
        if any(((m & o["m"]).sum() / (m | o["m"]).sum()) > 0.85 for o in items): continue
        ys, xs = np.where(m)
        items.append(dict(box=[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], area=float(a), m=m))
    items.sort(key=lambda d: -d["area"]); items = items[:maxm]
    ca = [square(im, it["box"], 0.1, mask=it["m"]) for it in items]; cb = [square(im, it["box"], 0.25, fill=(124, 124, 124)) for it in items]
    ea, eb = emb(ca), emb(cb)
    ew = emb([im])  # whole image
    out = dict(W=W, H=H, cont=cont, ew=ew[0], items=[])
    for it, a, b in zip(items, ea, eb):
        mm = cv2.resize(it["m"].astype(np.uint8), (MS, round(MS * H / W)), interpolation=cv2.INTER_AREA)
        out["items"].append(dict(box=[float(v) for v in it["box"]], area=it["area"], m=mm, ea=a, eb=b))
    return out


if __name__ == "__main__":
    name, src = sys.argv[1], sys.argv[2]
    paths = sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg")))
    if len(sys.argv) > 3:
        keep = set(json.load(open(sys.argv[3]))); paths = [p for p in paths if Path(p).stem in keep]
    outp = SP / f"research/ingr/seg_{name}.pkl"
    out = pickle.load(open(outp, "rb")) if outp.exists() else {}
    t0 = time.time()
    for k, p in enumerate(paths):
        key = Path(p).stem
        if key in out: continue
        try: out[key] = run(p)
        except Exception as e: print("ERR", p, e)
        if k % 25 == 0:
            print(k, len(paths), round(time.time() - t0), flush=True); pickle.dump(out, open(outp, "wb"))
    pickle.dump(out, open(outp, "wb")); print("done", name, len(out), round(time.time() - t0))
