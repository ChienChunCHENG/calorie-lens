"""Stage 1 (per video): decode, pick frames, run the research locator (loc17.onnx: YOLOE-seg) for plate/bowl/tray + food masks on each frame.
usage: python stage1.py <name> <vid_dir> <ids_file> [n_frames] [cams]  -> $SP/research/mv/s1_<name>/<dish>_<cam>.npz"""
import sys, os, pickle, time
from pathlib import Path
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parents[1])); sys.path.insert(0, str(Path(__file__).parents[1] / "portion"))
from common import SP
from snaplib import letterbox
from PIL import Image
import feat  # nms, mask_of
import torch; torch.set_num_threads(3)

W, H = 640, 360
MV = SP / "research/mv"
_loc = None


def loc():
    global _loc
    if _loc is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = 3
        _loc = ort.InferenceSession(str(SP / "research/portion/loc17.onnx"), so, providers=["CPUExecutionProvider"])
    return _loc


def decode(path):
    cap = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return fr


def fit_ell(mask):
    """ellipse from the biggest blob of a bool mask, ignoring contour points on the image border (clipped plates)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)[:, 0, :]
    if len(c) < 30: return None
    h, w = mask.shape
    onb = (c[:, 0] <= 2) | (c[:, 1] <= 2) | (c[:, 0] >= w - 3) | (c[:, 1] >= h - 3)
    cc = c[~onb]
    if len(cc) < 30: return None
    (cx, cy), (a, b), ang = cv2.fitEllipse(cc.reshape(-1, 1, 2).astype(np.float32))
    return dict(cx=cx, cy=cy, major=max(a, b), minor=min(a, b), angle=ang if a >= b else ang + 90, clip=float(onb.mean()), area=float(mask.sum()))


def run_frame(rgb):
    im = Image.fromarray(rgb)
    x, r, px, py = letterbox(im, 640)
    out = loc().run(None, {"images": x}); raw = out[0][0]; protos = out[1][0]; cs = raw[4:4 + 17]
    res = dict(plates=[], food=None)
    for b in feat.nms(cs, raw, 0.10, 0.5, None, 6, r, px, py, W, H, [14, 15, 16, 1, 2, 3], maxn=30)[:4]:
        mk = feat.mask_of(raw[4 + 17:4 + 17 + 32, b["i"]], protos, r, px, py, W, H, b["box"], half=1)
        e = fit_ell(mk)
        res["plates"].append(dict(box=b["box"], score=b["score"], cls=b["cls"], ell=e, mask=np.packbits(mk), area=int(mk.sum())))
    fb = feat.nms(cs, raw, 0.10, 0.5, None, 6, r, px, py, W, H, [0], maxn=10)
    fm = np.zeros((H, W), bool)
    for b in fb[:3]:
        fm |= feat.mask_of(raw[4 + 17:4 + 17 + 32, b["i"]], protos, r, px, py, W, H, b["box"], half=1)
    res["food"] = np.packbits(fm)
    return res


if __name__ == "__main__":
    name, vd, ids = sys.argv[1], Path(sys.argv[2]), [l.strip() for l in open(sys.argv[3]) if l.strip()]
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    cams = sys.argv[5] if len(sys.argv) > 5 else "ABCD"
    od = MV / f"s1_{name}"; od.mkdir(parents=True, exist_ok=True)
    for k, d in enumerate(ids):
        for c in cams:
            p = vd / d / f"camera_{c}.h264"; o = od / f"{d}_{c}.pkl"
            if not p.exists() or o.exists(): continue
            t = time.time()
            try:
                fr = decode(p)
                if len(fr) < 6: continue
                idx = sorted(set(np.linspace(0, len(fr) - 1, stride).round().astype(int).tolist())); res = [run_frame(fr[i]) for i in idx]
                pickle.dump(dict(idx=idx, nframes=len(fr), frames=np.stack([fr[i] for i in idx]), res=res), open(o, "wb"))
            except Exception as ex: print("ERR", d, c, ex, flush=True)
            print(k, d, c, len(fr), f"{time.time()-t:.1f}s", flush=True)
