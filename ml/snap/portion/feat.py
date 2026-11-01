"""Stage 1: run the shipped snap pipeline (locator + MobileCLIP crop classifier) on images and cache everything portion.py needs.

Uses the research locator loc17.onnx = the 14 shipped prompts (rows 0..13, identical scores) + container prompts plate/bowl/tray (rows 14..16),
plus the 32 mask-prototype channels. Items = exactly what spec/snap_contract.md produces (NMS iou, contain, max_det, food filter, CLIP gate).
usage: python feat.py <name> <image dir or labels.json> -> $SP/research/portion/feat_<name>.pkl
"""
import sys, json, pickle, glob, os
from pathlib import Path
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import MODELS, SP
from snaplib import load_img, letterbox, clip_prep, crop_box, iou_xyxy, inside_frac

RP = SP / "research/portion"
L = json.load(open(MODELS / "locator.json")); C = json.load(open(MODELS / "clip.json")); V = json.load(open(MODELS / "vocab.json", encoding="utf-8"))
NC = 17; NSHIP = 14
M = np.frombuffer((MODELS / "vocab_emb.bin").read_bytes(), "<f2").astype(np.float32).reshape(len(V), -1)
ISNEG = np.array([bool(v.get("neg")) for v in V])
_loc = _clip = None


def sessions():
    global _loc, _clip
    if _loc is None:
        _loc = ort.InferenceSession(str(RP / "loc17.onnx"), providers=["CPUExecutionProvider"])
        _clip = ort.InferenceSession(str(MODELS / "clip_image.onnx"), providers=["CPUExecutionProvider"])
    return _loc, _clip


def sig(x): return 1 / (1 + np.exp(-x))


def mask_of(coef, protos, r, px, py, W, H, box, half=2):
    """instance mask in image space at 1/half resolution, restricted to its box. returns packed bits + shape"""
    m = sig(coef @ protos.reshape(32, -1)).reshape(160, 160)
    s = 160 / 640
    x0, y0 = int(round(px * s)), int(round(py * s)); x1, y1 = 160 - x0, 160 - y0
    m = m[y0:y1, x0:x1]
    w2, h2 = W // half, H // half
    m = cv2.resize(m, (w2, h2), interpolation=cv2.INTER_LINEAR) > 0.5
    bx = [int(v / half) for v in box]
    keep = np.zeros_like(m); keep[bx[1]:bx[3] + 1, bx[0]:bx[2] + 1] = True
    return m & keep


def nms(cs, raw, conf, iou, contain, max_det, r, px, py, W, H, idxs, maxn=100):
    sub = cs[idxs]; best = sub.argmax(0); sc = sub.max(0)
    keep = np.where(sc >= conf)[0]; keep = keep[np.argsort(-sc[keep])]
    cand = []
    for i in keep:
        cx, cy, w, h = raw[:4, i]
        b = [(cx - w / 2 - px) / r, (cy - h / 2 - py) / r, (cx + w / 2 - px) / r, (cy + h / 2 - py) / r]
        b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
        if b[2] - b[0] < 4 or b[3] - b[1] < 4: continue
        if any(iou_xyxy(b, c["box"]) > iou for c in cand): continue
        cand.append(dict(box=b, score=float(sc[i]), cls=int(idxs[best[i]]), i=int(i)))
        if len(cand) >= maxn: break
    out = []
    for c in cand:
        if contain is not None and any(inside_frac(c["box"], o["box"]) >= contain for o in out): continue
        out.append(c)
    return out[:max_det]


def classify(im_crop, gate):
    _, cs = sessions()
    e = cs.run(None, {C["input"]: clip_prep(im_crop, C["size"], C["resize"], C["crop"])[None].astype(np.float32)})[0][0]; e /= np.linalg.norm(e)
    z = C["logit_scale"] * (M @ e); p = np.exp(z - z.max()); p /= p.sum()
    pneg = float(p[ISNEG].sum()); fp = np.where(ISNEG, 0, p); top = float(fp.max())
    if pneg > gate[0] or top < gate[1]: return None
    q = fp / fp.sum(); order = np.argsort(-q)[:C["mix"]["k"]]
    cut = int(np.searchsorted(np.cumsum(q[order]), C["mix"]["cum"]) + 1); order = order[:max(1, cut)]
    pr = q[order] / q[order].sum()
    return dict(cand=[(int(i), float(w)) for i, w in zip(order, pr)], top1=float(pr[0]), pneg=pneg, top_food=top)


def extract(path):
    loc, _ = sessions()
    im = load_img(path); W, H = im.size
    x, r, px, py = letterbox(im, 640)
    out = loc.run(None, {"images": x}); raw = out[0][0]; protos = out[1][0]
    cs = raw[4:4 + NC]
    ship = nms(cs, raw, L["conf"], L["iou"], L["contain"], L["max_det"], r, px, py, W, H, list(range(NSHIP)))
    res = dict(W=W, H=H, items=[], containers=[], foodboxes=[])
    g = (C["gate"]["p_neg_max"], C["gate"]["p_food_min"])
    for b in ship:
        if not L["classes"][b["cls"]]["food"]: continue
        cl = classify(crop_box(im, b["box"], L["crop_pad"]), g)
        if cl is None: continue
        mk = mask_of(raw[4 + NC:4 + NC + 32, b["i"]], protos, r, px, py, W, H, b["box"])
        res["items"].append(dict(box=b["box"], score=b["score"], cls=b["cls"], mask=np.packbits(mk), mshape=mk.shape, **cl))
    if not ship and L.get("fallback_whole"):
        f = L["fallback_whole"]; cl = classify(im, (f["p_neg_max"], f["p_food_min"]))
        if cl: res["items"].append(dict(box=[0, 0, W, H], score=0, cls=-1, mask=None, mshape=None, **cl))
    # container candidates: classes 14..16 (plate, bowl, tray) + 'bowl of food'(2), 'plate of food'(3), 'dish of food'(1); looser NMS, no contain
    for b in nms(cs, raw, 0.10, 0.5, None, 6, r, px, py, W, H, [14, 15, 16, 1, 2, 3], maxn=30):
        mk = mask_of(raw[4 + NC:4 + NC + 32, b["i"]], protos, r, px, py, W, H, b["box"])
        res["containers"].append(dict(box=b["box"], score=b["score"], cls=b["cls"], mask=np.packbits(mk), mshape=mk.shape))
    return res


if __name__ == "__main__":
    name, src = sys.argv[1], sys.argv[2]
    paths = sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.jpeg")))
    if name == "dev":
        keep = {d["dish"] for d in json.load(open(src + "/labels.json"))}; paths = [p for p in paths if Path(p).stem in keep]
    out = {}
    for k, p in enumerate(paths):
        try: out[Path(p).stem] = extract(p)
        except Exception as e: print("ERR", p, e)
        if k % 50 == 0: print(k, len(paths), flush=True)
    pickle.dump(out, open(RP / f"feat_{name}.pkl", "wb"))
    print("done", name, len(out))
