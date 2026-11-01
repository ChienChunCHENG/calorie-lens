"""Cache L2-normalised crop embeddings (shipped clip_image.onnx) for every evaluation set, so lambda tuning is pure numpy.
Outputs $SP/research/prior/cache/<set>.npz + <set>.json (region metadata). Run: python cache_embeds.py"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, onnxruntime as ort
from common import MODELS, SP
from snaplib import load_img, letterbox, clip_prep, crop_box, decode_locator
from evalsets import sets

CACHE = SP / "research/prior/cache"; CACHE.mkdir(parents=True, exist_ok=True)
L = json.load(open(MODELS / "locator.json")); C = json.load(open(MODELS / "clip.json"))
so = ort.SessionOptions(); so.intra_op_num_threads = 3
ls = ort.InferenceSession(str(MODELS / "locator.onnx"), so); cs = ort.InferenceSession(str(MODELS / "clip_image.onnx"), so)
nc = len(L["classes"])

def emb(im):
    e = cs.run(None, {C["input"]: clip_prep(im, C["size"], C["resize"], C["crop"])[None]})[0][0]
    return e / np.linalg.norm(e)

def regions(path):
    """the exact shipped region logic: food boxes (crop_pad) ; whole-image fallback only when locator returned no box at all"""
    im = load_img(path); x, r, px, py = letterbox(im, L["imgsz"]); raw = ls.run(None, {L["input"]: x})[0][0]
    d = decode_locator(raw, nc, r, px, py, im.width, im.height, L["conf"], L["iou"], L["contain"], L["max_det"])
    out = []
    for b in d:
        if L["classes"][b["cls"]]["food"]: out.append(("box", b["box"], emb(crop_box(im, b["box"], L["crop_pad"]))))
    if not d: out.append(("fallback", [0, 0, im.width, im.height], emb(im)))
    return out

def save(name, meta, E):
    np.savez(CACHE / f"{name}.npz", E=np.array(E, dtype=np.float32)); json.dump(meta, open(CACHE / f"{name}.json", "w"))
    print(name, len(meta), flush=True)

if __name__ == "__main__":
    S = sets(); only = sys.argv[1:]
    man = json.load(open(SP / "uxset/manifest.json"))["items"]
    jobs = {}
    jobs["neg_user"] = [it["path"] for it in S["neg_user"]]
    jobs["neg_extra"] = [it["path"] for it in S["neg_extra"]]
    jobs["drinks"] = [it["path"] for it in S["drinks"]]
    jobs["neg_ux"] = [str(SP / "uxset" / x["file"]) for x in man if x["set"] == "negatives"]
    jobs["ux_img"] = sorted(glob.glob(str(SP / "ux/img/*.jpg")))
    for name, paths in jobs.items():
        if only and name not in only: continue
        meta, E = [], []
        for p in paths:
            try: rs = regions(p)
            except Exception as ex: print("skip", p, ex); continue
            for kind, box, e in rs:
                meta.append(dict(path=p, kind=kind, box=[float(v) for v in box])); E.append(e)
        # images with zero kept regions still matter (count as 'nothing logged'): record image list
        json.dump(paths, open(CACHE / f"{name}.images.json", "w"))
        save(name, meta, E)
    if not only or "food101" in only:
        meta, E = [], []
        for it in S["food101"]:
            im = load_img(it["path"]); meta.append(dict(path=it["path"], label=it["label"])); E.append(emb(im))
        save("food101", meta, E)
    if not only or "oi" in only:
        meta, E = [], []
        for x in man:
            if x["set"] not in ("meals", "multi"): continue
            im = load_img(str(SP / "uxset" / x["file"]))
            for b in x["boxes"]:
                W, H = im.size; box = [b["x1"] * W, b["y1"] * H, b["x2"] * W, b["y2"] * H]
                if box[2] - box[0] < 24 or box[3] - box[1] < 24: continue
                meta.append(dict(id=x["id"], set=x["set"], cls=b["class"], path=x["file"])); E.append(emb(crop_box(im, box, L["crop_pad"])))
        save("oi", meta, E)
