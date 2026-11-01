"""End-to-end check using ONLY the shipped files in web/public/models (locator.onnx, clip_image.onnx, vocab_emb.bin, json)."""
import json, sys, numpy as np, onnxruntime as ort
from common import MODELS
from snaplib import *
from evalsets import sets
L = json.load(open(MODELS / "locator.json")); C = json.load(open(MODELS / "clip.json")); V = json.load(open(MODELS / "vocab.json", encoding="utf-8"))
M = np.frombuffer((MODELS / "vocab_emb.bin").read_bytes(), "<f2").astype(np.float32).reshape(len(V), -1)
neg = np.array([bool(v.get("neg")) for v in V]); ls = ort.InferenceSession(str(MODELS / "locator.onnx")); cs = ort.InferenceSession(str(MODELS / "clip_image.onnx"))
nc = len(L["classes"])
def cls(im, gate):
    e = cs.run(None, {C["input"]: clip_prep(im, C["size"], C["resize"], C["crop"])[None]})[0][0]; e /= np.linalg.norm(e)
    z = C["logit_scale"] * (M @ e); p = np.exp(z - z.max()); p /= p.sum(); pf = np.where(neg, 0, p)
    if p[neg].sum() > gate[0] or pf.max() < gate[1]: return None
    return int(pf.argmax())
def run(path):
    im = load_img(path); x, r, px, py = letterbox(im, L["imgsz"]); raw = ls.run(None, {L["input"]: x})[0][0]
    d = decode_locator(raw, nc, r, px, py, im.width, im.height, L["conf"], L["iou"], L["contain"], L["max_det"])
    food = [b for b in d if L["classes"][b["cls"]]["food"]]; g = (C["gate"]["p_neg_max"], C["gate"]["p_food_min"]); out = []
    for b in food:
        k = cls(crop_box(im, b["box"], L["crop_pad"]), g)
        if k is not None: out.append(V[k]["id"])
    if not d and L.get("fallback_whole"):
        f = L["fallback_whole"]; k = cls(im, (f["p_neg_max"], f["p_food_min"]))
        if k is not None: out.append(V[k]["id"])
    return out
S = sets()
for n in ("neg_user", "neg_extra", "drinks"):
    r = [(it["path"].split("\\")[-1], run(it["path"])) for it in S[n]]
    print(n, "images logging anything:", sum(1 for _, o in r if o), "/", len(r), [(a, o) for a, o in r if o and n != "drinks"][:6])
    if n == "drinks": print("  drink top1s:", [o[0] if o else None for _, o in r][:12])
