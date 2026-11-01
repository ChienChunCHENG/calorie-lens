import sys, json, hashlib, numpy as np
from common import SP
from snaplib import decode_locator, iou_xyxy
from evalsets import sets
imgsz = 640
SUP = json.load(open(SP / "snap/superset.json")); names = [s[0] for s in SUP]
S = sets()
def load(p):
    z = np.load(SP / "snap" / f"loc{imgsz}" / (hashlib.md5(p.encode()).hexdigest()[:12] + ".npz")); return z["raw"].astype(np.float32), z["meta"]
C = {n: [(it, *load(it["path"])) for it in S[n] if True] for n in ("neg_user", "neg_extra", "drinks", "uec")}
def run(foodn, negn, conf, contain=0.85, iou=0.6, md=8):
    idx = [names.index(n) for n in foodn + negn]; isfood = [True] * len(foodn) + [False] * len(negn)
    def boxes(raw, meta):
        r, px, py, W, H = meta
        d = decode_locator(raw, len(SUP), r, px, py, W, H, conf, iou, contain, md, cls_idx=idx)
        return [x for x in d if isfood[x["cls"]]], d
    res = {}
    for n in ("neg_user", "neg_extra", "drinks"):
        k = 0; nb = 0
        for it, raw, meta in C[n]:
            f, d = boxes(raw, meta); k += len(f) > 0; nb += len(f)
        res[n] = (k, len(C[n]))
    tp = ng = npred = 0
    for it, raw, meta in C["uec"]:
        f, d = boxes(raw, meta); G = it["boxes"]; npred += len(f); ng += len(G)
        tp += sum(any(iou_xyxy(g, x["box"]) >= .5 for x in f) for g in G)
    res["uec_recall"] = round(tp / ng, 3); res["uec_boxes/img"] = round(npred / len(C["uec"]), 1)
    return res
A = ["food", "dish of food", "bowl of food", "plate of food", "drink", "cup"]
B = ["food", "bowl of food", "plate of food", "drink", "cup"]
N1 = ["empty cup", "empty glass", "empty bowl", "empty plate", "frying pan", "wok", "cooking pot", "bottle", "mug", "paper cup", "tumbler", "table", "hand", "chopsticks", "spoon", "kitchen utensil", "tray"]
N2 = ["empty cup", "empty glass", "empty bowl", "empty plate", "frying pan", "wok", "cooking pot", "bottle"]
A2 = A + ["cup of drink", "glass of drink", "bubble tea"]
N3=["empty cup","empty glass","paper cup","tumbler","mug"]
N4=["empty cup","empty glass","wok","frying pan"]
N5=["empty cup","empty glass","empty bowl","empty plate"]
for label,f,n,ct,md in [("A nocontain md50",A,[],1.01,50),("A c.85 md50",A,[],0.85,50),("A c.95 md8",A,[],0.95,8),("A+N3",A,N3,0.95,8),("A+N4",A,N4,0.95,8),("A+N5",A,N5,0.95,8),("A2+N3",A2,N3,0.95,8),("A2+N3+N4",A2,N3+N4[2:],0.95,8)]:
    for conf in (0.05,0.1):
        print(label,conf,run(f,n,conf,contain=ct,md=md),flush=True)
