"""Evaluate scorers (text baseline vs prototype variants) with the EXACT snap pipeline from tune.py (locator unchanged, gate/mixture unchanged):
negatives, drinks, UEC multi-item recall/precision, Food-101 (whole image + largest located box) with the extended US/EU label map, Taiwanese,
plus Open Images held-out crops. Test sets here are never used for fitting.
`scorer @ e` replaces `M @ e` of tune.py (see protolib)."""
import sys, json, pickle, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import tune                                   # noqa: E402  (imports the cached eval data of ml/snap)
from common import SP, MODELS                 # noqa: E402
from snaplib import load_img, clip_prep, crop_box   # noqa: E402
import class_map as cm                        # noqa: E402
import protolib as pl                         # noqa: E402

CACHE2 = SP / "research/protos/emb_eval_cache.pkl"


def setup():
    # extended Food-101 map: original accepted ids first, then every vocab id of the concept(s) the class belongs to
    ext = {}
    for k, d in cm.C.items():
        for f in d["f101"]: ext[f] = list(d["vocab"])
    for f, ids in tune.FOOD101_MAP.items():
        ext[f] = list(dict.fromkeys(ids + ext.get(f, [])))
    tune.FOOD101_MAP = ext
    tune.data()
    if CACHE2.exists(): tune._E.update(pickle.load(open(CACHE2, "rb")))
    import onnxruntime as ort
    def sess3():
        so = ort.SessionOptions(); so.intra_op_num_threads = 3
        return ort.InferenceSession(str(MODELS / "clip_image.onnx"), so, providers=["CPUExecutionProvider"])
    tune.clip_session = sess3


def save_cache():
    pickle.dump(tune._E, open(CACHE2, "wb"))


def final_params():
    P = tune.P_with(cs="A2+N3", conf=0.15, contain=0.6, iou=0.3, fb_neg=0.005, fb_food=0.02, max_det=6, pad=0.2, mode="shortest", tpl="t1/custom",
                    p_neg_max=0.1, p_food_min=0.05, k=3, cum=0.7, fallback="no_boxes")
    P["mix"]["sure"] = 0.7
    tune.EXACT_IOU = P["iou"]
    return P


def run_pipeline(scorer, P=None):
    P = P or final_params()
    tune.matrix = lambda _P: scorer
    return tune.evaluate(P)


def food101_extra(scorer, P=None):
    """per-class whole-image top-1 on the Food-101 validation images (300), for the per-class win/loss table"""
    P = P or final_params(); out = {}
    for it in tune.S["food101"]:
        acc = tune.FOOD101_MAP.get(it["label"])
        if not acc: continue
        ok = {tune.ID2I[a] for a in acc}
        e = tune._E[tune.key_of(it["path"], "whole", None, 0.0, P["mode"])].astype(np.float32)
        sc = np.where(tune.ISNEG, -9, scorer @ e)
        o = np.argsort(-sc)[:5]
        d = out.setdefault(it["label"], [0, 0, 0]); d[0] += 1; d[1] += int(o[0] in ok); d[2] += int(bool(set(o) & ok))
    return out


def oi_heldout(scorer, P=None, pad=0.2):
    """Open Images held-out crops: top-1/top-5 over food rows. returns {concept: [n, top1, top5]}"""
    root = SP / "research/protos/oi_ho"; boxes = json.load(open(root / "boxes.json"))
    ids, _ = pl.load_ids(MODELS); id2i = {x: i for i, x in enumerate(ids)}
    cache = SP / "research/protos/emb_oi_ho.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); E = z["emb"].astype(np.float32); keep = z["keep"]
    else:
        sess = tune.clip_session(); E = np.zeros((len(boxes), 512), np.float32); keep = np.zeros(len(boxes), bool); t0 = time.time()
        for i, b in enumerate(boxes):
            p = root / "img" / f"{b['id']}.jpg"
            if not p.exists(): continue
            im = load_img(p); x1, y1, x2, y2 = b["box"]
            c = crop_box(im, [x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height], pad)
            e = sess.run(None, {"image": clip_prep(c, 256, "shortest", "center")[None].astype(np.float32)})[0][0]
            E[i] = e / np.linalg.norm(e); keep[i] = True
            if i % 200 == 0: print("oi_ho", i, len(boxes), f"{time.time() - t0:.0f}s", flush=True)
        np.savez(cache, emb=E.astype(np.float16), keep=keep)
    out = {}; grp = {}
    for i, b in enumerate(boxes):
        if not keep[i]: continue
        ok = {id2i[v] for v in cm.C[b["concept"]]["vocab"]}
        sc = np.where(tune.ISNEG, -9, scorer @ E[i]); o = np.argsort(-sc)[:5]
        d = out.setdefault(b["concept"], [0, 0, 0]); d[0] += 1; d[1] += int(o[0] in ok); d[2] += int(bool(set(o) & ok))
        g = grp.setdefault(cm.C[b["concept"]]["grp"], [0, 0, 0]); g[0] += 1; g[1] += int(o[0] in ok); g[2] += int(bool(set(o) & ok))
    return out, grp


def summarize(R):
    f = R["food101"]
    return dict(negU=R["neg_user.all"][0], negX=R["neg_extra.all"][0], negX_tune=R["neg_extra.tune"][0], negX_held=R["neg_extra.held"][0],
                drinks_logged=R["drinks.all"][0], uec_recall=R["uec"]["recall"], uec_prec=R["uec"]["precision"], uec_items=R["uec"]["items_per_img"],
                f101_n=f["n_mappable"], f101_logged=f["logged_mappable"], f101_whole_top1=f["whole_top1"], f101_whole_top5=f["whole_top5"],
                f101_item_top1=f["item_top1"], f101_item_topK=f["item_topK"], tw_item_top1=R["tw"]["item_top1"], tw_whole_top1=R["tw"]["whole_top1"])
