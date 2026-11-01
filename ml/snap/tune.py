"""Full snap pipeline in Python + parameter tuning (see ml/snap/README.md).

Stages (all cached under $SP/snap/):
  python tune.py embed      # locator raw outputs (cache_loc.py) -> candidate boxes -> CLIP embeddings of crops/whole images
  python tune.py text       # text matrices for the template variants
  python tune.py sweep      # grid / curves -> $SP/snap/sweep.json + printed tables
  python tune.py final      # evaluate the chosen params, write locator_classes.json + clip.json (gate/mix/templates), results.json
Pipeline mirror of spec/snap_contract.md steps 2-6. `evaluate()` is the single source of the reported numbers.
"""
import sys, json, hashlib, pickle, re, math, itertools, time
from pathlib import Path
import numpy as np
from common import SP, MODELS, load_clip
from snaplib import *
from evalsets import sets, FOOD101_MAP, TW_ACCEPT

HERE = Path(__file__).parent
CACHE = SP / "snap"
SUP = [tuple(x) for x in json.load(open(CACHE / "superset.json"))]
SUPN = [s[0] for s in SUP]
IMGSZ = 640
A = ["food", "dish of food", "bowl of food", "plate of food", "drink", "cup"]
A2 = A + ["cup of drink", "glass of drink", "bubble tea"]
N3 = ["empty cup", "empty glass", "paper cup", "tumbler", "mug"]
N4 = ["wok", "frying pan"]
CLASSSETS = {  # name -> (food prompts, non-food prompts)
    "A": (A, []), "A2": (A2, []), "A+N3": (A, N3), "A2+N3": (A2, N3), "A2+N3+N4": (A2, N3 + N4), "A+N3+N4": (A, N3 + N4),
}
CAND_CONF, CAND_MD = 0.05, 14
PADS = [0.0, 0.1, 0.2]
MODES = ["shortest", "squash"]
S = sets()
VOCAB = build_vocab(); NF = len(load_foods())
ISNEG = np.array([bool(v.get("neg")) for v in VOCAB])
ID2I = {v["id"]: i for i, v in enumerate(VOCAB)}
KCAL_TYP = np.array([v["kcal100"] * v["portion_g"][1] / 100 for v in VOCAB])


def key_of(*a): return hashlib.md5("|".join(map(str, a)).encode()).hexdigest()[:16]


def loc_raw(path):
    z = np.load(CACHE / f"loc{IMGSZ}" / (hashlib.md5(path.encode()).hexdigest()[:12] + ".npz"))
    return z["raw"].astype(np.float32), z["meta"]


def csidx(cs):
    f, n = CLASSSETS[cs]; names = f + n
    return [SUPN.index(x) for x in names], [True] * len(f) + [False] * len(n)


EXACT_IOU = None  # final stage: decode with the real iou instead of post-filtering the iou-0.6 candidate list


def candidates(path, cs):
    """boxes after NMS (no contain), conf floor 0.05: list of dict(box, score, cls, food) in image px; plus image size."""
    raw, (r, px, py, W, H) = loc_raw(path)
    idx, isfood = csidx(cs)
    d = decode_locator(raw, len(SUP), r, px, py, int(W), int(H), CAND_CONF, EXACT_IOU or 0.6, 1.01, CAND_MD, cls_idx=idx)
    for x in d: x["food"] = isfood[x["cls"]]
    return d, (int(W), int(H))


# ------------------------------------------------------------------------------------------------------- embedding stage
def clip_session():
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 6
    return ort.InferenceSession(str(MODELS / "clip_image.onnx"), so, providers=["CPUExecutionProvider"])


def stage_embed():
    ef = CACHE / "emb.pkl"
    E = pickle.load(open(ef, "rb")) if ef.exists() else {}
    sess = clip_session(); t0 = time.time(); todo = 0
    allitems = [(n, it) for n, its in S.items() for it in its]
    for k, (n, it) in enumerate(allitems):
        p = it["path"]; need = []  # (key, pil, mode)
        im = None
        specs = [("whole", None, 0.0)]
        for cs in CLASSSETS:
            d, _ = candidates(p, cs)
            for x in d:
                if x["food"]:
                    for pad in PADS: specs.append(("box", tuple(int(round(v)) for v in x["box"]), pad))
        seen = set()
        for kind, box, pad in specs:
            for mode in MODES:
                kk = key_of(p, kind, box, pad, mode)
                if kk in E or kk in seen: continue
                seen.add(kk)
                if im is None: im = load_img(p)
                c = im if kind == "whole" else crop_box(im, box, pad)
                need.append((kk, clip_prep(c, 256, mode, "center" if mode == "shortest" else None)))
        for i in range(0, len(need), 16):
            ch = need[i:i + 16]
            e = sess.run(None, {"image": np.stack([a for _, a in ch]).astype(np.float32)})[0]
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            for (kk, _), v in zip(ch, e): E[kk] = v.astype(np.float16)
        todo += len(need)
        if k % 40 == 0:
            print(f"{k}/{len(allitems)} new={todo} {time.time() - t0:.0f}s", flush=True); pickle.dump(E, open(ef, "wb"))
    pickle.dump(E, open(ef, "wb")); print("done", len(E), f"{time.time() - t0:.0f}s")


def stage_text():
    model, pre, tok = load_clip(); foods = load_foods(); cache = {}
    for ft, nt in [("t1", "custom"), ("t2", "custom"), ("t3", "custom"), ("t4", "custom"), ("t1", "simple"), ("t1", "first")]:
        text_matrix(model, tok, foods, FOOD_TEMPLATES[ft], nt, cache); print(ft, nt, flush=True)
    pickle.dump(cache, open(CACHE / "text.pkl", "wb"))


TPL_KEYS = {"t1/custom": (("a photo of {}.",), "custom"), "t2/custom": (("a photo of {}, a type of food.",), "custom"),
            "t3/custom": (tuple(FOOD_TEMPLATES["t3"]), "custom"), "t4/custom": (tuple(FOOD_TEMPLATES["t4"]), "custom"),
            "t1/simple": (("a photo of {}.",), "simple"), "t1/first": (("a photo of {}.",), "first")}
_E = _T = None


def data():
    global _E, _T
    if _E is None:
        _E = pickle.load(open(CACHE / "emb.pkl", "rb")); _T = pickle.load(open(CACHE / "text.pkl", "rb"))
    return _E, _T


# -------------------------------------------------------------------------------------------------------- pipeline + metrics
def classify(e, M, scale, gate, mixc, need_mix):
    """e: unit embedding; returns None if gated out, else dict(top=[(idx,p)...], samples?)"""
    z = scale * (M @ e); z = z - z.max(); p = np.exp(z); p /= p.sum()
    pneg = float(p[ISNEG].sum()); fp = np.where(ISNEG, 0.0, p)
    top_food = float(fp.max())
    q = fp / fp.sum(); order = np.argsort(-q)[:mixc["k"]]
    cum = np.cumsum(q[order]); cut = int(np.searchsorted(cum, mixc["cum"]) + 1); order = order[:max(1, cut)]
    pr = q[order] / q[order].sum()
    out = dict(pneg=pneg, top_food=top_food, cand=[(int(i), float(w)) for i, w in zip(order, pr)])
    out["gated"] = pneg > gate["p_neg_max"] or top_food < gate["p_food_min"]
    if need_mix and not out["gated"]:
        s = mixture(VOCAB, pr, list(order)); out["p10"], out["p50"], out["p90"] = np.percentile(s, [10, 50, 90])
        out["top1_kcal"] = float(KCAL_TYP[order[0]])
    return out


_SESS = None
_IMC = {}


def emb_get(path, kind, box, pad, mode):
    """cached CLIP embedding; computes (and caches) on a miss so non-candidate boxes still evaluate exactly"""
    global _SESS
    E, _ = data(); kk = key_of(path, kind, box, pad, mode)
    if kk not in E:
        if _SESS is None: _SESS = clip_session()
        im = _IMC.get(path) or _IMC.setdefault(path, load_img(path))
        c = im if kind == "whole" else crop_box(im, box, pad)
        e = _SESS.run(None, {"image": clip_prep(c, 256, mode, "center" if mode == "shortest" else None)[None].astype(np.float32)})[0][0]
        E[kk] = (e / np.linalg.norm(e)).astype(np.float16)
    return E[kk]


def run_image(it, P, M, need_mix=False):
    E, _ = data(); p = it["path"]
    d, (W, H) = candidates(p, P["cs"])
    d = [x for x in d if x["score"] >= P["conf"]]
    if P["iou"] < 0.6 and not EXACT_IOU:  # stricter NMS on the (iou 0.6) candidate list, same greedy order
        d2 = []
        for c in d:
            if any(iou_xyxy(c["box"], o["box"]) > P["iou"] for o in d2): continue
            d2.append(c)
        d = d2
    kept = []
    for c in d:
        if any(inside_frac(c["box"], o["box"]) >= P["contain"] for o in kept): continue
        kept.append(c)
    kept = kept[:P["max_det"]]
    food = [x for x in kept if x["food"]]
    regions = [(x["box"], emb_get(p, "box", tuple(int(round(v)) for v in x["box"]), P["pad"], P["mode"]), P["gate"]) for x in food]
    if (P["fallback"] == "no_boxes" and not kept) or (P["fallback"] == "no_food" and not food):
        regions = [([0, 0, W, H], emb_get(p, "whole", None, 0.0, P["mode"]), dict(p_neg_max=P["fb_neg"], p_food_min=P["fb_food"]))]
    items = []
    for box, e, gate in regions:
        r = classify(e.astype(np.float32), M, P["scale"], gate, P["mix"], need_mix)
        if not r["gated"]: r["box"] = box; items.append(r)
    return items


def whole_cands(it, P, M):
    E, _ = data()
    e = E[key_of(it["path"], "whole", None, 0.0, P["mode"])].astype(np.float32)
    return classify(e, M, P["scale"], dict(p_neg_max=2, p_food_min=0), P["mix"], False)


def matrix(P):
    _, T = data(); k = TPL_KEYS[P["tpl"]]; return T[(k[0], k[1])]


def half(items, which):  # tune/held-out split by index parity
    return [x for i, x in enumerate(items) if which == "all" or i % 2 == (0 if which == "tune" else 1)]


def evaluate(P, full=True):
    M = matrix(P); R = {}
    # (a) negatives and (b) drinks
    for name in ("neg_user", "neg_extra", "drinks"):
        for w in (("all",) if name == "neg_user" else ("all", "tune", "held")):
            its = half(S[name], "tune" if w == "tune" else "held" if w == "held" else "all")
            logged = 0; drink = 0
            for it in its:
                items = run_image(it, P, M)
                logged += len(items) > 0
                drink += any(VOCAB[x["cand"][0][0]]["cat"] == "drink" for x in items)
            R[f"{name}.{w}"] = (logged / len(its), drink / len(its), len(its))
    if not full: return R
    # (c) UEC: recall/precision of logged items
    tp = ng = npred = ptp = 0
    for it in S["uec"]:
        items = run_image(it, P, M); G = it["boxes"]; ng += len(G); npred += len(items)
        tp += sum(any(iou_xyxy(g, x["box"]) >= .5 for x in items) for g in G)
        ptp += sum(any(iou_xyxy(g, x["box"]) >= .5 for g in G) for x in items)
    R["uec"] = dict(recall=tp / ng, precision=ptp / max(npred, 1), items_per_img=npred / len(S["uec"]), gt_per_img=ng / len(S["uec"]))
    # Food-101
    n = logged = t1 = t5 = wt1 = wt5 = nm = lg_m = 0; err_mix = []; err_top1 = []; err_top1_whole = []
    for it in S["food101"]:
        items = run_image(it, P, M, need_mix=True); logged += len(items) > 0
        acc = FOOD101_MAP.get(it["label"])
        if not acc: continue
        nm += 1; ok = {ID2I[a] for a in acc}; lg_m += len(items) > 0
        w = whole_cands(it, P, M); wt1 += w["cand"][0][0] in ok
        # whole-image top-5 over food rows
        _E, _ = data(); e = _E[key_of(it["path"], "whole", None, 0.0, P["mode"])].astype(np.float32)
        sc = np.where(ISNEG, -9, M @ e); wt5 += bool(set(np.argsort(-sc)[:5]) & ok)
        if items:
            big = max(items, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
            t1 += big["cand"][0][0] in ok
            t5 += bool({c for c, _ in big["cand"]} & ok)  # top-K mixture candidates (<=mix.k)
            truth = KCAL_TYP[ID2I[acc[0]]]
            tot_mix = sum(x["p50"] for x in items); tot_top1 = sum(x["top1_kcal"] for x in items)
            err_mix.append(abs(tot_mix - truth) / truth); err_top1.append(abs(tot_top1 - truth) / truth)
    R["food101"] = dict(n=len(S["food101"]), logged_any=logged / len(S["food101"]), n_mappable=nm, logged_mappable=lg_m / nm,
                        whole_top1=wt1 / nm, whole_top5=wt5 / nm, item_top1=t1 / max(lg_m, 1), item_topK=t5 / max(lg_m, 1),
                        kcal_mape_mix_median=float(np.median(err_mix)) if err_mix else None, kcal_mape_top1=float(np.median(err_top1)) if err_top1 else None,
                        kcal_n=len(err_mix))
    # Taiwanese
    t1 = t3 = lg = wt1 = wt3 = 0
    for it in S["tw"]:
        rx = re.compile(TW_ACCEPT[it["label"]], re.I); items = run_image(it, P, M)
        match = lambda c: bool(rx.search(VOCAB[c]["en"]))
        w = whole_cands(it, P, M); E_, _ = data()
        e = E_[key_of(it["path"], "whole", None, 0.0, P["mode"])].astype(np.float32); sc = np.where(ISNEG, -9, M @ e); top3 = np.argsort(-sc)[:3]
        wt1 += match(top3[0]); wt3 += any(match(c) for c in top3)
        lg += len(items) > 0
        if items:  # best over items: any item whose top1 / top-3 matches
            t1 += any(match(x["cand"][0][0]) for x in items); t3 += any(any(match(c) for c, _ in x["cand"][:3]) for x in items)
    n = len(S["tw"]); R["tw"] = dict(n=n, logged_any=lg / n, item_top1=t1 / n, item_top3=t3 / n, whole_top1=wt1 / n, whole_top3=wt3 / n)
    return R


BASE = dict(cs="A2+N3", conf=0.10, contain=0.95, iou=0.6, fb_neg=0.1, fb_food=0.02, max_det=8, pad=0.1, mode="shortest", tpl="t1/custom", scale=64.25,
            gate=dict(p_neg_max=0.5, p_food_min=0.02), mix=dict(k=5, cum=0.9, sure=0.5), fallback="no_boxes")


def P_with(**kw):
    P = json.loads(json.dumps(BASE))
    for k, v in kw.items():
        if k in ("p_neg_max", "p_food_min"): P["gate"][k] = v
        elif k in ("k", "cum", "sure"): P["mix"][k] = v
        else: P[k] = v
    return P


def brief(R):
    f = lambda t: f"{t[0]:.2f}/{t[1]:.2f}"
    s = f"negU {R['neg_user.all'][0]:.2f} negX {R['neg_extra.all'][0]:.2f} (t{R['neg_extra.tune'][0]:.2f}/h{R['neg_extra.held'][0]:.2f}) drinkLogged {R['drinks.all'][0]:.2f} drinkIsDrink {R['drinks.all'][1]:.2f}"
    if "uec" in R:
        s += f" | uecR {R['uec']['recall']:.2f} P {R['uec']['precision']:.2f} it/img {R['uec']['items_per_img']:.1f} | f101 logged {R['food101']['logged_mappable']:.2f} w1 {R['food101']['whole_top1']:.2f} w5 {R['food101']['whole_top5']:.2f} i1 {R['food101']['item_top1']:.2f} kcalMix {R['food101']['kcal_mape_mix_median']:.2f} top1 {R['food101']['kcal_mape_top1']:.2f} | tw logged {R['tw']['logged_any']:.2f} i1 {R['tw']['item_top1']:.2f} i3 {R['tw']['item_top3']:.2f} w1 {R['tw']['whole_top1']:.2f}"
    return s


def score(R):
    """Scalar used for greedy selection. Hard constraint: user's negatives must log nothing."""
    if R["neg_user.all"][0] > 0: return -9 + (1 - R["neg_user.all"][0])
    return (R["drinks.all"][1] + R["food101"]["logged_mappable"] + R["uec"]["recall"] + 0.5 * R["uec"]["precision"] + 0.5 * R["tw"]["logged_any"]
            + 0.5 * R["food101"]["item_top1"] - 1.0 * R["food101"]["kcal_mape_mix_median"] - 2.0 * R["neg_extra.all"][0])


AXES = [("tpl", ["t1/custom", "t4/custom"]), ("mode", MODES), ("pad", PADS), ("cs", list(CLASSSETS)), ("conf", [0.05, 0.08, 0.1, 0.15, 0.2]),
        ("contain", [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]), ("iou", [0.3, 0.4, 0.5, 0.6]), ("fallback", ["no_boxes", "no_food", "never"]),
        ("fb_neg", [0.005, 0.02, 0.05, 0.1]), ("fb_food", [0.02, 0.1, 0.2, 0.3]),
        ("p_neg_max", [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]), ("p_food_min", [0.0, 0.02, 0.05, 0.1, 0.15]),
        ("max_det", [4, 6, 8]), ("k", [3, 5, 8]), ("cum", [0.7, 0.9, 0.98])]


def sweep():
    P = P_with(); log = {}
    cur = evaluate(P); print("BASE", brief(cur), "score", round(score(cur), 3), flush=True)
    for rnd in range(2):
        for ax, vals in AXES:
            res = []
            for v in vals:
                Q = json.loads(json.dumps(P)); Q2 = P_with(**{**flat(P), ax: v}) ; R = evaluate(Q2)
                res.append((v, score(R), R)); print(f"  r{rnd} {ax}={v}: {brief(R)} score {score(R):.3f}", flush=True)
            best = max(res, key=lambda t: t[1]); P = P_with(**{**flat(P), ax: best[0]})
            log[f"r{rnd}.{ax}"] = [(v, sc, R) for v, sc, R in res]
            print(f"-> {ax} = {best[0]}", flush=True)
    json.dump(dict(P=P, log=log), open(CACHE / "sweep.json", "w"), default=float, indent=1)
    print("CHOSEN", json.dumps(P))


def flat(P):
    d = {k: v for k, v in P.items() if k not in ("gate", "mix")}; d.update(P["gate"]); d.update({k: v for k, v in P["mix"].items()}); return d


def final():
    global EXACT_IOU
    P = P_with(cs="A2+N3", conf=0.15, contain=0.6, iou=0.3, fb_neg=0.005, fb_food=0.02, max_det=6, pad=0.2, mode="shortest", tpl="t1/custom",
               p_neg_max=0.1, p_food_min=0.05, k=3, cum=0.7, fallback="no_boxes")
    EXACT_IOU = P["iou"]
    R = evaluate(P); print("FINAL", brief(R), flush=True)
    # calibrate mix.sure: accuracy of top-1 (renormalised food prob) bins on mappable Food-101 items + TW items
    M = matrix(P); pts = []
    for it in S["food101"]:
        acc = FOOD101_MAP.get(it["label"])
        if not acc: continue
        ok = {ID2I[a] for a in acc}
        for x in run_image(it, P, M): pts.append((x["cand"][0][1], x["cand"][0][0] in ok))
    for it in S["tw"]:
        rx = re.compile(TW_ACCEPT[it["label"]], re.I)
        for x in run_image(it, P, M): pts.append((x["cand"][0][1], bool(rx.search(VOCAB[x["cand"][0][0]]["en"]))))
    pts = np.array(pts, dtype=float); cal = []
    for lo, hi in [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi); cal.append((lo, hi, int(m.sum()), float(pts[m, 1].mean()) if m.sum() else None))
    print("calibration (top-1 mixture weight -> accuracy):", cal)
    curve = []
    for pn in [0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        Rq = evaluate(P_with(**{**flat(P), "p_neg_max": pn}))
        curve.append(dict(p_neg_max=pn, neg_user=Rq["neg_user.all"][0], neg_extra=Rq["neg_extra.all"][0], drinks_logged=Rq["drinks.all"][0], drinks_as_drink=Rq["drinks.all"][1],
                          uec_recall=Rq["uec"]["recall"], food101_logged=Rq["food101"]["logged_mappable"], tw_logged=Rq["tw"]["logged_any"])); print(curve[-1], flush=True)
    curve2 = []
    for pf in [0.0, 0.02, 0.05, 0.1, 0.15]:
        Rq = evaluate(P_with(**{**flat(P), "p_food_min": pf}))
        curve2.append(dict(p_food_min=pf, neg_user=Rq["neg_user.all"][0], neg_extra=Rq["neg_extra.all"][0], drinks_logged=Rq["drinks.all"][0], uec_recall=Rq["uec"]["recall"],
                           food101_logged=Rq["food101"]["logged_mappable"], tw_logged=Rq["tw"]["logged_any"])); print(curve2[-1], flush=True)
    alt = {}
    for name, kw in [("contain_off(1.01),iou0.6", dict(contain=1.01, iou=0.6)), ("no_neg_classes(cs=A2)", dict(cs="A2")), ("fallback_never", dict(fallback="never"))]:
        EXACT_IOU = kw.get("iou", P["iou"]); Rq = evaluate(P_with(**{**flat(P), **kw})); alt[name] = Rq; print(name, brief(Rq), flush=True)
    EXACT_IOU = P["iou"]
    sure = next((lo for lo, hi, n, a in cal if a is not None and a >= 0.7 and n >= 8), 0.5)
    P["mix"]["sure"] = sure
    json.dump(dict(P=P, results=R, calibration=cal, gate_curve=curve, food_min_curve=curve2, alternatives=alt), open(HERE / "results.json", "w"), default=float, indent=1)
    cl = [dict(name=n, food=True) for n in A2] + [dict(name=n, food=False) for n in N3]
    (HERE / "locator_classes.json").write_text(json.dumps(dict(imgsz=IMGSZ, classes=cl, conf=P["conf"], iou=P["iou"], contain=P["contain"], max_det=P["max_det"], crop_pad=P["pad"],
        fallback_whole=dict(when="no_boxes", p_neg_max=P["fb_neg"], p_food_min=P["fb_food"])), indent=1), encoding="utf-8")
    cp = MODELS / "clip.json"; c = json.loads(cp.read_text(encoding="utf-8"))
    c.update(resize=P["mode"], crop="center", template="a photo of {}.", templates=list(FOOD_TEMPLATES["t1"]), neg_prompts="custom",
             gate=dict(p_neg_max=P["gate"]["p_neg_max"], p_food_min=P["gate"]["p_food_min"]), mix=P["mix"])
    cp.write_text(json.dumps(c, indent=1) + chr(10), encoding="utf-8")
    print("wrote", HERE / "results.json", "sure =", sure)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "embed": stage_embed()
    elif st == "text": stage_text()
    elif st == "sweep": sweep()
    elif st == "final": final()
    elif st == "probe":
        P = P_with(); print(brief(evaluate(P)))
