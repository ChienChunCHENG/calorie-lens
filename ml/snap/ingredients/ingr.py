"""Reference implementation of the ingredient/component-level calorie estimator (see SPEC.md, RESULTS.md).

Pipeline for one photo:
  1. FastSAM-s class-agnostic masks (imgsz 640) + YOLOE-11s 'plate/bowl/tray' container mask        [cache_seg.run]
  2. MobileCLIP-S1 embedding of each mask (masked crop and context crop)                              [cache_seg.run]
  3. zero-shot label each mask over the component vocabulary (+ background rows)                      [label_items]
  4. paint masks smallest-on-top -> disjoint region per component; area as a fraction of the plate    [areas]
  5. grams = K[group] * frac^b ; kcal = grams * kcal/g (soft over the label distribution) + oil allowance [estimate]
All numbers in `params.json` are fitted on the fit split (fit_params.py); this file only applies them."""
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def load_vocab(path=HERE / "ingredients_vocab.json"):
    return json.load(open(path, encoding="utf-8"))


def text_embs(rows, bg, cache=HERE / "ingredients_vocab_emb.npy"):
    """rows: vocab; bg: background prompts. Returns [len(rows)+len(bg), 512] fp32 L2-normalised ('a photo of {}.' as in clip.json)."""
    if cache.exists():
        e = np.load(cache).astype(np.float32)
        if len(e) == len(rows) + len(bg): return e
    import sys, torch
    sys.path.insert(0, str(HERE.parent))
    from common import load_clip
    m, _, tok = load_clip()
    texts = [f"a photo of {r['en']}." for r in rows] + [f"a photo of {b}." for b in bg]
    with torch.no_grad():
        e = torch.nn.functional.normalize(m.encode_text(tok(texts)), dim=-1).numpy()
    np.save(cache, e.astype(np.float16)); return e


LOGIT_SCALE = 64.25


def label_items(seg, E, nK, mix=0.5, temp=1.0):
    """per-item probability over [K components | BG rows]. Embedding = mix*masked-crop + (1-mix)*context-crop."""
    if not seg["items"]: return np.zeros((0, E.shape[0]))
    ea = np.stack([it["ea"] for it in seg["items"]]); eb = np.stack([it["eb"] for it in seg["items"]])
    e = mix * ea + (1 - mix) * eb; e /= np.linalg.norm(e, axis=1, keepdims=True)
    z = LOGIT_SCALE / temp * (e @ E.T); z -= z.max(1, keepdims=True); p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def plate_mask(seg, shape):
    """container mask at stored resolution (highest-score YOLOE plate/bowl/tray); falls back to the whole frame."""
    h, w = shape
    c = max(seg["cont"], key=lambda c: c["score"]) if seg["cont"] else None
    if c is None: return np.ones((h, w), bool), False
    m = c["mask"] > 0.5
    return (m if m.sum() > 0.02 * h * w else np.ones((h, w), bool)), True


def areas(seg, P, nK, min_in=0.5, max_frac=0.95, norm="plate"):
    """Disjoint paint of item masks (largest first, so the smallest mask wins each pixel). Returns (area_per_item [n] as fraction of the
    plate area, owned pixels). Masks that lie mostly outside the plate or cover (nearly) all of it are dropped."""
    its = seg["items"]
    if not its: return np.zeros(0)
    h, w = its[0]["m"].shape
    plate, found = plate_mask(seg, (h, w)); pa = plate.sum()
    own = -np.ones((h, w), int)
    order = np.argsort([-it["m"].sum() for it in its])
    ok = np.zeros(len(its), bool)
    for i in order:
        m = its[i]["m"] > 0.5
        if m.sum() == 0: continue
        inside = (m & plate).sum() / m.sum()
        if found and inside < min_in: continue
        if m.sum() > max_frac * pa and len(its) > 1: continue
        own[m & plate] = i; ok[i] = True
    den = pa if norm == "plate" else h * w   # 'image' = fixed-camera pixel scale (valid only when the camera never moves, e.g. Nutrition5k)
    return np.array([(own == i).sum() / den for i in range(len(its))])


def component_areas(seg, probs, A, nK, bg_thr=0.5):
    """sum of soft-assigned area per component: area_k = sum_i A_i * P_ik (K food rows only; BG probability removes the area)."""
    if len(A) == 0: return np.zeros(nK)
    return (A[:, None] * probs[:, :nK]).sum(0)


def estimate(comp_area, rows, params, probs=None, A=None):
    """kcal point estimate and per-component breakdown. params: {K:{group:g_per_plate}, b, oil, thr}."""
    K = params["K"]; b = params["b"]; out = {}; tot = 0.0
    for k, r in enumerate(rows):
        a = comp_area[k]
        if a < params["thr"]: continue
        g = K.get(r["group"], K["default"]) * (a ** b)
        out[r["id"]] = (g, g * r["kcal_g"]); tot += g * r["kcal_g"]
    return tot + params["oil"], out


def interval(point, rel):
    """lognormal band; rel = fitted sigma of ln(true/pred) (validation)."""
    return point * math.exp(-1.2816 * rel), point * math.exp(1.2816 * rel)
