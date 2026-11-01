"""Portion-from-photo reference implementation: the plate/bowl is the ruler.

Input  : the shipped snap pipeline items (box, mask, candidates [(vocab idx, p)]) + container candidates, from feat.py.
Output : grams / kcal per item and per dish (samples for p10/p50/p90).

Steps (what the browser has to port, see SPEC.md):
  1. pick_container()      -> ellipse (cx, cy, major px, minor px, angle) + type ("plate"|"bowl"|"tray") from the locator's container mask
  2. cm_per_px             = D_type_cm / major_px           (D_type from portion_params.json or a user setting "my plate is X cm")
  3. food_pixels()         -> boolean mask inside the ellipse of pixels that differ from the plate colour (Lab distance > T)
  4. item areas            -> every food pixel goes to the item whose box contains it (smallest box wins), else to the nearest box
     area_cm2 = n_px * cm_per_px^2 / tilt      with tilt = clip(minor/major, 0.5, 1) (foreshortening of an obliquely seen circle)
  5. grams                 = k_g * area_cm2^beta_g          g = food group of the candidate (group_of)
  6. kcal                  = sum over candidates p_c * grams_c * kcal100_c / 100 ; uncertainty by Monte Carlo (sample_dish)
"""
import json, re, math
import numpy as np, cv2

GROUPS = ["veg", "fruit", "starch", "protein", "fried", "sweet", "liquid", "mixed"]
CAT_GROUP = dict(vegetable="veg", fruit="fruit", rice="starch", noodle="starch", bread="starch", staple="starch", meat="protein", seafood="protein",
                 dessert="sweet", snack="sweet", soup="liquid", drink="liquid", sauce="liquid", dish="mixed", sandwich="mixed")
FRIED_RE = re.compile(r"\b(fried|deep[- ]fried|tempura|battered|fritters?|fries|crispy|breaded|nuggets?|tonkatsu|cutlet|schnitzel|croquettes?|pakora)\b", re.I)
NUT_RE = re.compile(r"\b(almonds?|walnuts?|peanuts?|cashews?|pistachios?|hazelnuts?|pecans?|nuts?|seeds?|trail mix|dried)\b", re.I)


def group_of(v):
    """vocab row -> group. Order matters: nuts/seeds/dried go to 'sweet' (dense snack), fried/battered name beats the category."""
    en = v.get("en", "")
    if NUT_RE.search(en): return "sweet"
    if FRIED_RE.search(en) and v["cat"] not in ("drink", "soup", "sauce"): return "fried"
    return CAT_GROUP.get(v["cat"], "mixed")


def unpack(mask, shape):
    return np.unpackbits(mask)[:shape[0] * shape[1]].reshape(shape).astype(bool)


# ---- 1. container ------------------------------------------------------------------------------------------------------------------
CONT_TYPE = {14: "plate", 15: "bowl", 16: "tray", 1: "plate", 2: "bowl", 3: "plate"}


def fit_ellipse(mask, half=2):
    """major/minor axes (px of the full-res image) of the biggest blob of the mask via cv2.fitEllipse."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 8 or cv2.contourArea(c) < 100: return None
    (cx, cy), (a, b), ang = cv2.fitEllipse(c)
    return dict(cx=cx * half, cy=cy * half, major=max(a, b) * half, minor=min(a, b) * half, angle=ang)


def containers_of(F, min_score=0.15, min_frac=0.12, dish_weight=0.8, max_n=4):
    """Valid container candidates (score >= min_score and at least min_frac of the image's short side wide), best first, each with its ellipse.
    Score = detector score of prompts plate/bowl/tray (rows 14-16) or of the shipped dish/bowl/plate-of-food rows (1-3, times dish_weight)."""
    S = min(F["W"], F["H"]); out = []
    for c in F["containers"]:
        w, h = c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]
        s = c["score"] * (1.0 if c["cls"] >= 14 else dish_weight)
        if s < min_score or max(w, h) < min_frac * S: continue
        e = fit_ellipse(unpack(c["mask"], c["mshape"])) if c["mask"] is not None else None
        if e is None:  # box fallback
            x1, y1, x2, y2 = c["box"]; e = dict(cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, major=max(x2 - x1, y2 - y1), minor=min(x2 - x1, y2 - y1), angle=0)
        e["type"] = CONT_TYPE[c["cls"]]; e["score"] = s; e["box"] = c["box"]; out.append(e)
    out.sort(key=lambda e: -e["score"])
    return out[:max_n]


def container_for_item(item_box, conts, min_inside=0.5):
    """the container (best score first) that holds >= min_inside of the item's box; if none does, None (item falls back to the typical portion)."""
    best = None
    for j, e in enumerate(conts):
        f = inside_frac(item_box, e["box"])
        if f >= min_inside and (best is None or f > best[0] + 0.25 or (abs(f - best[0]) <= 0.25 and e["score"] > conts[best[1]]["score"])): best = (f, j)
    return None if best is None else best[1]


def inside_frac(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); ar = (a[2] - a[0]) * (a[3] - a[1])
    return i / ar if ar > 0 else 0


# ---- 3. food pixels ----------------------------------------------------------------------------------------------------------------
def ellipse_mask(shape, e, shrink=0.97):
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, ((e["cx"], e["cy"]), (e["major"] * shrink, e["minor"] * shrink), e["angle"]), 1, -1)
    return m.astype(bool)


def food_pixels(img_rgb, e, thr=18.0, ring=(0.80, 0.97)):
    """Pixels inside the container ellipse whose Lab distance to the plate colour exceeds thr. The plate colour is the peak (histogram,
    bins 12/8/8 in L/a/b) of the Lab colours in the outer ring 0.70-0.97 of the ellipse, averaged over the peak bin."""
    H, W = img_rgb.shape[:2]
    inner = ellipse_mask((H, W), e, ring[0]); outer = ellipse_mask((H, W), e, ring[1]); rg = outer & ~inner
    lab = cv2.cvtColor(cv2.GaussianBlur(img_rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255.0; lab[..., 1:] -= 128
    px = lab[rg]
    if len(px) < 50: return np.zeros((H, W), bool), None
    q = np.floor(px / np.array([12.0, 8.0, 8.0])).astype(int)
    key = q[:, 0] * 10000 + (q[:, 1] + 50) * 100 + (q[:, 2] + 50)
    u, cnt = np.unique(key, return_counts=True); k0 = u[cnt.argmax()]
    sel = key == k0
    ref = px[sel].mean(0) if sel.sum() >= 20 else np.median(px, 0)
    d = np.sqrt(((lab - ref) ** 2).sum(-1))
    fg = (d > thr) & outer
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return fg, ref


# ---- 4. area per item --------------------------------------------------------------------------------------------------------------
def assign_to_boxes(fg, boxes):
    """count fg pixels per box: contained -> smallest containing box, else nearest box."""
    ys, xs = np.nonzero(fg)
    n = len(boxes); counts = np.zeros(n)
    if len(xs) == 0 or n == 0: return counts
    B = np.array(boxes, float); area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    dx = np.maximum(np.maximum(B[None, :, 0] - xs[:, None], xs[:, None] - B[None, :, 2]), 0)
    dy = np.maximum(np.maximum(B[None, :, 1] - ys[:, None], ys[:, None] - B[None, :, 3]), 0)
    dist = np.hypot(dx, dy) + area[None, :] * 1e-9  # inside -> 0, ties -> smaller box
    return np.bincount(dist.argmin(1), minlength=n).astype(float)


def analyze(img_rgb, F, thr=18.0):
    """-> dict(item_ell=[ellipse or None per item], area_px=[food px per item], food_px). Each item uses the container that holds its box; the food
    pixels of a container are shared among the items in it (contained -> smallest box, else nearest box).
    (Instance-mask and box-based areas were tried and were much worse on Nutrition5k, see RESULTS.md.)"""
    conts = containers_of(F); items = F["items"]
    out = dict(item_ell=[None] * len(items), area_px=[0.0] * len(items), food_px=0)
    owner = [container_for_item(it["box"], conts) for it in items]
    for j, e in enumerate(conts):
        idx = [i for i, o in enumerate(owner) if o == j]
        if not idx: continue
        fg, _ = food_pixels(img_rgb, e, thr)
        cnt = assign_to_boxes(fg, [items[i]["box"] for i in idx]); out["food_px"] += int(fg.sum())
        for i, c in zip(idx, cnt): out["area_px"][i] = float(c); out["item_ell"][i] = e
    return out


def area_cm2(px, e, diam_cm):
    s = diam_cm / e["major"]
    tilt = min(1.0, max(0.5, e["minor"] / e["major"]))
    return px * s * s / tilt


# ---- 5/6. grams and kcal -----------------------------------------------------------------------------------------------------------
class Params:
    def __init__(self, d): self.d = d

    @classmethod
    def load(cls, path): return cls(json.load(open(path)))

    def diam(self, ctype): return self.d["diam_cm"].get(ctype, self.d["diam_cm"]["plate"])

    def density(self, kcal100):
        """kcal per gram actually used: shrink the identified food's density toward the dataset-wide RHO0 (gamma<1 = distrust the name)."""
        gm = self.d.get("gamma", 1.0)
        return max(kcal100 / 100.0, 0.02) ** gm * self.d.get("rho0", 1.2) ** (1 - gm)

    def grams(self, g, a_cm2):
        k, b = self.d["groups"][g]["k"], self.d["groups"][g]["beta"]
        return k * max(a_cm2, 1e-3) ** b


def typical_kcal(V, it):  # shipped behaviour: typical portion, mixture mean of medians
    return sum(w * V[i]["portion_g"][1] * V[i]["kcal100"] / 100.0 for i, w in it["cand"])


def sample_dish(V, grp, P, items, areas_cm2, n=4000, seed=1234, sigma_item=None, sigma_dish=None):
    """Monte-Carlo kcal samples (n,) for the dish. An item whose area is None (no container) uses the vocab's typical portion and portion sigma.
    sigma_item: log-sd of grams per item (independent); sigma_dish: log-sd of one shared scale factor (plate-diameter error)."""
    rng = np.random.default_rng(seed)
    si = P.d["sigma_item"] if sigma_item is None else sigma_item
    sd = P.d["sigma_dish"] if sigma_dish is None else sigma_dish
    tot = np.zeros(n); common = np.exp(sd * rng.standard_normal(n))
    for it, a in zip(items, areas_cm2):
        pr = np.array([w for _, w in it["cand"]]); pr /= pr.sum()
        c = rng.choice(len(pr), size=n, p=pr); z = rng.standard_normal(n); s = np.zeros(n)
        for k, (i, _) in enumerate(it["cand"]):
            m = c == k
            if a is None:
                lo, med, hi = V[i]["portion_g"]; sg = math.log(hi / lo) / 3.29 if lo > 0 and hi > lo else 0.0
                g = med * np.exp(sg * z[m] * P.d.get("fallback_sigma_mult", 1.0))
                s[m] = g * V[i]["kcal100"] / 100.0
            else:
                g = P.grams(grp[i], a) * np.exp(si * z[m]) * common[m]
                s[m] = g * P.density(V[i]["kcal100"])
        tot += s
    return tot
