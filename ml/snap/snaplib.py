"""Core of the snap pipeline in Python (mirror of spec/snap_contract.md): vocab/text embeddings, preprocessing, locator decode, mixture."""
import json, math
import numpy as np
from PIL import Image, ImageOps
from common import MODELS, SP, ROOT


# --- negative rows: (id, label, [full-sentence prompts]) -----------------------------------------------------------------------
def _n(i, en, *ps):
    return dict(id="neg_" + i, en=en, prompts=list(ps) or [f"a photo of {en}."])


NEGS = [
    # empty / non-food containers
    _n("empty_cup", "empty cup", "a photo of an empty cup.", "a cup with nothing in it.", "an empty coffee cup, no drink inside."),
    _n("empty_glass", "empty glass", "a photo of an empty glass.", "a glass with nothing in it.", "an empty drinking glass."),
    _n("empty_paper_cup", "empty paper cup", "a photo of an empty paper cup.", "a stack of empty paper cups.", "disposable paper cups with nothing in them."),
    _n("empty_plastic_cup", "empty plastic cup", "a photo of an empty plastic cup.", "an empty clear plastic cup with a lid and straw."),
    _n("empty_mug", "empty mug", "a photo of an empty mug.", "an empty ceramic mug."),
    _n("empty_tumbler", "empty tumbler", "a photo of an empty tumbler.", "an empty stainless steel tumbler.", "an empty thermos bottle."),
    _n("empty_wine_glass", "empty wine glass", "a photo of an empty wine glass."),
    _n("glass_residue", "glass with a little residue", "a glass with only a few drops of liquid left.", "a dirty empty glass."),
    _n("water_glass", "glass of water", "a photo of a glass of plain water.", "a clear glass of water."),
    _n("water_bottle", "water bottle", "a photo of a water bottle.", "a plastic bottle of water.", "an empty plastic bottle."),
    _n("empty_bowl", "empty bowl", "a photo of an empty bowl.", "a bowl with nothing in it.", "an empty white ceramic bowl."),
    _n("empty_plate", "empty plate", "a photo of an empty plate.", "a clean empty dish.", "an empty white plate on a table."),
    _n("empty_tray", "empty tray", "a photo of an empty tray.", "an empty lunch box.", "an empty food container."),
    _n("dirty_plate", "plate with leftovers wiped clean", "a finished plate with no food left.", "an empty bowl after the meal."),
    _n("glass_jar", "empty jar", "a photo of an empty glass jar."),
    # cookware
    _n("empty_wok", "empty wok", "a photo of an empty wok.", "a wok with nothing in it.", "an empty black frying pan."),
    _n("frying_pan", "frying pan", "a photo of a frying pan on a stove.", "an empty non-stick pan."),
    _n("cooking_pot", "cooking pot", "a photo of an empty cooking pot.", "an empty saucepan.", "a stock pot with a lid."),
    _n("stove", "stove", "a photo of a gas stove.", "a stove top burner."),
    _n("kettle", "kettle", "a photo of an electric kettle.", "a rice cooker."),
    _n("baking_tray", "baking tray", "a photo of an empty baking tray.", "an empty oven rack."),
    # utensils
    _n("chopsticks", "chopsticks", "a photo of chopsticks.", "a pair of chopsticks on a table."),
    _n("cutlery", "knife fork and spoon", "a photo of a spoon and fork.", "cutlery on a table."),
    _n("utensil", "ladle or tongs", "a photo of a kitchen ladle.", "a kitchen utensil."),
    _n("cutting_board", "cutting board", "a photo of an empty cutting board.", "a kitchen knife."),
    # surfaces / scenes
    _n("table", "table", "a photo of an empty table.", "a wooden table top.", "a table with nothing on it."),
    _n("counter", "kitchen counter", "a photo of a kitchen counter.", "a marble countertop.", "an empty kitchen worktop."),
    _n("tablecloth", "tablecloth", "a photo of a tablecloth.", "a white paper on a table."),
    _n("sink", "sink", "a photo of a kitchen sink.", "a dish rack."),
    _n("room", "room interior", "a photo of a room interior.", "a wall.", "the floor."),
    _n("fridge", "refrigerator", "a photo of the inside of an empty refrigerator.", "a kitchen cabinet."),
    _n("blur", "blurry photo", "a blurry out of focus photo.", "a dark photo.", "a photo of nothing in particular."),
    # people
    _n("hand", "hand", "a photo of a human hand.", "a hand holding a phone.", "fingers."),
    _n("person", "person", "a photo of a person.", "a face.", "a selfie."),
    # packaging / text / objects
    _n("packaging", "food packaging", "a photo of empty food packaging.", "an empty snack wrapper.", "a cardboard box."),
    _n("bag", "bag", "a photo of a paper bag.", "a plastic bag.", "a shopping bag."),
    _n("can", "empty can", "a photo of an empty can.", "a crushed aluminium can.", "a bottle cap."),
    _n("menu", "menu or text", "a photo of a menu.", "a screenshot of text.", "a receipt."),
    _n("phone", "phone screen", "a screenshot of a smartphone app.", "a smartphone.", "a laptop computer."),
    _n("napkin", "napkin", "a photo of a napkin.", "a paper tissue.", "a towel."),
    _n("object", "household object", "a photo of a household object.", "a book.", "a keyboard.", "a toy."),
    _n("plant", "plant", "a photo of a plant.", "a flower vase.", "a potted plant."),
    _n("pet", "pet", "a photo of a cat.", "a photo of a dog."),
    _n("outside", "street scene", "a photo of a street.", "a photo of a building.", "a car."),
]

FOOD_TEMPLATES = {
    "t1": ["a photo of {}."],
    "t2": ["a photo of {}, a type of food."],
    "t3": ["a photo of {}.", "a photo of {}, a type of food."],
    "t4": ["a photo of {}.", "a photo of a plate of {}.", "a close-up photo of {}."],
}


def load_foods():
    v = json.load(open(SP / "research/r3/vocab.json", encoding="utf-8"))
    out = []
    for x in v:
        lo, hi = x["portion_range_g"]; g = x["portion"]["g"]
        out.append(dict(id=x["id"], en=x["en"], zh=x.get("zh"), kcal100=x["kcal_per_100g"], portion_g=[lo, g, hi], cat=x["category"]))
    return out


def build_vocab():
    foods = load_foods()
    negs = [dict(id=n["id"], en=n["en"], kcal100=0, portion_g=[0, 0, 0], cat="neg", neg=True) for n in NEGS]
    return foods + negs


def encode_texts(model, tok, texts, bs=256):
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            e = model.encode_text(tok(texts[i:i + bs]))
            outs.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(outs).numpy()


def text_matrix(model, tok, foods, food_tpl, neg_tpl="custom", cache=None):
    """rows: foods (ensemble over food_tpl, normalised) then NEGS (ensemble over neg prompts, normalised)."""
    key = (tuple(food_tpl), neg_tpl)
    if cache is not None and key in cache: return cache[key]

    def ens(texts_per_row):
        flat = [t for ts in texts_per_row for t in ts]
        e = encode_texts(model, tok, flat); k = 0; rows = []
        for ts in texts_per_row:
            m = e[k:k + len(ts)].mean(0); k += len(ts); rows.append(m / np.linalg.norm(m))
        return np.stack(rows)

    F = ens([[t.format(f["en"]) for t in food_tpl] for f in foods])
    if neg_tpl == "custom": nt = [n["prompts"] for n in NEGS]
    elif neg_tpl == "simple": nt = [[f"a photo of {n['en']}."] for n in NEGS]
    else: nt = [[n["prompts"][0]] for n in NEGS]
    N = ens(nt)
    M = np.concatenate([F, N]).astype(np.float32)
    if cache is not None: cache[key] = M
    return M


# --- image helpers ----------------------------------------------------------------------------------------------------------------
def load_img(path, long_edge=1280):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = max(im.size)
    if s > long_edge:
        r = long_edge / s; im = im.resize((round(im.width * r), round(im.height * r)), Image.BILINEAR)
    return im


def letterbox(im, size):
    r = min(size / im.width, size / im.height)
    nw, nh = round(im.width * r), round(im.height * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (px, py))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, px, py


def clip_prep(im, size=256, resize="shortest", crop="center"):
    if resize == "squash":
        t = im.resize((size, size), Image.BILINEAR)
    else:
        r = size / min(im.size); t = im.resize((max(size, round(im.width * r)), max(size, round(im.height * r))), Image.BILINEAR)
        if crop == "center":
            l, tp = (t.width - size) // 2, (t.height - size) // 2; t = t.crop((l, tp, l + size, tp + size))
    return np.asarray(t, dtype=np.float32).transpose(2, 0, 1) / 255.0


def crop_box(im, box, pad):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    return im.crop((max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h)), min(im.width, int(math.ceil(x2 + pad * w))), min(im.height, int(math.ceil(y2 + pad * h)))))


# --- locator decode / NMS ---------------------------------------------------------------------------------------------------------
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def inside_frac(a, b):  # fraction of a's area inside b
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); ar = (a[2] - a[0]) * (a[3] - a[1])
    return i / ar if ar > 0 else 0


def decode_locator(raw, nc, r, px, py, W, H, conf, iou, contain, max_det, cls_idx=None):
    """raw: [4+nc(+32), N]. cls_idx: optional subset of class rows. Returns [dict(box, score, cls)] in image px after NMS+contain, BEFORE
    the food filter. cls is the index into the *subset*."""
    cs = raw[4:4 + nc]
    if cls_idx is not None: cs = cs[cls_idx]
    best = cs.argmax(0); sc = cs.max(0)
    keep = np.where(sc >= conf)[0]
    keep = keep[np.argsort(-sc[keep])]
    cand = []
    for i in keep:
        cx, cy, w, h = raw[:4, i]
        b = [(cx - w / 2 - px) / r, (cy - h / 2 - py) / r, (cx + w / 2 - px) / r, (cy + h / 2 - py) / r]
        b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
        if b[2] - b[0] < 4 or b[3] - b[1] < 4: continue
        if any(iou_xyxy(b, c["box"]) > iou for c in cand): continue
        cand.append(dict(box=b, score=float(sc[i]), cls=int(best[i])))
        if len(cand) >= 100: break
    out = []
    for c in cand:  # contain: drop a box >= contain inside a higher-scoring kept box
        if any(inside_frac(c["box"], o["box"]) >= contain for o in out): continue
        out.append(c)
    return out[:max_det]


# --- mixture ----------------------------------------------------------------------------------------------------------------------
def sigma_of(lo, hi): return math.log(hi / lo) / 3.29 if lo > 0 and hi > lo else 0.0


def mixture(vocab, probs, idxs, n=4000, seed=1234):
    """kcal samples (n,) for a mixture over candidates idxs with weights probs."""
    rng = np.random.default_rng(seed)
    p = np.asarray(probs, dtype=np.float64); p = p / p.sum()
    c = rng.choice(len(idxs), size=n, p=p)
    z = rng.standard_normal(n)
    out = np.empty(n)
    for k, i in enumerate(idxs):
        v = vocab[i]; lo, med, hi = v["portion_g"]; s = sigma_of(lo, hi)
        m = c == k; out[m] = med * np.exp(s * z[m]) * v["kcal100"] / 100.0
    return out
