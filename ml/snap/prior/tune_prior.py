"""Tune the commonness prior on the OI-val split only; report OI-val/OI-test/Food-101 and gate checks. Pure numpy over cached embeddings (cache_embeds.py).
usage: python tune_prior.py            (grid + report json to $SP/research/prior/tune.json)"""
import sys, os, json, re, hashlib, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from prior_lib import *
from evalsets import FOOD101_MAP

CACHE = SP / "research/prior/cache"
C = json.load(open(MODELS / "clip.json")); LS = C["logit_scale"]
V = json.load(open(os.environ.get("PRIOR_VOCAB", MODELS / "vocab.json"), encoding="utf-8")); OLD_IDS = {v["id"] for v in load_vocab()}; NEG = np.array([bool(v.get("neg")) for v in V]); N = len(V)
M = np.frombuffer(Path(os.environ.get("PRIOR_EMB", MODELS / "vocab_emb.bin")).read_bytes(), "<f2").astype(np.float32).reshape(N, -1)
EN = [v["en"].lower() for v in V]; ID = [v["id"] for v in V]; CAT = [v["cat"] for v in V]
GATE = (C["gate"]["p_neg_max"], C["gate"]["p_food_min"]); GATE_FB = (0.005, 0.02)


def load(name): return np.load(CACHE / f"{name}.npz")["E"], json.load(open(CACHE / f"{name}.json"))


def mask(rx=None, ids=(), cats=()):
    m = np.zeros(N, bool)
    for i in range(N):
        if NEG[i]: continue
        if ID[i] in ids or CAT[i] in cats or (rx and re.search(rx, EN[i])): m[i] = True
    return m


# --- Open Images class -> accepted vocab rows. STRICT = a specific food; LENIENT = a generic OI category accepted if the row is any plausible member. ---
OI_STRICT = {
    "Hamburger": mask(r"hamburger|cheeseburger|burger"), "Pizza": mask(r"pizza"), "French fries": mask(r"french fries|fries|potato.*fried", ids=("us_home_fries_potato",)),
    "Salad": mask(r"salad"), "Sushi": mask(r"sushi|sashimi|maki|nigiri|onigiri"), "Hot dog": mask(r"hot dog|frankfurter|corn dog"), "Burrito": mask(r"burrito"), "Bagel": mask(r"bagel"),
    "Doughnut": mask(r"doughnut|donut"), "Cake": mask(r"cake|cupcake|brownie"), "Cookie": mask(r"cookie|biscotti"), "Muffin": mask(r"muffin"), "Pretzel": mask(r"pretzel"),
    "Croissant": mask(r"croissant|pastry"), "Egg": mask(r"\begg|omelet"), "Shrimp": mask(r"shrimp|prawn"), "Oyster": mask(r"oyster"), "Coffee (drink)": mask(r"coffee|latte|cappuccino|espresso|americano|mocha"),
    "Tea": mask(r"\btea\b"), "Juice": mask(r"juice"), "Cocktail": mask(r"cocktail|margarita|martini|alcohol|beer|wine|liquor|whisk|vodka|gin\b|rum\b|sangria|mixed drink|daiquiri"),
    "Ice cream": mask(r"ice cream|gelato|frozen yogurt|sundae|sherbet"), "Broccoli": mask(r"broccoli"), "Carrot": mask(r"carrot"), "Cucumber": mask(r"cucumber"),
    "Tomato": mask(r"tomato"), "Apple": mask(r"\bapple"), "Banana": mask(r"banana|plantain"), "Orange (fruit)": mask(r"orange|tangerine|clementine|mandarin|ponkan"), "Grape": mask(r"grape"),
    "Strawberry": mask(r"strawberr"), "Pineapple": mask(r"pineapple"), "Cheese": mask(r"cheese"), "Bread": mask(r"bread|toast|roll\b|bun\b|baguette"), "Sandwich": mask(r"sandwich|burger|sub\b|wrap"),
    "Submarine sandwich": mask(r"sandwich|sub\b"), "Waffle": mask(r"waffle"), "Pancake": mask(r"pancake|crepe"), "Taco": mask(r"taco"), "Popcorn": mask(r"popcorn"), "Candy": mask(r"candy|chocolate|gum\b|lollipop"),
    "Lemon (plant)": mask(r"lemon"), "Mango": mask(r"mango"), "Pumpkin": mask(r"pumpkin|squash"), "Zucchini": mask(r"zucchini|squash"), "Common fig": mask(r"\bfig"), "Coconut": mask(r"coconut"),
    "Cantaloupe": mask(r"cantaloupe|melon"), "Watermelon": mask(r"watermelon"), "Pear": mask(r"\bpear"), "Peach": mask(r"peach|nectarine"), "Pomegranate": mask(r"pomegranate"), "Potato": mask(r"potato"),
    "Pasta": mask(r"pasta|spaghetti|macaroni|noodle|lasagna|ravioli"), "Crab": mask(r"crab"), "Lobster": mask(r"lobster"), "Mushroom": mask(r"mushroom"), "Cabbage": mask(r"cabbage|coleslaw"),
    "Garden Asparagus": mask(r"asparagus"), "Bell pepper": mask(r"pepper"), "Squid": mask(r"squid|calamari"), "Artichoke": mask(r"artichoke"), "Radish": mask(r"radish"), "Peach ": mask(r"peach"),
}
OI_LENIENT = {
    "Pastry": mask(r"pastry|croissant|danish|strudel|turnover|pie\b|tart|eclair|puff|doughnut|donut|bun\b|roll\b|cake|churro|cookie|muffin|scone|brioche|bagel|waffle|pancake|crepe|biscuit|bread"),
    "Baked goods": mask(r"pastry|croissant|danish|strudel|pie\b|tart|cake|cookie|muffin|scone|brioche|bagel|waffle|biscuit|bread|bun\b|roll\b|doughnut|donut|cracker|pretzel|brownie|churro|pita|tortilla|cupcake"),
    "Dessert": mask(r"cake|cookie|pie\b|ice cream|pudding|candy|chocolate|brownie|doughnut|donut|tart|gelato|mousse|cheesecake|sundae|custard|tiramisu|pastry|churro|jello|gelatin|cupcake|sherbet|waffle|crepe|eclair|macaron|frozen yogurt|danish|strudel|cobbler|flan|candy|dessert"),
    "Snack": mask(r"chips|cracker|pretzel|popcorn|nuts|candy|bar\b|cookie|granola|trail mix|jerky|snack|rice cake|seeds|puffs|cheetos|tortilla chip"),
    "Fruit": mask(cats=("fruit",), rx=r"apple|banana|orange|grape|berr|melon|pineapple|mango|peach|pear|cherr|plum|kiwi|lemon|lime|papaya|fig\b|pomegranate|tangerine|clementine|apricot|nectarine|fruit"),
    "Vegetable": mask(cats=("vegetable",)), "Dairy Product": mask(r"cheese|milk|yogurt|cream|butter|kefir|custard|pudding|ice cream"),
    "Fast food": mask(r"burger|fries|pizza|hot dog|nugget|taco|sandwich|chicken tender|chicken wing|nachos|onion ring|burrito|fried chicken|corn dog|chicken strip|quesadilla|wrap|hash brown|milk shake|soft drink|cheesesteak"),
    "Drink": mask(cats=("drink",)), "Food": mask(rx=r"."),
    "Seafood": mask(r"fish|shrimp|crab|lobster|salmon|tuna|oyster|clam|mussel|squid|calamari|scallop|seafood|sushi|sardine|herring|cod|tilapia|trout|prawn"),
    "Fish": mask(r"fish|salmon|tuna|cod|tilapia|trout|sardine|herring|mackerel|sushi|sashimi"), "Shellfish": mask(r"shrimp|crab|lobster|oyster|clam|mussel|scallop|prawn|seafood"),
    "Cream": mask(r"cream|pudding|custard|mousse|whipped|yogurt|ice cream|frosting"), "Tart": mask(r"tart|pie\b|pastry|danish|strudel|cake|flan"),
}
OI_STRICT = {k: v for k, v in OI_STRICT.items() if v.any()}


def split_of(i): return "val" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 == 0 else "test"


def scores(E, lam_term, lam):
    return LS * (E @ M.T) + lam * lam_term[None, :]


def softmax(z): z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def rank_hits(S, accept_masks):
    """S [n,N] scores (food rows only compared). returns top1, top3 hit arrays."""
    Sf = np.where(NEG[None, :], -1e9, S); top3 = np.argsort(-Sf, 1)[:, :3]
    t1 = np.array([accept_masks[k][top3[k, 0]] for k in range(len(S))]); t3 = np.array([accept_masks[k][top3[k]].any() for k in range(len(S))])
    return t1, t3


class Data:
    def __init__(self):
        self.oiE, self.oim = load("oi"); self.f1E, self.f1m = load("food101")
        keep = [k for k, x in enumerate(self.oim) if x["cls"] in OI_STRICT or x["cls"] in OI_LENIENT]
        self.oi_idx = np.array(keep); self.oi_acc = [(OI_STRICT.get(self.oim[k]["cls"]) if self.oim[k]["cls"] in OI_STRICT else OI_LENIENT[self.oim[k]["cls"]]) for k in keep]
        self.oi_strict = np.array([self.oim[k]["cls"] in OI_STRICT for k in keep]); self.oi_split = np.array([split_of(self.oim[k]["id"]) for k in keep])
        self.oi_cls = [self.oim[k]["cls"] for k in keep]
        self.f1_acc = []; self.f1_keep = []; self.f1_acc_old = []; self.f1_keep_old = []
        isold = np.array([i in OLD_IDS for i in ID])
        for k, x in enumerate(self.f1m):
            m = f101_mask(x["label"])
            if m is not None and m.any():
                self.f1_keep.append(k); self.f1_acc.append(m)
                if (m & isold).any(): self.f1_keep_old.append(k); self.f1_acc_old.append(m & isold)
        self.f1_keep = np.array(self.f1_keep); self.f1_keep_old = np.array(self.f1_keep_old)
        self.regions = {n: (load(n), json.load(open(CACHE / f"{n}.images.json"))) for n in ("neg_user", "neg_extra", "neg_ux", "drinks", "ux_img")}


F101_EXTRA = {  # Food-101 label -> regex on en (added to evalsets.FOOD101_MAP ids)
    "apple_pie": r"apple pie|apple turnover|apple crisp|apple cobbler", "beef_carpaccio": r"carpaccio|raw beef", "beet_salad": r"beet", "caprese_salad": r"mozzarella|caprese",
    "carrot_cake": r"carrot cake", "crab_cakes": r"crab cake|crab", "creme_brulee": r"creme brulee|custard|flan", "croque_madame": r"ham and cheese|croque|ham sandwich.*cheese|grilled ham",
    "filet_mignon": r"filet mignon|steak|sirloin", "fish_and_chips": r"fish.*(fried|stick|fillet)|fried fish|fish and chips|breaded fish", "french_fries": r"french fries|fries",
    "greek_salad": r"greek|feta", "lobster_bisque": r"bisque|lobster", "lobster_roll_sandwich": r"lobster", "macarons": r"macaron|macaroon", "poutine": r"poutine|fries.*gravy|cheese fries",
    "prime_rib": r"prime rib|roast beef|pot roast|beef steak|ribs|steak", "pulled_pork_sandwich": r"pork sandwich|barbecue.*sandwich|pulled pork|pork.*bun", "red_velvet_cake": r"cake",
    "seaweed_salad": r"seaweed|kelp|nori|salad", "tuna_tartare": r"tuna|tartare", "chicken_wings": r"chicken wing", "risotto": r"risotto", "escargots": r"snail|escargot",
}


def f101_mask(label):
    ids = FOOD101_MAP.get(label, [])
    m = np.zeros(N, bool)
    for i, v in enumerate(ID):
        if v in ids: m[i] = True
    if label in F101_EXTRA:
        for i in range(N):
            if not NEG[i] and re.search(F101_EXTRA[label], EN[i]): m[i] = True
        if label == "french_fries": m[ID.index("us_home_fries_potato")] = True
    return m if m.any() else None


def evaluate(D, term, lam, gate_mode="raw", want_gate=True):
    r = {}
    S = scores(D.oiE[D.oi_idx], term, lam); t1, t3 = rank_hits(S, D.oi_acc)
    for sp in ("val", "test"):
        for nm, mk in (("all", np.ones(len(t1), bool)), ("strict", D.oi_strict)):
            m = (D.oi_split == sp) & mk
            r[f"oi_{sp}_{nm}"] = (round(float(t1[m].mean()), 4), round(float(t3[m].mean()), 4), int(m.sum()))
    S = scores(D.f1E[D.f1_keep], term, lam); t1, t3 = rank_hits(S, D.f1_acc); r["food101"] = (round(float(t1.mean()), 4), round(float(t3.mean()), 4), len(t1))
    S = scores(D.f1E[D.f1_keep_old], term, lam); t1, t3 = rank_hits(S, D.f1_acc_old); r["food101_old"] = (round(float(t1.mean()), 4), round(float(t3.mean()), 4), len(t1))
    if want_gate:
        for n, ((E, meta), imgs) in D.regions.items():
            if n == "ux_img": continue
            S = scores(E, term, lam); S0 = scores(E, term, 0.0)
            Pg = softmax(S0 if gate_mode == "raw" else S)
            logged = set()
            for k, x in enumerate(meta):
                g = GATE_FB if x["kind"] == "fallback" else GATE
                pf = np.where(NEG, 0, Pg[k])
                if Pg[k][NEG].sum() <= g[0] and pf.max() >= g[1]: logged.add(x["path"])
            r[f"gate_{n}"] = (len(logged), len(imgs))
    return r


def main():
    D = Data(); print("OI crops used", len(D.oi_idx), "strict", int(D.oi_strict.sum()), "f101 imgs", len(D.f1_keep), "classes", len({D.f1m[k]['label'] for k in D.f1_keep}))
    base = evaluate(D, np.zeros(N), 0.0); print("BASE", json.dumps(base))
    res = []
    for agg in ("sum", "rep", "mix"):
        c, kind, _ = row_counts(agg)
        for alpha in (1.0, 5.0):
            p = prior_from_counts(c, NEG, alpha, 1.0); term = logprior_term(p, NEG)
            for lam in (0.5, 1, 1.5, 2, 3, 4, 6, 8):
                for gm in ("raw", "prior"):
                    r = evaluate(D, term, lam, gm); r.update(agg=agg, alpha=alpha, lam=lam, gate=gm); res.append(r)
                    print(agg, alpha, lam, gm, r["oi_val_all"][:2], r["oi_val_strict"][:2], r["gate_neg_user"], r["gate_neg_ux"], r["gate_neg_extra"], r["gate_drinks"], flush=True)
    json.dump(dict(base=base, grid=res), open(SP / "research/prior/tune.json", "w"))


if __name__ == "__main__":
    main()
