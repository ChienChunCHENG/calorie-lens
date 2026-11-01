"""Commonness prior from NHANES/WWEIA 2021-2023 (DR1IFF_L + DR2IFF_L individual-food files, public domain) + hand lists.
Everything is aligned to web/public/models/vocab.json rows (2,278 foods then 44 negative rows)."""
import json, os, re, sys, urllib.request
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODELS, SP

HERE = Path(__file__).resolve().parent
WORK = SP / "research/prior"          # scratch: nhanes xpt/pkl, members.json
NH_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{}.xpt"

DEFAULT_COUNT = 4.0   # non-FNDDS rows (Taiwan FDA / estimated): not US-common -> about the 25th percentile of FNDDS rows (median FNDDS row = 10 reports)
TIERS = {"A": 800.0, "B": 250.0, "C": 80.0}   # pseudo report-counts: A ~ top 3% of FNDDS rows, B ~ top 8%, C ~ top 20%

# --- non-FNDDS rows that ARE common in the US/EU (tier of pseudo-count). Everything else non-FNDDS keeps DEFAULT_COUNT. ---
HAND_A = """apple banana orange strawberries blueberries grapes tomato potato broccoli carrot cucumber onion spinach butter honey whole_milk bacon ketchup
hamburger_bun croissant beer red_wine white_wine cola avocado potato_chips vanilla_ice_cream iced_americano latte_coffee cappuccino_coffee lemon watermelon
pineapple roast_chicken waffle cooked_white_rice sweet_corn bell_pepper peanuts almonds mango pear peach""".split()
HAND_B = """garlic cherries kiwi_fruit cantaloupe_melon zucchini button_mushroom iceberg_lettuce romaine_lettuce green_beans green_peas cabbage cauliflower asparagus celery
eggplant pumpkin raisins fig pomegranate cashews walnuts pistachios low_fat_milk plain_yogurt full_fat_plain_yogurt sliced_cheese olive_oil milk_chocolate dark_chocolate
chocolate_ice_cream rolled_oats boiled_egg fried_egg pan_fried_egg scrambled_egg_tw egg_raw chicken_breast_raw chicken_wing_raw sirloin_steak_raw filet_mignon_steak_raw pork_ribs_raw
atlantic_salmon_raw white_shrimp_raw sponge_cake tiramisu black_forest_cake cream_puff dinner_roll honey_glazed_donut macaroni egg_noodles hot_dog_sausage
sushi_nigiri sushi_roll_maki ramen_shoyu ramen_miso ramen_tonkotsu pho pad_thai hummus falafel kebab_doner naan samosa miso_soup kimchi gyoza curry_rice chicken_curry_indian
tonkatsu edamame matcha_latte mocha_coffee caramel_macchiato energy_drink sports_drink coconut_water spring_roll_fried tempura_shrimp biryani bulgogi soy_sauce
fresh_spring_roll_viet banh_mi sashimi onigiri udon_soup instant_noodles_cooked ramen_noodles_cooked hot_pot_mixed chicken_cutlet_tw salt_pepper_chicken fried_chicken_drumstick
karaage korean_fried_chicken""".split()
HAND_C = """whisky sake soju black_tea_unsweet green_tea_unsweet oolong_tea jasmine_tea ginger_tea lemon_black_tea cocoa_milk chocolate_syrup marshmallow gummy_candy soda_crackers
chestnuts taro roasted_sweet_potato steamed_sweet_potato yellow_sweet_potato red_sweet_potato dried_mango stir_fried_greens stir_fried_noodles egg_fried_rice shrimp_fried_rice
kimchi_fried_rice yaki_udon zaru_soba soba_noodles firm_tofu soft_tofu mapo_tofu kung_pao_chicken sweet_sour_pork lo_mein dan_bing egg_tart_portuguese pineapple_bun mooncake
dorayaki takoyaki okonomiyaki yakitori tamagoyaki natto grilled_mackerel grilled_eel nasi_goreng laksa tom_yum_soup green_curry_chicken satay_chicken beef_rendang
cheese_cake_tw ham_egg_sandwich_tw spaghetti_bolognese_tw bubble_milk_tea milk_tea_tw mango_smoothie watermelon_juice sugarcane_juice lemon_juice_tw""".split()

# --- European (and general Western) commonness boost: regex on the vocab `en` field -> floor of TIERS["B"] reports. ---
EURO = [  # (label, regex on en)
    ("croissant", r"^croissant"), ("baguette / french bread", r"french or vienna bread|baguette"), ("schnitzel", r"schnitzel|veal cutlet"), ("paella", r"paella"),
    ("risotto", r"risotto"), ("bratwurst", r"bratwurst"), ("fish and chips", r"fish and chips|fish & chips"), ("pierogi", r"pierogi"), ("moussaka", r"moussaka"),
    ("quiche", r"quiche"), ("crepe", r"\bcrepe"), ("muesli", r"muesli"), ("lasagna", r"lasagna"), ("ravioli", r"ravioli"), ("tortellini", r"tortellini"),
    ("gnocchi", r"gnocchi"), ("carbonara", r"carbonara"), ("focaccia", r"focaccia"), ("ciabatta", r"ciabatta"), ("pretzel (soft)", r"soft pretzel"),
    ("sauerkraut", r"^sauerkraut"), ("goulash", r"goulash"), ("cassoulet", r"cassoulet"), ("ratatouille", r"ratatouille"), ("gazpacho", r"gazpacho"),
    ("gyro / souvlaki", r"gyro|souvlaki"), ("hummus", r"^hummus"), ("falafel", r"^falafel$"), ("couscous", r"couscous"), ("tabbouleh", r"tabbouleh"), ("kebab", r"kebab"),
    ("shepherd's pie", r"shepherd"), ("scone", r"^scone"), ("baked beans", r"^baked beans$"), ("porridge / oatmeal", r"^oatmeal$"), ("danish pastry", r"danish pastry"),
    ("strudel", r"apple strudel"), ("brioche", r"brioche"), ("eclair", r"eclair"), ("tiramisu", r"tiramisu"), ("panna cotta", r"panna cotta"), ("gelato", r"gelato"),
    ("churros", r"churros"), ("empanada", r"^empanada|beef empanada|chicken empanada"), ("bruschetta", r"bruschetta"), ("antipasto", r"antipasto"), ("prosciutto", r"prosciutto"),
    ("salami", r"^salami"), ("chorizo", r"chorizo"), ("brie / camembert", r"brie cheese|camembert"), ("feta", r"feta cheese"), ("mozzarella", r"^mozzarella cheese"),
    ("gouda", r"gouda"), ("meatballs", r"^meatballs with sauce|swedish meatballs"), ("beef wellington", r"beef wellington"), ("mussels", r"^mussels"), ("calamari", r"fried calamari"),
    ("sardines", r"^sardines"), ("herring", r"herring"), ("borscht", r"borscht"), ("minestrone", r"minestrone"), ("french toast", r"^french toast"), ("pancakes", r"^pancakes"),
    ("waffle", r"^waffle$"), ("spaghetti / pasta with meat sauce", r"spaghetti|pasta with tomato.*meat|pasta with meat"), ("smoked salmon", r"smoked salmon|lox"),
    ("roast beef / pot roast", r"pot roast beef"), ("sausage roll / pigs in blanket", r"sausage roll|pigs in a blanket"), ("cottage pie", r"cottage pie"),
    ("yorkshire pudding", r"yorkshire"), ("tzatziki", r"tzatziki"), ("caesar salad", r"caesar"), ("mayonnaise", r"^mayonnaise"), ("steak", r"^beef steak$|^steak$"),
]


def ensure_nhanes():
    WORK.joinpath("nhanes").mkdir(parents=True, exist_ok=True)
    for f in ("DR1IFF_L", "DR2IFF_L"):
        p = WORK / "nhanes" / f"{f}.xpt"
        if not p.exists(): urllib.request.urlretrieve(NH_URL.format(f), p)


def nhanes_counts():
    """Series: FNDDS food code -> number of food-line reports in day1+day2 recalls (unweighted; one line = one reported food)."""
    import pandas as pd
    ensure_nhanes(); parts = []
    for f, c in (("DR1IFF_L", "DR1IFDCD"), ("DR2IFF_L", "DR2IFDCD")):
        pk = WORK / f"{f}.pkl"
        if pk.exists(): df = pd.read_pickle(pk)
        else:
            df = pd.read_sas(WORK / "nhanes" / f"{f}.xpt", format="xport")[[c]].rename(columns={c: "code"}); df["code"] = df["code"].astype("int64"); df.to_pickle(pk)
        parts.append(df.code)
    return pd.concat(parts).value_counts()


def load_vocab():
    return json.load(open(MODELS / "vocab.json", encoding="utf-8"))


def load_raw():
    return json.load(open(SP / "research/r3/vocab.json", encoding="utf-8"))   # same ids/order as web vocab.json food rows (has `source`)


def members():
    p = WORK / "members.json"
    if not p.exists(): raise SystemExit("run build_members.py from research/r3 first (see RESULTS.md)")
    return json.load(open(p))


def row_counts(agg="sum"):
    """per vocab row (foods + negs): (count, kind, euro_labels). kind in fndds|tier|default|neg.
    agg: sum = reports of every FNDDS code merged into the row's cluster; rep = only the representative code; mix = average of the two."""
    V = load_vocab(); R = load_raw(); M = members(); cnt = nhanes_counts()
    n = len(V); c = np.zeros(n); kind = ["neg"] * n; euro = [[] for _ in range(n)]
    hand = {i: "A" for i in HAND_A}; hand.update({i: "B" for i in HAND_B}); hand.update({i: "C" for i in HAND_C})
    for i, v in enumerate(V):
        if v.get("neg"): continue
        assert R[i]["id"] == v["id"]
        m = re.search(r"FNDDS 2021-2023 #(\d+)", R[i]["source"])
        if m:
            mem = [k for k, _ in M[m.group(1)]]
            s = sum(cnt.get(k, 0) for k in mem); r = cnt.get(int(m.group(1)), 0)
            c[i] = s if agg == "sum" else r if agg == "rep" else 0.5 * (s + r)
            kind[i] = "fndds"
        else:
            c[i] = DEFAULT_COUNT; kind[i] = "default"
            if v["id"] in hand: c[i] = max(c[i], TIERS[hand[v["id"]]]); kind[i] = "tier"
        for lab, rx in EURO:
            if re.search(rx, v["en"], re.I):
                euro[i].append(lab); c[i] = max(c[i], TIERS["B"])
    return c, kind, euro


def prior_from_counts(c, neg, alpha=1.0, temper=1.0):
    """prior over FOOD rows only (neg rows get 0 here); normalised to sum 1. temper t: prior^t renormalised."""
    f = ~neg; p = np.zeros_like(c, dtype=np.float64)
    p[f] = (c[f] + alpha) / (c[f].sum() + alpha * f.sum())
    if temper != 1.0:
        p[f] = p[f] ** temper; p[f] /= p[f].sum()
    return p


def logprior_term(p, neg):
    """centred log prior added (times lambda) to the cosine logits: log p_i - mean_{food}(log p); neg rows = 0 (neutral)."""
    lp = np.zeros_like(p); f = ~neg
    lp[f] = np.log(p[f]); lp[f] -= lp[f].mean()
    return lp
