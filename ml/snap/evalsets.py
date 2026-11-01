"""Evaluation sets + label mappings used by tune.py (all data lives in the session scratchpad; see ml/snap/README.md)."""
import glob, json, os, re
from common import SP

FOOD101_MAP = {  # Food-101 label -> accepted vocab ids; FIRST id is the canonical entry whose typical-portion kcal is "truth". Labels not listed
    # have no equivalent entry in the 2,278-item vocab and are excluded from name accuracy and kcal checks.
    "beef_tartare": ["us_steak_tartare"], "bibimbap": ["bibimbap", "us_korean_bibimbap"], "bread_pudding": ["us_bread_pudding"],
    "breakfast_burrito": ["us_burrito", "us_beef_burrito_with_rice", "us_pork_burrito_with_rice", "us_meat_burrito_with_rice", "us_chicken_burrito_with_rice", "us_cheese_burrito_with_beans"],
    "bruschetta": ["us_bruschetta"], "ceviche": ["us_ceviche"], "cheesecake": ["cheese_cake_tw", "us_fruit_cheesecake", "us_chocolate_cheesecake"],
    "chocolate_mousse": ["us_mousse"], "churros": ["us_churros"], "club_sandwich": ["us_club_sandwich_on_white", "us_club_sandwich_on_white_with_cheese"],
    "cup_cakes": ["us_cake_or_cupcake"], "deviled_eggs": ["us_deviled_egg"],
    "donuts": ["us_doughnut", "honey_glazed_donut", "us_doughnut_holes", "us_jelly_doughnut", "us_cake_type_doughnut_with_icing", "us_chocolate_doughnut"],
    "falafel": ["falafel"], "french_onion_soup": ["us_french_onion_soup"], "frozen_yogurt": ["us_frozen_yogurt"], "garlic_bread": ["us_garlic_bread"],
    "gnocchi": ["us_potato_gnocchi", "us_cheese_gnocchi"], "grilled_cheese_sandwich": ["us_grilled_cheese_sandwich"],
    "gyoza": ["gyoza", "guo_tie"], "hot_and_sour_soup": ["sour_hot_soup"], "huevos_rancheros": ["us_huevos_rancheros"],
    "lasagna": ["us_lasagna_with_meat", "us_meatless_lasagna_with_vegetables", "us_lasagna_with_chicken_or_turkey"], "miso_soup": ["miso_soup"],
    "oysters": ["oysters"], "pad_thai": ["pad_thai"], "pancakes": ["us_pancakes"],
    "pizza": ["us_pizza", "us_pizza_with_pepperoni", "us_pizza_with_meat_gluten_free_thin_crust"],
    "pork_chop": ["us_pork_chop", "us_stuffed_pork_chop", "pork_chop_rice_bento"], "samosa": ["samosa"],
    "spaghetti_bolognese": ["spaghetti_bolognese_tw"], "spring_rolls": ["spring_roll_fried", "spring_roll_taiwan_peanut"],
    "strawberry_shortcake": ["us_strawberry_shortcake_cake"], "sushi": ["us_sushi", "sushi_nigiri", "sushi_roll_maki", "us_avocado_sushi_roll"],
    "tiramisu": ["tiramisu"], "waffles": ["waffle", "us_fruit_waffle", "us_cinnamon_waffle", "us_chocolate_waffle", "us_gluten_free_waffle"],
    "grilled_salmon": ["us_salmon_fish", "atlantic_salmon_raw"],
}

# Taiwanese photo set: dish -> regex on the vocab entry's English name (lenient semantic match; documented in README).
TW_ACCEPT = {
    "lu_rou_fan": r"lu rou fan|braised pork belly rice|braised pork rice|minced pork.*rice|braised pork.*bento",
    "rice_white": r"\b(white|steamed|cooked|plain) rice\b|^rice$|brown rice|rice, ",
    "chicken_leg": r"drumstick|chicken leg|fried chicken|chicken.*bento|chicken cutlet|ji pai",
    "stirfried_greens": r"water spinach|greens|spinach|bok choy|stir-fried (veg|lettuce|cabbage|pea)|ong choy|lettuce|cabbage|sweet potato leaves|vegetable",
    "beef_noodle_soup": r"beef noodle|braised beef.*noodle|lanzhou",
    "potstickers": r"potsticker|pan-fried (pork )?dumpling|gyoza|guo tie|jiaozi|dumpling",
    "scallion_pancake": r"scallion pancake|green onion pancake|cong you bing|egg scallion pancake",
    "popcorn_chicken": r"popcorn chicken|salt.*pepper chicken|yan su ji|karaage|fried chicken|chicken nugget|chicken bites",
    "bubble_milk_tea": r"bubble|boba|pearl|milk tea",
    "pork_chop": r"pork chop|pork cutlet|tonkatsu|katsu",
    "braised_egg": r"braised .*egg|soy sauce egg|tea egg",
    "braised_tofu": r"braised .*tofu|tofu|dou gan",
    "fried_egg": r"fried egg|pan-fried egg|sunny|omelet|fried eggs",
}


def sets():
    """name -> list of dict(path, label?, ...)"""
    S = {}
    S["neg_user"] = [dict(path=p) for p in sorted(glob.glob(str(SP / "negatives/*.jpg")))]
    S["neg_extra"] = [dict(path=p) for d in ("neg_extra", "neg2") for p in sorted(glob.glob(str(SP / f"eval/{d}/*.jpg")))]
    S["drinks"] = [dict(path=p) for d in ("drinks", "drinks2") for p in sorted(glob.glob(str(SP / f"eval/{d}/*.jpg")))]
    lab = json.load(open(SP / "eval/food101/labels.json"))
    S["food101"] = [dict(path=str(SP / "eval/food101" / i["file"]), label=i["label"]) for i in lab["images"]]
    S["tw"] = [dict(path=p, label=re.sub(r"_\d+$", "", os.path.splitext(os.path.basename(p))[0])) for p in sorted(glob.glob(str(SP / "foodimg/*.jpg")))]
    gt = json.load(open(SP / "research/r4/gt_multi.json"))
    S["uec"] = [dict(path=str(SP / f"research/r4/data/{k}.jpg"), boxes=v["boxes"]) for k, v in gt.items()]
    return S
