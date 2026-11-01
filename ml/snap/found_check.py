"""Found-rate check for the live lens: runs the REAL built site (window.__clTest hook) on still photos and reports how
often the lens shows the right food, and why it doesn't (no box / rejected as not-food / wrong name).

    cd web && npm run build -- --base=/calorie-lens/ && npx vite preview --base=/calorie-lens/ --port 4178
    python ml/snap/found_check.py --set meals --out found_meals.json [--limit 300] [--times 2]

Photos and truth: scratchpad/uxset/manifest.json (Open Images, CC BY 2.0; main_class = the food that covers the most area).
"""
import argparse, base64, json, os, re, statistics, time
from pathlib import Path
from playwright.sync_api import sync_playwright

SP = Path(os.environ.get("SP", r"C:/Users/chien/AppData/Local/Temp/claude/C--Users-chien-Downloads-calorie-len/134fdf1a-1063-43e8-b87c-b79446e942b0/scratchpad"))

# Open Images class -> regex on the displayed food name (from ml/snap/prior/tune_prior.py OI_STRICT, trimmed).
ACCEPT = {
    "Hamburger": r"hamburger|cheeseburger|burger", "Pizza": r"pizza", "French fries": r"fries|potato.*fried",
    "Salad": r"salad", "Sushi": r"sushi|sashimi|maki|nigiri|onigiri", "Hot dog": r"hot dog|frankfurter|corn dog", "Burrito": r"burrito",
    "Bagel": r"bagel", "Doughnut": r"doughnut|donut", "Cake": r"cake|cupcake|brownie", "Cookie": r"cookie|biscotti", "Muffin": r"muffin",
    "Pretzel": r"pretzel", "Croissant": r"croissant|pastry", "Egg (Food)": r"\begg|omelet", "Shrimp": r"shrimp|prawn", "Oyster": r"oyster",
    "Coffee": r"coffee|latte|cappuccino|espresso|americano|mocha", "Tea": r"\btea\b", "Juice": r"juice", "Ice cream": r"ice cream|gelato|frozen yogurt|sundae",
    "Broccoli": r"broccoli", "Carrot": r"carrot", "Cucumber": r"cucumber", "Tomato": r"tomato", "Apple": r"\bapple", "Banana": r"banana|plantain",
    "Orange": r"orange|tangerine|clementine|mandarin", "Grape": r"grape", "Strawberry": r"strawberr", "Pineapple": r"pineapple", "Cheese": r"cheese",
    "Bread": r"bread|toast|roll\b|bun\b|baguette", "Sandwich": r"sandwich|burger|sub\b|wrap|blt", "Submarine sandwich": r"sandwich|sub\b",
    "Waffle": r"waffle", "Pancake": r"pancake|crepe", "Taco": r"taco", "Popcorn": r"popcorn", "Candy": r"candy|chocolate|lollipop",
    "Lemon": r"lemon", "Mango": r"mango", "Pumpkin": r"pumpkin|squash", "Zucchini": r"zucchini|squash", "Coconut": r"coconut",
    "Cantaloupe": r"cantaloupe|melon", "Watermelon": r"watermelon", "Pear": r"\bpear", "Peach": r"peach|nectarine", "Pomegranate": r"pomegranate",
    "Potato": r"potato", "Pasta": r"pasta|spaghetti|macaroni|noodle|lasagna|ravioli|fettuccine|carbonara", "Crab": r"crab", "Lobster": r"lobster",
    "Mushroom": r"mushroom", "Cabbage": r"cabbage|coleslaw", "Garden Asparagus": r"asparagus", "Bell pepper": r"pepper", "Squid": r"squid|calamari",
    "Artichoke": r"artichoke", "Radish": r"radish", "Wine": r"wine", "Beer": r"beer|ale\b|lager", "Cocktail": r"cocktail|margarita|martini|mojito|daiquiri|sangria",
    "Milk": r"milk", "Grapefruit": r"grapefruit", "Lime": r"\blime", "Honeycomb": r"honey", "Guacamole": r"guacamole|avocado", "Hamburger ": r"burger",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4178/calorie-lens/")
    ap.add_argument("--set", default="meals"); ap.add_argument("--limit", type=int, default=300); ap.add_argument("--times", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = [i for i in json.load(open(SP / "uxset/manifest.json", encoding="utf-8"))["items"] if i["set"] == a.set][: a.limit]
    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_context(viewport={"width": 390, "height": 844}).new_page()
        pg.goto(a.url)
        pg.wait_for_function("window.__clTest", timeout=60_000)
        pg.evaluate("window.__clTest.warm()")
        pg.wait_for_function("window.__clTest.ready()", timeout=300_000)
        for it in man:
            data = base64.b64encode((SP / "uxset" / it["file"]).read_bytes()).decode()
            t0 = time.time()
            r = pg.evaluate("""async ([d, n]) => { const b = await (await fetch('data:image/jpeg;base64,' + d)).blob();
                const r = await window.__clTest.analyze(b, n); return JSON.parse(JSON.stringify(r)); }""", [data, a.times])
            names = [t["name"] for t in r["tracks"]]
            rows.append({"id": it["id"], "cls": it.get("main_class"), "names": names, "kcal": [t["p50"] for t in r["tracks"]],
                         "nBoxes": r.get("nBoxes"), "ms": round((time.time() - t0) * 1000)})
        b.close()
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    score(rows)


def score(rows):
    known = [r for r in rows if r["cls"] in ACCEPT]
    def found(r): return any(re.search(ACCEPT[r["cls"]], n, re.I) for n in r["names"])
    nb = [r for r in known if not r["nBoxes"]]; shown_none = [r for r in known if r["nBoxes"] and not r["names"]]
    wrong = [r for r in known if r["names"] and not found(r)]
    print(f"photos {len(rows)} (scored {len(known)}): right food shown {sum(map(found, known))}/{len(known)} = {100 * sum(map(found, known)) / max(1, len(known)):.0f}%")
    print(f"  no box found {len(nb)} | boxes but nothing shown (gate) {len(shown_none)} | shown but wrong name {len(wrong)}")
    print(f"  any label shown {sum(bool(r['names']) for r in rows)}/{len(rows)}; median ms/photo {statistics.median(r['ms'] for r in rows)}")
    print("  wrong examples:", [(r["cls"], r["names"][:2]) for r in wrong[:8]])


if __name__ == "__main__":
    main()
