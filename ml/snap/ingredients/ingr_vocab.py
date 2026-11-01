"""Ingredient/component vocabulary. Canonical component -> (CLIP prompt text, group, [Nutrition5k ingredient names merged into it], kcal_per_g override or None).
kcal/g for N5k-backed entries = gram-weighted mean over the FIT-split dishes (see build_vocab()), falling back to the ingredients_metadata value.
Entries with n5k=[] are extras (Taiwanese / Asian components and whole mixed dishes) with hand-set kcal/g (USDA-style cooked values)."""
import json, csv, collections

C = {}  # key -> (prompt, group, [n5k names], kcal_g or None)


def a(key, prompt, grp, n5k=(), kg=None): C[key] = (prompt, grp, list(n5k), kg)


# --- staples
a("white_rice", "steamed white rice", "staple", ["white rice", "rice", "country rice", "pilaf", "risotto"])
a("brown_rice", "brown rice or wild rice", "staple", ["brown rice", "wild rice"])
a("fried_rice", "fried rice", "staple", ["fried rice"])
a("noodles", "cooked noodles", "staple", ["noodles", "chow mein", "lo mein", "rice noodles", "spaghetti"])
a("pasta", "cooked pasta", "staple", ["pasta", "macaroni", "orzo", "ravioli", "tortellini", "lasagna", "macaroni and cheese", "gnocchi", "pasta salad"])
a("quinoa", "cooked quinoa", "staple", ["quinoa", "bulgur", "millet", "wheat berry", "couscous", "barley", "polenta", "grits", "tabouli"])
a("potato_roast", "roasted potatoes", "staple", ["roasted potatoes", "potatoes", "red potatoes", "baked potatoes", "potato salad", "mashed potatoes"])
a("hash_browns", "hash browns", "staple", ["hash browns", "french fries", "potato skins"])
a("sweet_potato", "roasted sweet potato", "staple", ["sweet potato", "yam"])
a("corn_cob", "corn on the cob", "staple", ["corn on the cob"])
a("corn_kernels", "corn kernels", "veg", ["corn", "succotash", "hominy"])
a("bread", "slices of bread or toast", "bread", ["bread", "toast", "white bread", "wheat bread", "whole wheat bread", "multigrain bread", "rye bread", "sourdough bread", "potato bread", "raisin bread", "garlic bread", "focaccia", "pita bread", "flatbread", "naan", "challah", "cornbread", "rolls", "buns", "crispbread", "breadsticks", "croutons", "stuffing", "french toast"])
a("bagel", "bagel", "bread", ["bagels", "english muffins"])
a("tortilla", "tortilla or wrap", "bread", ["tortilla", "wraps", "tacos", "taco shells", "burritos", "quesadillas", "tostadas", "enchiladas"])
a("pancake_waffle", "pancakes or waffles", "bread", ["pancakes", "waffles", "crepes"])
a("oatmeal", "bowl of oatmeal", "staple", ["oatmeal", "oats", "cereal", "raisin bran", "muesli", "granola", "cereal bars", "granola bars"])
a("pizza", "slice of pizza", "mixed", ["cheese pizza", "pizza", "pepperoni pizza", "calzones", "pizza dough"])
a("sandwich", "a sandwich", "mixed", ["sandwiches", "hamburgers", "burgers", "cheeseburgers", "hot dogs", "gyros", "veggie burgers"])
# --- proteins
a("chicken", "cooked chicken meat", "protein", ["chicken", "grilled chicken", "chicken breast", "chicken thighs", "roast chicken", "chicken drumsticks", "ground chicken", "turkey", "turkey breast", "roast turkey", "ground turkey", "chicken salad", "duck"])
a("fried_chicken", "fried chicken", "fried", ["fried chicken", "chicken nuggets", "chicken wings"])
a("beef", "cooked beef", "protein", ["beef", "roast beef", "brisket", "ground beef", "meatballs", "meatloaf", "corned beef", "pastrami", "bison", "veal"])
a("steak", "grilled steak", "protein", ["steak"])
a("pork", "cooked pork", "protein", ["pork", "roast pork", "pork chops", "ground pork", "ribs", "lamb", "lamb chops", "ham"])
a("sausage", "sausage", "protein", ["sausage", "chicken apple sausage", "soy sausage", "bologna", "salami"])
a("bacon", "strips of bacon", "protein", ["bacon", "turkey bacon", "pepperoni"])
a("fish", "cooked fish fillet", "seafood", ["fish", "tilapia", "cod", "haddock", "trout", "catfish", "flounder", "snapper", "halibut", "mahi mahi", "swordfish", "artic char", "mackerel", "carp", "eel", "herring", "sardines", "anchovies", "tuna", "tuna salad"])
a("salmon", "salmon fillet", "seafood", ["salmon", "smoked salmon"])
a("shrimp", "shrimp or seafood", "seafood", ["shrimp", "seafood", "crab", "lobster", "scallops", "mussels", "clams", "oysters", "squid", "calamari", "octopus", "crawfish", "sushi"])
a("scrambled_eggs", "scrambled eggs", "egg", ["scrambled eggs", "omelets", "quiche", "souffle", "egg salad"])
a("egg_whites", "egg whites or cooked eggs", "egg", ["egg whites", "eggs", "poached eggs", "hard boiled eggs", "fried eggs", "egg yolks"])
a("tofu", "tofu", "protein", ["tofu", "tempeh", "falafel"])
a("beans", "cooked beans or lentils", "protein", ["chickpeas", "white beans", "black beans", "kidney beans", "pinto beans", "lentils", "refried beans", "baked beans", "bean(seed) ", "lima beans", "edamame", "hummus"])
# --- vegetables
for k, p, n in [
    ("broccoli", "broccoli", ["broccoli"]), ("cauliflower", "cauliflower", ["cauliflower"]), ("brussels", "brussels sprouts", ["brussels sprouts"]),
    ("carrot", "sliced carrots", ["carrot", "baby carrots"]), ("green_beans", "green beans", ["green beans", "snow peas", "peas", "green peas", "okra"]),
    ("asparagus", "asparagus", ["asparagus", "leeks"]),
    ("leafy_salad", "mixed salad greens", ["mixed greens", "lettuce", "garden salad", "salads", "caesar salad", "greek salad", "taco salad", "endive", "alfalfa", "bean sprouts", "cabbage", "coleslaw"]),
    ("spinach", "spinach or kale leaves", ["spinach (raw)", "spinach (cooked)", "kale", "arugula", "collards", "chard", "mustard greens", "tatsoi"]),
    ("bok_choy", "bok choy or Chinese greens", ["bok choy"]),
    ("tomato", "tomatoes", ["tomatoes", "cherry tomatoes", "sun dried tomatoes", "salsa", "tomatillo"]),
    ("cucumber", "cucumber slices", ["cucumbers", "celery", "radishes", "pickles", "jicama", "zucchini", "water chestnuts"]),
    ("squash", "roasted squash", ["squash", "pumpkins", "chayote squash", "eggplant", "parsnips", "turnips", "beets", "celery root", "fennel", "artichokes"]),
    ("bell_pepper", "bell peppers", ["bell peppers", "jalapenos"]), ("onion", "cooked onions", ["onions", "green onions", "shallots", "garlic"]),
    ("mushroom", "mushrooms", ["mushroom"]), ("mixed_veg", "mixed cooked vegetables", ["mixed vegetables", "sauerkraut", "kimchi", "seaweed", "nopales"]),
    ("avocado", "avocado", ["avocado", "guacamole"]), ("olives", "olives", ["olives", "capers"]),
]: a(k, p, "veg", n)
# --- fruit
for k, p, n in [
    ("berries", "berries", ["berries", "raspberries", "blackberries", "blueberries", "cranberries", "cherries"]), ("strawberry", "strawberries", ["strawberries"]),
    ("apple", "apple slices", ["apple", "applesauce"]), ("pear", "pear", ["pears", "peach", "peaches", "plums", "nectarines", "apricots", "figs"]),
    ("grapes", "grapes", ["grapes"]), ("melon", "melon cubes", ["cantaloupe", "honeydew melons", "watermelon"]),
    ("pineapple", "pineapple chunks", ["pineapple", "mangos", "papayas"]), ("banana", "banana", ["banana", "banana with peel"]),
    ("citrus", "orange or citrus slices", ["orange", "orange with peel", "mandarin oranges", "grapefruits", "lemon", "lime", "kiwi", "pomegranate"]),
    ("dried_fruit", "raisins or dried fruit", ["raisins", "prunes", "dates"]),
]: a(k, p, "fruit", n)
# --- nuts / dairy / sweet / sauces
a("almonds", "almonds", "nut", ["almonds", "nuts", "mixed nuts", "pecans", "walnuts", "cashews", "peanuts", "pistachios", "brazil nuts", "macadamia nuts", "pine nuts", "trail mix", "soy nuts", "chestnuts", "corn nuts"])
a("seeds", "seeds", "nut", ["chia seeds", "pumpkin seeds", "sunflower seeds", "seeds"])
a("cheese", "cheese", "dairy", ["cheese", "cheddar cheese", "swiss cheese", "mozzarella cheese", "parmesan cheese", "feta cheese", "goat cheese", "blue cheese", "colby cheese", "american cheese", "provolone cheese", "muenster cheese", "brie cheese", "gouda cheese", "havarti cheese", "romano cheese", "mexican cheese", "string cheese", "camembert cheese", "gorgonzola cheese", "cream cheese", "cottage cheese", "nut cheese"])
a("yogurt", "bowl of yogurt", "dairy", ["yogurt", "greek yogurt", "soy yogurt", "sour cream", "custard", "puddings", "mousse"])
a("baked_sweet", "a baked dessert (cake, brownie, cookie)", "sweet", ["brownies", "cookies", "chocolate chip cookies", "cakes", "cupcakes", "muffins", "scones", "donuts", "pastries", "croissants", "biscuits", "pies", "apple pie", "cheesecake", "chocolate cake", "carrot cake", "sandwich cookies", "turnover", "strudels", "cobbler", "fudge", "chocolate", "dark chocolate", "candies", "ice creams", "sorbet", "sundaes"])
a("dressing", "salad dressing or sauce drizzle", "sauce", ["vinaigrette", "ranch dressing", "caesar dressing", "salad dressing", "italian dressing", "french dressing", "mayonnaise", "barbecue sauce", "pesto", "teriyaki sauce", "pasta sauce", "pizza sauce", "gravy", "ketchup", "mustard", "soy sauce", "tahini", "chutney", "relish", "balsamic vinegar", "vinegar"])
a("oil", "olive oil", "oil", ["olive oil", "vegetable oil", "oil", "butter", "margarine", "fish oil"])
a("syrup_sugar", "syrup honey or sugar", "sauce", ["syrup", "maple syrup", "honey", "sugar", "brown sugar", "agave nectar", "jams", "peanut butter", "almond butter", "spreads", "flour", "corn starch", "cornmeal", "bread crumbs"])
a("soup", "a bowl of soup", "dish", ["broth", "chicken soup", "tomato soup", "miso soup", "chowders", "stews", "chili", "gumbo", "curries", "goulash"])
a("dumpling", "dumplings", "mixed", ["dumplings", "spring rolls", "egg rolls", "samosas", "empanadas", "tamales", "pot pies", "fritters", "tempura", "onion rings"])
a("chips", "chips or crackers", "snack", ["potato chips", "chips", "tortilla chips", "corn chips", "crackers", "pretzels", "popcorn", "rice cakes", "wafers", "nachos"])
# extras for non-N5k (Taiwanese / Asian) components, kcal per g
a("braised_pork", "braised pork belly", "protein", [], 3.3)
a("pork_cutlet", "breaded fried pork cutlet", "fried", [], 2.9)
a("fried_tofu", "fried tofu", "fried", [], 2.2)
a("braised_tofu", "braised tofu", "protein", [], 1.3)
a("tea_egg", "braised tea egg", "egg", [], 1.5)
a("stirfry_greens", "stir-fried green vegetables", "veg", [], 0.55)
a("chinese_cabbage", "napa cabbage", "veg", [], 0.3)
a("pickled_veg", "pickled vegetables", "veg", [], 0.3)
a("ground_pork_sauce", "minced pork sauce over rice", "protein", [], 1.9)
a("fish_ball", "fish balls or meatballs", "protein", [], 1.5)
a("scallion_pancake", "scallion pancake", "bread", [], 3.0)
a("steamed_bun", "steamed bun", "bread", [], 2.3)
a("beef_noodle_soup", "beef noodle soup", "dish", [], 0.9)
a("noodle_soup", "noodle soup", "dish", [], 0.7)
a("curry", "curry with rice", "dish", [], 1.3)
a("bubble_tea", "bubble milk tea", "drink", [], 0.6)

# background rows (never food): used to reject plate / table / tray / utensil masks
BG = ["an empty white plate", "a wooden table", "a metal tray", "a ceramic bowl", "chopsticks or a fork", "a shadow on a table", "a paper napkin", "a black frame or edge of a photo"]


def n5k_map():
    m = {}
    for k, (_, _, ns, _) in C.items():
        for n in ns: m.setdefault(n, k)  # first mapping wins
    return m


def build_vocab(fit_dishes, meta_csv, out):
    """fit_dishes: dishes with ingr [{name,g,kcal}]. Writes the vocab json; kcal/g = sum kcal / sum g over fit dishes where >= 50 g seen."""
    meta = {r[0]: float(r[2]) for r in list(csv.reader(open(meta_csv)))[1:]}
    n2k = n5k_map()
    g = collections.defaultdict(float); kc = collections.defaultdict(float)
    for d in fit_dishes:
        for i in d["ingr"]:
            k = n2k.get(i["name"])
            if k: g[k] += i["g"]; kc[k] += i["kcal"]
    rows = []
    for k, (p, grp, ns, kg) in C.items():
        if kg is None:
            if g[k] > 50: kg = kc[k] / g[k]
            else:
                vs = [meta[n] for n in ns if n in meta]; kg = sum(vs) / len(vs) if vs else 1.0
        rows.append(dict(id=k, en=p, group=grp, kcal_g=round(kg, 3), n5k=ns))
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    return rows, n2k
