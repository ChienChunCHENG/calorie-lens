"""36 new vocab rows (very common US/EU foods that the 2,278-row vocab lacks). kcal per 100 g, portion [min, typ, max] g.
src: 'FNDDS #code' = value copied from the FNDDS 2021-2023 table in research/r3/dl/f24 (USDA, public domain); 'FNDDS proxy' = nearest FNDDS dish used as proxy;
'composed' = computed from FNDDS components; 'est' = no FNDDS/SR Legacy entry with that name was found offline, kcal estimated from typical recipe composition
(the same convention as the 204 'estimated' rows already in vocab.json). Portions: FNDDS portion weights where the entry has a matching one, otherwise typical servings (est).
tier: hand floor for the prior (A/B/C = 800/250/80 reports, see prior_lib.TIERS)."""

# id, en, display, kcal100, portion_g, cat, src, tier
ROWS = [
    ("schnitzel", "wiener schnitzel (breaded fried veal or pork cutlet)", "Schnitzel", 240, [120, 200, 300], "meat", "est: no FNDDS/SR entry for the dish; breaded fried cutlet class 220-260 kcal/100 g", "B"),
    ("risotto", "risotto", "Risotto", 150, [150, 250, 400], "rice", "est: no FNDDS/SR entry; arborio rice cooked in stock with butter and parmesan, 140-170", "B"),
    ("fish_and_chips", "fish and chips", "Fish and chips", 266, [250, 400, 550], "dish", "composed: 140 g fried cod (FNDDS 26109140, 217) + 150 g fast-food fries (FNDDS 71401030, 312)", "B"),
    ("moussaka", "moussaka", "Moussaka", 96, [200, 300, 450], "dish", "FNDDS 77316600 (Eggplant and meat casserole = moussaka); likely low vs restaurant moussaka with bechamel (about 130-150)", "B"),
    ("muesli", "muesli", "Muesli", 360, [30, 55, 90], "dish", "est: SR Legacy muesli entries recalled as 340-370; not re-fetched offline", "B"),
    ("spaghetti_carbonara", "spaghetti carbonara", "Spaghetti carbonara", 190, [200, 300, 450], "noodle", "est: no FNDDS entry; pasta, egg, pecorino, guanciale", "B"),
    ("ciabatta", "ciabatta bread", "Ciabatta", 259, [40, 80, 130], "bread", "FNDDS proxy 51109010 (Bread, Italian)", "B"),
    ("goulash", "goulash (beef paprika stew)", "Goulash", 107, [200, 300, 450], "dish", "FNDDS proxy 27311410 (Stew, beef)", "B"),
    ("cassoulet", "cassoulet (white beans with sausage and duck)", "Cassoulet", 150, [250, 350, 500], "dish", "est: no FNDDS/SR entry; beans, pork sausage, duck confit", "C"),
    ("panna_cotta", "panna cotta", "Panna cotta", 230, [80, 130, 180], "dessert", "est: no FNDDS/SR entry; cream, milk, sugar, gelatin", "B"),
    ("smoked_salmon", "smoked salmon (lox)", "Smoked salmon", 117, [30, 60, 100], "seafood", "FNDDS 26137190 (Fish, salmon, smoked)", "B"),
    ("sausage_roll", "sausage roll", "Sausage roll", 330, [60, 100, 160], "dish", "est: no FNDDS entry; pork sausage meat in puff pastry, bakery labels about 320-340", "B"),
    ("cottage_pie", "cottage pie (shepherd's pie with beef)", "Cottage pie", 123, [200, 300, 450], "dish", "FNDDS 27311510 (Shepherd's pie)", "B"),
    ("yorkshire_pudding", "yorkshire pudding", "Yorkshire pudding", 268, [25, 45, 80], "bread", "FNDDS proxy 52311010 (Popover)", "B"),
    ("tzatziki", "tzatziki", "Tzatziki", 91, [30, 60, 120], "sauce", "FNDDS 11440060 (Tzatziki dip)", "B"),
    ("caesar_salad", "caesar salad with dressing, croutons and parmesan", "Caesar salad", 155, [150, 250, 400], "vegetable", "composed: FNDDS 72116150 (romaine, no dressing, 77) + about 15 g caesar dressing (FNDDS 83102000, 542) per 100 g", "B"),
    ("mayonnaise", "mayonnaise", "Mayonnaise", 680, [7, 14, 30], "sauce", "FNDDS 83107000 (Mayonnaise, regular)", "B"),
    ("french_fries", "french fries", "French fries", 312, [70, 115, 180], "snack", "FNDDS 71401030 (Potato, french fries, fast food); portions from FNDDS kids/value orders 70/85 g, medium est", "A"),
    ("guacamole", "guacamole", "Guacamole", 155, [30, 60, 120], "sauce", "FNDDS 63409010 (Guacamole, NFS)", "B"),
    ("salsa", "salsa", "Salsa", 34, [30, 60, 120], "sauce", "FNDDS 74402150 (Salsa, red)", "B"),
    ("greek_salad", "greek salad with feta and olives", "Greek salad", 95, [150, 250, 400], "vegetable", "composed: FNDDS 75146000 (Greek salad, no dressing, 48) + feta, olives, oil dressing (est)", "B"),
    ("cobb_salad", "cobb salad", "Cobb salad", 150, [200, 330, 500], "vegetable", "composed: FNDDS 75148010 (Cobb salad, no dressing, 93) + dressing (est)", "B"),
    ("fettuccine_alfredo", "fettuccine alfredo", "Fettuccine Alfredo", 180, [200, 320, 480], "noodle", "est: pasta (about 160) with Alfredo sauce (FNDDS 14650160, 150) plus extra cheese", "B"),
    ("eggs_benedict", "eggs benedict", "Eggs Benedict", 230, [150, 230, 330], "dish", "est: no FNDDS entry; muffin, ham, poached egg, hollandaise", "B"),
    ("shakshuka", "shakshuka (eggs poached in tomato sauce)", "Shakshuka", 100, [200, 300, 450], "dish", "est: no FNDDS entry; eggs in spiced tomato-pepper sauce", "C"),
    ("caprese_salad", "caprese salad (tomato and mozzarella)", "Caprese salad", 190, [100, 160, 250], "vegetable", "est: mozzarella (about 280) + tomato (18) + olive oil", "B"),
    ("pasta_salad", "pasta salad with mayonnaise", "Pasta salad", 221, [100, 200, 300], "noodle", "FNDDS 58148110 (Macaroni or pasta salad, made with mayonnaise)", "B"),
    ("baguette", "baguette", "Baguette", 272, [30, 60, 120], "bread", "FNDDS 51107010 (Bread, French or Vienna); portion est", "B"),
    ("apple_pie", "apple pie", "Apple pie", 296, [75, 150, 250], "dessert", "FNDDS 53301000 (Pie, apple); FNDDS slice weights 75/150/250", "A"),
    ("bagel_cream_cheese", "bagel with cream cheese", "Bagel with cream cheese", 285, [100, 130, 200], "bread", "composed: bagel (about 270) + about 15% cream cheese (FNDDS 14301010, 343)", "B"),
    ("blt_sandwich", "bacon lettuce and tomato sandwich (BLT)", "BLT sandwich", 231, [80, 105, 180], "sandwich", "FNDDS 27520610 (Bacon, lettuce, tomato sandwich on white); FNDDS weight 105 g", "B"),
    ("spaghetti_meatballs", "spaghetti with meatballs and tomato sauce", "Spaghetti and meatballs", 150, [250, 380, 550], "noodle", "est: no matching FNDDS entry found; pasta 160, meatballs with sauce 200", "B"),
    ("avocado_toast", "avocado toast", "Avocado toast", 220, [90, 140, 220], "bread", "est: toast (about 265) + avocado (FNDDS raw avocado about 160)", "B"),
    ("chicken_tikka_masala", "chicken tikka masala", "Chicken tikka masala", 150, [200, 300, 450], "dish", "est: no FNDDS entry; restaurant labels about 130-170", "B"),
    ("shawarma", "shawarma wrap", "Shawarma", 210, [150, 250, 350], "sandwich", "est: no FNDDS entry; meat, sauce and flatbread", "C"),
    ("croque_monsieur", "croque monsieur (grilled ham and cheese sandwich)", "Croque monsieur", 280, [150, 200, 280], "sandwich", "est: no FNDDS entry; ham, gruyere, bechamel on white bread", "B"),
]
