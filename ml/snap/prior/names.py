"""Clean display names. Rules turn the most-reported FNDDS description of a vocab row's cluster into a short natural English name;
MANUAL overrides fix clusters where the row `en` or the rule output is misleading. Rows without a clean name keep `en` (no `display` key)."""
import re

# segments of an FNDDS description that carry no information for a short name
NOISE = re.compile(
    r"(\bNFS\b|\bNS\b|as to|^not |fat not added|fat added|^made |made with|made from|from (restaurant|fast food|frozen|canned|dry|fresh|home|school|packaged|raw|mix|refrigerated)|home recipe|"
    r"restaurant|fast food|reduced|^low|lowfat|low fat|nonfat|fat free|skim|diet|light\b|unsweetened|sweetened|skin|lean|^with .*(oil|butter|margarine|fat|sugar|sweetener)|"
    r"canned|bottled|carton|frozen|dry\b|dried|cooked|raw$|^regular|plain$|^1 |^2 |patty|medium|small|large|calcium|fortified|enriched|sodium|salt|unsalted|"
    r"ready|instant|prepared|^ns|^all |^any |type|flavor|pre-|decaf|fluid|rinsed|drained|^solids|^liquid|no sauce|no added|without|coating|breaded|battered|"
    r"^baby|^junior|^toddler|^strained|(white|wheat|whole|multigrain|sour dough|rye|pumpernickel)$|^sliced$|^mashed$|^peel|^boneless$)", re.I)
CLASS_NOUNS = set("""cookie cookies cake cupcake soup cheese bread roll rolls crackers cracker cereal candy yogurt juice tea coffee milk egg eggs pie pudding potato rice pasta salad sauce
dressing tortilla muffin bagel chocolate pizza sandwich burger bar bars smoothie drink beans bean chips fish shrimp sausage ham turkey pork beef chicken lamb fruit nuts nut oil
jelly jam syrup dip spread gravy stew chili curry noodles omelet pancakes waffles donut doughnut biscuit cracker pretzels popcorn ice squash pepper peppers onion onions cabbage
lettuce spinach tomato tomatoes corn apple banana orange grapes melon berries cream butter margarine snack granola oatmeal""".split())

MANUAL = {
    # beverages
    "us_brewed_coffee": "Black coffee", "us_soft_drink": "Soda (cola)", "us_milk": "Milk", "us_evaporated_milk": "Whole milk", "us_iced_brewed_tea": "Iced tea",
    "us_hot_leaf_tea": "Hot tea", "us_100_orange_juice": "Orange juice", "us_100_apple_juice_with_calcium_added": "Apple juice", "us_fruit_juice_drink": "Fruit drink (lemonade)",
    "us_vegetable_and_fruit_juice_drink_with_high_vitamin_c": "Fruit juice drink", "us_instant_pre_lightened_and_pre_sweetened_with_low_calorie_swe": "Instant coffee",
    "us_iced_latte_decaffeinated_coffee": "Iced coffee", "us_fruit_smoothie": "Fruit smoothie", "us_almond_milk": "Almond milk", "us_root_beer_soft_drink": "Root beer",
    "cola": "Cola", "iced_americano": "Iced americano", "latte_coffee": "Latte", "cappuccino_coffee": "Cappuccino", "mocha_coffee": "Mocha", "caramel_macchiato": "Caramel macchiato",
    "matcha_latte": "Matcha latte",
    # produce
    "us_for_use_on_a_sandwich_spinach": "Lettuce and tomato", "us_baked_banana": "Banana", "us_baked_apple": "Apple", "us_tomatoes": "Tomatoes", "us_lettuce": "Lettuce",
    "us_raw_strawberries": "Strawberries", "us_carrots": "Carrots", "us_raw_grapes": "Grapes", "us_raw_blueberries": "Blueberries", "us_raw_avocado": "Avocado",
    "us_raw_watermelon": "Watermelon", "us_raw_clementine": "Clementine", "us_raw_cantaloupe": "Cantaloupe", "us_raw_romaine_lettuce": "Romaine lettuce",
    "us_raw_mixed_salad_greens": "Mixed salad greens", "us_corn": "Corn", "us_raw_peppers": "Peppers", "us_frozen_raspberries": "Raspberries", "us_green_cabbage": "Cabbage",
    "us_yellow_or_green_summer_squash": "Summer squash", "grapes": "Grapes", "us_applesauce": "Applesauce", "us_pickles": "Pickles",
    # staples
    "us_rice": "White rice", "us_scrambled_egg_white_omelet": "Scrambled eggs or omelet", "us_white_egg": "Fried egg", "us_boiled_or_poached_whole_egg": "Boiled egg",
    "us_egg_omelet_or_scrambled_egg_with_tomatoes": "Omelet with cheese", "us_white_made_from_home_recipe_or_purchased_at_a_bakery_bread": "White bread",
    "us_wheat_or_cracked_wheat_bread_with_raisins": "Wheat bread", "us_toasted_white_or_bread": "Toast", "us_toasted_sour_dough_bread": "Sourdough bread",
    "us_multigrain_bread_with_raisins": "Multigrain bread", "us_white_soft_roll": "Bread roll", "us_sweet_frosted_roll": "Cinnamon roll", "us_biscuit": "Biscuit",
    "us_corn_tortilla": "Corn tortilla", "us_oatmeal": "Oatmeal", "us_granola_cereal": "Granola", "us_o_s_honey_nut_cereal": "Breakfast cereal", "us_o_s_cereal": "Breakfast cereal",
    "us_cereal_or_granola_bar": "Granola bar", "us_wheat_bagel_with_raisins": "Bagel", "us_english_wheat_bran_muffin": "English muffin", "us_fruit_muffin": "Muffin",
    "us_hard_pretzels": "Pretzels", "us_graham_crackers": "Graham crackers", "us_butter_crackers": "Crackers", "us_saltine_crackers": "Saltine crackers",
    "us_cheese_crackers": "Cheese crackers", "us_cheese_filled_crackers_sandwich": "Peanut butter crackers",
    # potato
    "us_home_fries_potato": "French fries", "us_popped_potato_chips": "Potato chips", "us_mashed_potato": "Mashed potatoes", "us_boiled_peel_eaten_potato": "Baked potato",
    "us_hash_brown_potato": "Hash browns", "potato": "Potato", "potato_chips": "Potato chips", "ketchup": "Ketchup",
    # meat
    "us_chicken_breast": "Chicken breast", "us_beef_steak": "Beef steak", "us_pork_chop": "Pork chop", "us_pork_sausage": "Sausage", "us_chicken_nuggets": "Chicken nuggets",
    "us_chicken_tenders_or_strips": "Chicken tenders", "us_chicken_wing": "Chicken wings", "us_chicken_thigh": "Chicken thigh", "us_stewed_chicken": "Chicken",
    "us_salmon_fish": "Salmon", "us_smoked_or_cured_pork_bacon": "Bacon", "bacon": "Bacon", "us_luncheon_meat": "Deli ham", "roast_chicken": "Roast chicken",
    "us_chicken_fillet_sandwich": "Chicken sandwich", "us_cheeseburger": "Cheeseburger", "us_hamburger": "Hamburger", "us_hot_dog_sandwich": "Hot dog",
    # mixed dishes / sandwiches
    "us_pizza": "Cheese pizza", "us_pizza_with_meat_other_than_pepperoni": "Meat pizza", "us_pizza_with_pepperoni": "Pepperoni pizza",
    "us_pasta_with_tomato_based_sauce_and_beans_or_lentils": "Pasta with tomato sauce", "us_pasta_with_cream_sauce_and_meat": "Pasta with cream sauce",
    "us_macaroni_or_noodles_with_cheese_and_tuna": "Macaroni and cheese", "us_macaroni_or_pasta_salad_with_tuna_and_egg": "Macaroni salad",
    "us_peanut_butter_and_jelly_sandwich": "Peanut butter and jelly sandwich", "us_ham_sandwich_on_white_with_cheese": "Ham and cheese sandwich",
    "us_turkey_sandwich_on_white_with_cheese": "Turkey sandwich", "us_turkey_sandwich_on_white": "Turkey sandwich", "us_ham_sandwich_on_white": "Ham sandwich",
    "us_grilled_cheese_sandwich": "Grilled cheese sandwich", "us_egg_sandwich_on_bagel": "Egg sandwich", "us_egg_burrito_with_sausage": "Breakfast burrito",
    "us_noodle_soup": "Chicken noodle soup", "us_ramen_noodles_water_added_soup": "Ramen noodle soup", "us_vegetable_soup": "Vegetable soup", "us_croutons": "Croutons",
    "us_white_rice_with_tomatoes_and_or_tomato_based_sauce": "Rice with tomato sauce", "us_corn_tortilla_beef_taco": "Beef taco", "us_flour_tortilla_beef_taco": "Beef taco",
    "us_meat_no_cheese_taco": "Taco", "us_fast_food_restaurant_coleslaw": "Coleslaw", "us_cake_or_cupcake": "Cake", "us_brownie_cookie": "Brownie",
    "us_chocolate_chip_cookie": "Chocolate chip cookie", "us_oatmeal_cookie": "Oatmeal cookie", "us_chocolate_sandwich_cookie": "Sandwich cookie (Oreo)",
    "us_butter_or_sugar_cookie_with_fruit_and_or_nuts": "Sugar cookie", "us_other_chocolate_candy": "Chocolate candy", "us_caramel_filled_chocolate_candy": "Filled chocolate candy",
    "us_chocolate_candy_with_nuts": "Chocolate candy with nuts", "us_peanut_butter_filled_chocolate_candy": "Peanut butter chocolate", "us_hard_candy": "Hard candy",
    "us_fruit_snacks_candy": "Fruit snacks", "us_microwave_popcorn": "Popcorn", "us_cheese_flavored_corn_snacks": "Cheese puffs", "us_tortilla_chips": "Tortilla chips",
    "us_nacho_cheese_flavor_tortilla_chips": "Nacho chips", "us_plain_corn_chips": "Corn chips",
    # dairy
    "us_cheddar_cheese": "Cheddar cheese", "us_cheese": "Cheese", "us_greek_yogurt": "Greek yogurt", "us_fruit_yogurt": "Fruit yogurt", "us_yogurt": "Yogurt",
    "us_mozzarella_cheese": "Mozzarella", "us_parmesan_dry_grated_cheese": "Parmesan", "us_american_cheese": "American cheese", "us_cottage_cheese": "Cottage cheese",
    "us_chocolate_milk": "Chocolate milk", "us_oat_milk": "Oat milk",
    # non-FNDDS
    "grapes": "Grapes", "mango": "Mango", "sweet_corn": "Sweet corn", "cooked_white_rice": "White rice", "whole_milk": "Whole milk", "hamburger_bun": "Hamburger bun",
    "vanilla_ice_cream": "Vanilla ice cream", "bell_pepper": "Bell pepper", "kiwi_fruit": "Kiwi", "strawberries": "Strawberries", "blueberries": "Blueberries",
    "cantaloupe_melon": "Cantaloupe", "iceberg_lettuce": "Iceberg lettuce", "romaine_lettuce": "Romaine lettuce", "button_mushroom": "Mushrooms",
    "fresh_spring_roll_viet": "Fresh spring roll", "instant_noodles_cooked": "Instant noodles", "chicken_curry_indian": "Chicken curry", "curry_rice": "Japanese curry rice",
    "kebab_doner": "Doner kebab", "sushi_nigiri": "Sushi (nigiri)", "sushi_roll_maki": "Sushi roll", "ramen_shoyu": "Ramen", "ramen_tonkotsu": "Ramen (tonkotsu)", "ramen_miso": "Ramen (miso)",
    "scrambled_egg_tw": "Scrambled eggs", "pan_fried_egg": "Fried egg", "fried_egg": "Fried egg", "boiled_egg": "Boiled egg", "egg_raw": "Egg", "chicken_breast_raw": "Chicken breast (raw)",
    "milk_chocolate": "Milk chocolate", "dark_chocolate": "Dark chocolate", "sirloin_steak_raw": "Sirloin steak (raw)", "filet_mignon_steak_raw": "Filet mignon (raw)",
    "pork_ribs_raw": "Pork ribs (raw)", "atlantic_salmon_raw": "Salmon (raw)", "white_shrimp_raw": "Shrimp (raw)", "chicken_wing_raw": "Chicken wings (raw)",
}


def _tc(s):
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def rule_name(desc):
    """FNDDS description -> short name, or None if nothing sensible is left."""
    d = re.sub(r"\s*\(.*?\)", "", desc)
    segs = [s.strip() for s in d.split(",") if s.strip()]
    if not segs: return None
    head = segs[0]
    mods = [s for s in segs[1:] if not NOISE.search(s) and len(s.split()) <= 3]
    if NOISE.search(head) and len(segs) > 1: return None
    hl = head.lower()
    if hl.split()[-1] in CLASS_NOUNS or hl in CLASS_NOUNS:
        if mods and not NOISE.search(mods[0]) and hl.split()[-1] not in ("sandwich",):
            m = mods[0].lower()
            if re.match(r"(with|and)\b", m): return _tc(f"{hl} {m}")
            return _tc(f"{m} {hl}")
        return _tc(hl)
    if mods and mods[0].lower().startswith("with"): return _tc(f"{hl} {mods[0].lower()}")
    return _tc(hl)


def display_for(vid, en, top_desc, kind):
    if vid in MANUAL: return MANUAL[vid]
    if top_desc:
        n = rule_name(top_desc)
        if n:
            n = re.sub(r" (on|in) (white|wheat|a white|white bread|wheat bread)( bun)?$| from fast food / restaurant| excluding carrots| \(.*\)$", "", n)
            n = re.sub(r"^(Cereal|100%) ", "", n).replace(" / ", " ")
        if n and 2 < len(n) <= 40: return n
    if kind in ("tier", "default"):
        n = re.sub(r"\s*\(.*?\)", "", en).strip()
        return _tc(n) if n and len(n) <= 40 else None
    return None


# short, Lens-style labels (second pass: shortened/fixed names found while reviewing the top 800)
MANUAL.update({
    "us_pizza": "Pizza", "us_pizza_with_meat_other_than_pepperoni": "Meat pizza", "us_pizza_with_meat_and_vegetables": "Supreme pizza",
    "us_antipasto_with_ham": "Antipasto", "us_herring_fish": "Herring", "us_sardines_fish": "Sardines", "us_borscht_soup": "Borscht", "us_gazpacho_soup": "Gazpacho",
    "us_minestrone_soup": "Minestrone", "us_eclair_custard_or_cream_filled_cream_puff": "Eclair", "us_prosciutto_ham": "Prosciutto", "us_salami_sandwich_on_wheat_with_cheese": "Salami sandwich",
    "us_swedish_meatballs_with_cream_or_white_sauce": "Swedish meatballs", "us_apple_strudel": "Apple strudel", "us_gouda_or_edam_cheese": "Gouda", "us_brie_cheese": "Brie",
    "us_feta_cheese": "Feta", "us_camembert_cheese": "Camembert", "us_french_or_vienna_bread": "French bread", "us_meatless_cheese_quiche": "Quiche", "us_quiche_with_meat": "Quiche",
    "us_meatless_spinach_quiche": "Spinach quiche", "us_lasagna_with_meat": "Lasagna", "us_ravioli": "Ravioli", "us_fruit_filled_crepe": "Fruit crepe", "us_chocolate_filled_crepe": "Chocolate crepe",
    "us_empanada": "Empanada", "us_beef_empanada": "Beef empanada", "us_chicken_empanada": "Chicken empanada", "us_bratwurst_with_cheese": "Bratwurst", "us_soft_pretzels": "Soft pretzel",
    "us_shrimp": "Fried shrimp", "us_german_style_potato_salad": "Potato salad", "us_ginger_ale_soft_drink": "Ginger ale", "us_latte_flavored_coffee": "Latte",
    "us_fruit_flavored_drink_lemonade": "Lemonade", "us_cake_type_doughnut_with_icing": "Iced doughnut", "us_yeast_type_doughnut": "Doughnut", "us_frozen_asparagus": "Mixed vegetables",
    "us_white_or_bread": "Bread", "us_ice_cream_bar": "Ice cream bar", "us_turkey": "Roast turkey", "us_tuna_salad_sandwich_on_wheat_with_cheese": "Tuna sandwich",
    "us_pork_ribs": "Pork ribs", "us_baked_potato": "Potato", "us_sweet_potato": "Sweet potato", "us_beans": "Beans", "us_vanilla": "Ice cream", "us_hamburger": "Hamburger",
    "us_chili_with_meat": "Chili", "us_cooked_with_butter_or_margarine": "Plantain", "us_fried_chicken": "Fried chicken", "us_chicken_wings_with_other_sauce": "Chicken wings",
    "us_egg_sandwich_on_english_muffin_with_sausage": "Breakfast sandwich", "us_bologna_sandwich_on_wheat_with_cheese": "Bologna sandwich", "us_fried_dumpling_or_pot_sticker_wonton": "Dumplings",
    "us_beef_with_tomato_based_sauce": "Meatloaf", "us_chicken_or_turkey_and_noodles": "Chicken and noodles", "us_sweetened_soy_milk": "Soy milk", "us_fruit_flavored_drink": "Fruit drink",
    "us_beef_burrito_with_rice": "Burrito", "us_1_medium_patty_cheeseburger": "Cheeseburger", "us_cheese_only_quesadilla": "Quesadilla", "us_shrimp_fried_rice": "Shrimp fried rice",
    "us_fried_meatless_rice": "Fried rice", "us_beans_and_white_rice": "Beans and rice", "us_egg_omelet_or_scrambled_egg_with_vegetables": "Omelet", "us_pan_dulce": "Pan dulce",
    "us_ham": "Ham", "us_mushrooms": "Mushrooms", "us_kale": "Kale", "us_pasta": "Pasta", "us_crackers": "Crackers", "us_popcorn": "Popcorn",
    "pho": "Pho", "banh_mi": "Banh mi", "gyoza": "Gyoza", "bulgogi": "Bulgogi", "salt_pepper_chicken": "Popcorn chicken", "chicken_cutlet_tw": "Fried chicken cutlet",
    "sashimi": "Sashimi", "hot_dog_sausage": "Hot dog", "honey_glazed_donut": "Glazed donut", "ramen_noodles_cooked": "Ramen noodles", "hot_pot_mixed": "Hot pot",
    "dan_bing": "Egg crepe", "tonkatsu": "Pork cutlet", "karaage": "Fried chicken", "nasi_goreng": "Nasi goreng", "lo_mein": "Lo mein", "stir_fried_greens": "Stir-fried greens",
    "pineapple_bun": "Pineapple bun", "egg_tart_portuguese": "Egg tart", "takoyaki": "Takoyaki", "okonomiyaki": "Okonomiyaki", "tamagoyaki": "Rolled omelet", "natto": "Natto",
    "yakitori": "Yakitori", "grilled_mackerel": "Grilled mackerel", "grilled_eel": "Grilled eel", "satay_chicken": "Chicken satay", "green_curry_chicken": "Green curry",
    "milk_tea_tw": "Milk tea", "green_tea_unsweet": "Green tea", "black_tea_unsweet": "Black tea", "cocoa_milk": "Hot cocoa", "lemon_juice_tw": "Lemon drink",
    "zaru_soba": "Cold soba", "yaki_udon": "Yaki udon", "stir_fried_noodles": "Stir-fried noodles", "ham_egg_sandwich_tw": "Ham and egg sandwich", "mapo_tofu": "Mapo tofu",
    "chicken_biryani": "Biryani", "biryani": "Biryani", "onigiri": "Onigiri", "udon_soup": "Udon", "spaghetti_bolognese_tw": "Spaghetti bolognese", "roasted_sweet_potato": "Roasted sweet potato",
    "us_caesar_salad": "Caesar salad",
})

MANUAL.update({"us_turkey_or_chicken_burger": "Turkey burger", "us_egg_sandwich_on_croissant_with_bacon": "Breakfast sandwich", "us_fruit_juice": "Fruit juice",
  "us_iced_tea_lemonade_juice_drink": "Iced tea lemonade", "us_trail_mix_with_pretzels": "Trail mix", "us_beef_and_vegetables": "Beef and vegetables"})
MANUAL.update({"us_home_fries_potato": "Home fries"})
