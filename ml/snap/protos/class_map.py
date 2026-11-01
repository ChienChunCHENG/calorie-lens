"""Explicit source-class -> vocab-id mapping for the image prototypes (US/EU focus).
A CONCEPT = one prototype class. vocab = accepted vocab ids (FIRST = canonical row; every id listed receives the concept's image similarity at
runtime, text similarity then separates the variants). f101 = Food-101 class names (train split images, whole photo); oi = Open Images class
names (box crops). Where the 2,278-entry vocab has no exact dish the concept maps to the nearest GENERIC entry (generic=True; flagged in
RESULTS.md). Food-101 classes with no usable vocab entry are listed in NO_VOCAB (documented vocab gaps, not trained).
"""
C = {}


def c(name, vocab, f101=(), oi=(), generic=False, grp=None):
    C[name] = dict(vocab=list(vocab), f101=list(f101), oi=list(oi), generic=generic, grp=grp)


# ---- mains / fast food
c("pizza", ["us_pizza", "us_pizza_with_pepperoni", "us_pizza_with_meat_other_than_pepperoni", "us_pizza_with_meat_and_vegetables", "us_pizza_with_meat_gluten_free_thin_crust"], ["pizza"], ["Pizza"], grp="fastfood")
c("hamburger", ["us_hamburger", "us_cheeseburger", "us_1_medium_patty_hamburger", "us_1_medium_patty_cheeseburger", "us_1_large_patty_hamburger", "us_1_large_patty_cheeseburger", "us_1_small_patty_hamburger", "us_2_medium_patties_double_hamburger"], ["hamburger"], ["Hamburger"], grp="fastfood")
c("hot_dog", ["us_hot_dog", "us_hot_dog_sandwich", "us_meat_and_poultry_hot_dog"], ["hot_dog"], ["Hot dog"], grp="fastfood")
c("french_fries", ["us_home_fries_potato", "us_fry_shaped_potato_sticks"], ["french_fries"], ["French fries"], generic=True, grp="fastfood")
c("taco", ["us_taco", "us_meat_no_cheese_taco", "us_cheese_only_taco", "us_flour_tortilla_beef_taco", "us_corn_tortilla_beef_taco", "us_fish_taco"], ["tacos"], ["Taco"], grp="fastfood")
c("burrito", ["us_burrito", "us_beef_burrito_with_rice", "us_pork_burrito_with_rice", "us_meat_burrito_with_rice", "us_chicken_burrito_with_rice", "us_cheese_burrito_with_beans"], ["breakfast_burrito"], ["Burrito"], grp="fastfood")
c("nachos", ["us_nachos", "us_chicken_nachos", "us_nachos_with_beans", "us_cheese_only_nachos", "us_beef_or_pork_nachos"], ["nachos"], grp="fastfood")
c("quesadilla", ["us_quesadilla", "us_chicken_quesadilla", "us_cheese_only_quesadilla", "us_beef_or_pork_quesadilla"], ["chicken_quesadilla"], grp="fastfood")
c("onion_rings", ["us_fried_onion_rings"], ["onion_rings"], grp="fastfood")
c("chicken_wings", ["us_chicken_wing", "us_chicken_wings_with_other_sauces_or_seasoning"], ["chicken_wings"], grp="meat")
c("grilled_cheese", ["us_grilled_cheese_sandwich"], ["grilled_cheese_sandwich"], grp="sandwich")
c("club_sandwich", ["us_club_sandwich_on_white", "us_club_sandwich_on_white_with_cheese"], ["club_sandwich"], grp="sandwich")
c("bbq_pork_sandwich", ["us_barbecue_pork_sandwich", "us_barbecue_sandwich"], ["pulled_pork_sandwich"], grp="sandwich")
c("sandwich", ["us_sandwich", "us_ham_sandwich_on_white", "us_turkey_sandwich_on_white", "us_multiple_meat_sandwich_on_white", "us_italian_sandwich_on_white", "us_cheese_sandwich"], [], ["Sandwich", "Submarine sandwich"], generic=True, grp="sandwich")
c("garlic_bread", ["us_garlic_bread", "us_garlic_bread_with_melted_cheese", "us_garlic_bread_with_parmesan_cheese"], ["garlic_bread"], grp="bread")
c("bruschetta", ["us_bruschetta"], ["bruschetta"], grp="bread")
# ---- meat
c("steak", ["us_steak", "us_beef_steak"], ["filet_mignon", "prime_rib"], generic=True, grp="meat")
c("ribs", ["us_ribs", "us_pork_ribs"], ["baby_back_ribs"], grp="meat")
c("pork_chop", ["us_pork_chop", "us_stuffed_pork_chop"], ["pork_chop"], grp="meat")
c("steak_tartare", ["us_steak_tartare"], ["beef_tartare"], grp="meat")
c("peking_duck", ["us_peking_duck"], ["peking_duck"], grp="meat")
# ---- pasta / rice / noodles
c("spaghetti_meat", ["spaghetti_bolognese_tw"], ["spaghetti_bolognese"], grp="pasta")
c("pasta", ["us_pasta", "us_pasta_with_sauce", "us_pasta_with_cream_sauce_and_meat", "us_pasta_with_vegetables"], ["spaghetti_carbonara"], ["Pasta"], generic=True, grp="pasta")
c("lasagna", ["us_lasagna_with_meat", "us_meatless_lasagna_with_vegetables", "us_lasagna_with_chicken_or_turkey"], ["lasagna"], grp="pasta")
c("ravioli", ["us_ravioli", "us_ravioli_with_tomato_sauce", "us_ravioli_with_cream_sauce"], ["ravioli"], grp="pasta")
c("gnocchi", ["us_potato_gnocchi", "us_cheese_gnocchi"], ["gnocchi"], grp="pasta")
c("mac_cheese", ["us_macaroni_or_noodles", "macaroni"], ["macaroni_and_cheese"], generic=True, grp="pasta")
c("ramen", ["us_ramen_bowl", "us_ramen_bowl_with_meat_and_egg", "ramen_tonkotsu", "ramen_shoyu", "ramen_miso"], ["ramen"], grp="asian")
c("pho", ["us_pho_soup_with_meat"], ["pho"], grp="asian")
c("pad_thai", ["pad_thai"], ["pad_thai"], grp="asian")
c("fried_rice", ["egg_fried_rice", "shrimp_fried_rice", "us_fried_rice_with_pork", "us_fried_rice_with_beef", "us_fried_rice_with_shrimp", "yangzhou_fried_rice"], ["fried_rice"], grp="asian")
c("bibimbap", ["bibimbap", "us_korean_bibimbap"], ["bibimbap"], grp="asian")
c("sushi", ["us_sushi", "sushi_nigiri", "sushi_roll_maki", "us_avocado_sushi_roll"], ["sushi"], ["Sushi"], grp="asian")
c("sashimi", ["sashimi", "tuna_sashimi"], ["sashimi"], grp="asian")
c("gyoza", ["gyoza", "guo_tie"], ["gyoza", "dumplings"], grp="asian")
c("spring_rolls", ["spring_roll_fried", "spring_roll_taiwan_peanut"], ["spring_rolls"], grp="asian")
c("samosa", ["samosa"], ["samosa"], grp="asian")
c("falafel", ["falafel"], ["falafel"], grp="asian")
c("hummus", ["hummus"], ["hummus"], grp="asian")
c("miso_soup", ["miso_soup"], ["miso_soup"], grp="soup")
c("hot_sour_soup", ["sour_hot_soup"], ["hot_and_sour_soup"], grp="soup")
c("takoyaki", ["takoyaki"], ["takoyaki"], grp="asian")
c("edamame", ["edamame"], ["edamame"], grp="asian")
c("chicken_curry", ["chicken_curry_indian", "us_chicken_curry_with_rice", "green_curry_chicken"], ["chicken_curry"], grp="asian")
c("french_onion_soup", ["us_french_onion_soup"], ["french_onion_soup"], grp="soup")
c("clam_chowder", ["us_new_england_clam_chowder_soup", "us_manhattan_clam_chowder_soup"], ["clam_chowder"], grp="soup")
c("paella", ["us_paella", "us_paella_with_seafood"], ["paella"], grp="pasta")
# ---- seafood
c("oysters", ["oysters"], ["oysters"], ["Oyster"], grp="seafood")
c("mussels", ["us_mussels"], ["mussels"], grp="seafood")
c("scallops", ["scallops", "us_fried_scallops", "us_baked_or_broiled_scallops"], ["scallops"], grp="seafood")
c("fried_calamari", ["us_fried_calamari", "fried_squid_tw"], ["fried_calamari"], grp="seafood")
c("grilled_salmon", ["us_salmon_fish", "atlantic_salmon_raw"], ["grilled_salmon"], grp="seafood")
c("ceviche", ["us_ceviche"], ["ceviche"], grp="seafood")
c("crab_cake", ["us_cake_crab", "us_crab_cake_sandwich"], ["crab_cakes"], grp="seafood")
c("lobster", ["us_lobster", "us_lobster_salad"], ["lobster_roll_sandwich"], ["Lobster"], generic=True, grp="seafood")
c("crab", ["us_crab", "us_soft_shell_crab"], [], ["Crab"], grp="seafood")
c("shrimp", ["us_shrimp", "us_shrimp_cocktail", "tempura_shrimp"], [], ["Shrimp"], grp="seafood")
# ---- eggs / breakfast
c("pancakes", ["us_pancakes", "us_fruit_pancakes", "us_chocolate_pancakes", "us_pumpkin_pancakes"], ["pancakes"], ["Pancake"], grp="breakfast")
c("waffle", ["waffle", "us_fruit_waffle", "us_cinnamon_waffle", "us_chocolate_waffle", "us_gluten_free_waffle"], ["waffles"], ["Waffle"], grp="breakfast")
c("french_toast", ["us_french_toast", "us_gluten_free_french_toast"], ["french_toast"], grp="breakfast")
c("omelette", ["us_egg_omelet_or_scrambled_egg_with_tomatoes", "us_egg_omelet_or_scrambled_egg_with_cheese_and_tomatoes", "us_scrambled_egg_white_omelet", "tamagoyaki"], ["omelette"], generic=True, grp="breakfast")
c("deviled_eggs", ["us_deviled_egg"], ["deviled_eggs"], grp="breakfast")
c("huevos_rancheros", ["us_huevos_rancheros"], ["huevos_rancheros"], grp="breakfast")
c("egg", ["fried_egg", "boiled_egg", "scrambled_egg_tw", "us_boiled_or_poached_whole_egg"], [], ["Egg"], grp="breakfast")
# ---- desserts / bakery
c("cake", ["us_cake_or_cupcake", "us_chocolate_flourless_cake", "black_forest_cake", "us_pound_cake", "sponge_cake", "us_torte_cake", "us_angel_food_cake", "us_cream_cake"], ["chocolate_cake", "carrot_cake", "red_velvet_cake", "cup_cakes"], ["Cake"], generic=True, grp="dessert")
c("cheesecake", ["cheese_cake_tw", "us_fruit_cheesecake", "us_chocolate_cheesecake"], ["cheesecake"], grp="dessert")
c("tiramisu", ["tiramisu"], ["tiramisu"], grp="dessert")
c("strawberry_shortcake", ["us_strawberry_shortcake_cake"], ["strawberry_shortcake"], grp="dessert")
c("donut", ["us_doughnut", "honey_glazed_donut", "us_doughnut_holes", "us_jelly_doughnut", "us_cake_type_doughnut_with_icing", "us_chocolate_doughnut"], ["donuts"], ["Doughnut"], grp="dessert")
c("churros", ["us_churros"], ["churros"], grp="dessert")
c("beignet", ["us_beignet"], ["beignets"], grp="dessert")
c("baklava", ["us_baklava"], ["baklava"], grp="dessert")
c("bread_pudding", ["us_bread_pudding"], ["bread_pudding"], grp="dessert")
c("creme_brulee", ["us_creme_brulee", "us_custard"], ["creme_brulee"], grp="dessert")
c("mousse", ["us_mousse", "us_chocolate_pudding"], ["chocolate_mousse"], grp="dessert")
c("macarons", ["us_macaroon_cookie"], ["macarons"], generic=True, grp="dessert")
c("apple_pie", ["us_pie", "us_apple_fast_food_pie", "us_berry_pie"], ["apple_pie"], ["Tart"], generic=True, grp="dessert")
c("ice_cream", ["vanilla_ice_cream", "chocolate_ice_cream", "us_ice_cream_cone", "us_ice_cream_sundae", "us_ice_cream_bar", "us_vanilla_gelato", "us_chocolate_gelato"], ["ice_cream"], ["Ice cream"], grp="dessert")
c("frozen_yogurt", ["us_frozen_yogurt", "us_soft_serve_vanilla_frozen_yogurt", "us_frozen_yogurt_cone"], ["frozen_yogurt"], grp="dessert")
c("cookie", ["us_cookie", "us_oatmeal_cookie", "us_brownie_cookie"], [], ["Cookie"], grp="dessert")
c("muffin", ["us_muffin", "us_carrot_muffin", "us_oatmeal_muffin"], [], ["Muffin"], grp="bread")
c("croissant", ["croissant", "us_chocolate_croissant", "us_cheese_croissant"], [], ["Croissant"], grp="bread")
c("bagel", ["us_bagel_with_fruit_other_than_raisins", "us_bagel_with_raisins", "us_pumpernickel_bagel", "us_wheat_bran_bagel"], [], ["Bagel"], grp="bread")
c("pretzel", ["us_soft_pretzels", "us_pretzels", "us_hard_pretzels"], [], ["Pretzel"], grp="bread")
c("bread", ["us_white_or_bread", "us_french_or_vienna_bread", "us_toasted_white_or_bread", "us_italian_grecian_bread"], [], ["Bread"], generic=True, grp="bread")
c("popcorn", ["us_popcorn"], [], ["Popcorn"], grp="snack")
c("candy", ["us_candy", "gummy_candy"], [], ["Candy"], generic=True, grp="snack")
# ---- salads
c("garden_salad", ["us_raw_mixed_salad_greens", "us_tomato_and_or_carrots_lettuce_salad_with_cheese", "us_cheese_lettuce_salad_with_egg", "us_chicken_or_turkey_garden_salad_with_bacon_and_cheese", "us_tomato_lettuce_salad_with_avocado"], ["caesar_salad", "greek_salad"], ["Salad"], generic=True, grp="salad")
c("caprese", ["us_tomato_and_basil_mozzarella_cheese"], ["caprese_salad"], grp="salad")
c("beet_salad", ["us_beets", "us_pickled_beets"], ["beet_salad"], generic=True, grp="salad")
c("seaweed_salad", ["us_pickled_seaweed"], ["seaweed_salad"], generic=True, grp="salad")
# ---- fruit / veg / dairy (Open Images)
for _nm, _oi, _ids in [
    ("apple", "Apple", ["apple"]), ("banana", "Banana", ["banana"]), ("orange", "Orange (fruit)", ["orange"]), ("strawberry", "Strawberry", ["strawberries"]),
    ("grapes", "Grape", ["green_grapes", "grapes", "us_raw_grapes"]), ("watermelon", "Watermelon", ["watermelon"]), ("pineapple", "Pineapple", ["pineapple"]),
    ("mango", "Mango", ["mango"]), ("peach", "Peach", ["peach"]), ("pear", "Pear", ["pear"]), ("lemon", "Lemon (plant)", ["lemon"]),
    ("cantaloupe", "Cantaloupe", ["cantaloupe_melon"]), ("pomegranate", "Pomegranate", ["pomegranate"]),
    ("tomato", "Tomato", ["tomato"]), ("carrot", "Carrot", ["carrot", "us_carrots"]), ("broccoli", "Broccoli", ["broccoli"]),
    ("cabbage", "Cabbage", ["cabbage"]), ("cucumber", "Cucumber", ["cucumber"]), ("zucchini", "Zucchini", ["zucchini"]),
    ("mushroom", "Mushroom", ["us_mushrooms"]), ("potato", "Potato", ["potato", "us_baked_potato"]), ("pumpkin", "Pumpkin", ["pumpkin"]),
    ("bell_pepper", "Bell pepper", ["bell_pepper"]), ("asparagus", "Garden Asparagus", ["asparagus"]), ("artichoke", "Artichoke", ["us_artichoke"]),
    ("cheese", "Cheese", ["us_cheese", "us_cheddar_cheese", "us_swiss_cheese", "us_brie_cheese"]),
    ("coconut", "Coconut", ["coconut_water"]),
]:
    c(_nm, _ids, [], [_oi], grp="produce")
# ---- drinks (Open Images). Only classes that are the drink itself
c("beer", ["beer"], [], ["Beer"], grp="drink")
c("wine", ["red_wine", "white_wine"], [], ["Wine"], grp="drink")
c("juice", ["us_orange_juice_beverage", "us_apple_juice_beverage", "us_fruit_juice_drink"], [], ["Juice"], grp="drink")
c("milk", ["us_milk"], [], ["Milk"], grp="drink")
c("tea", ["black_tea_unsweet", "green_tea_unsweet", "jasmine_tea", "oolong_tea"], [], ["Tea"], grp="drink")
c("coffee", ["latte_coffee", "cappuccino_coffee", "iced_americano", "us_decaffeinated_coffee"], [], ["Coffee (drink)"], grp="drink")

NO_VOCAB = {  # Food-101 classes with no usable vocab entry (not trained; vocab gaps to fix by adding foods)
    "beef_carpaccio": "no entry", "cannoli": "no entry", "eggs_benedict": "no entry", "escargots": "no entry", "foie_gras": "no entry",
    "fish_and_chips": "no entry", "lobster_bisque": "no entry", "poutine": "no entry", "risotto": "no entry", "panna_cotta": "no entry",
    "shrimp_and_grits": "grits only", "tuna_tartare": "no entry", "croque_madame": "no entry", "cheese_plate": "no entry",
}

# ---- NEGATIVE concepts (appended LAST so earlier concept indices stay valid). Open Images validation/test boxes of non-food objects ->
# neg_* rows, so the gate keeps working when food rows get image prototypes. Cups/plates/bowls are NOT used (they may contain food).
def _neg(name, rowid, oi): C[name] = dict(vocab=[rowid], f101=[], oi=list(oi), generic=False, grp="neg")
_neg("n_cutlery", "neg_cutlery", ["Spoon", "Fork", "Knife", "Chopsticks", "Tableware"])
_neg("n_utensil", "neg_utensil", ["Kitchen utensil", "Kitchen knife"])
_neg("n_board", "neg_cutting_board", ["Cutting board"])
_neg("n_table", "neg_table", ["Table", "Kitchen & dining room table", "Chair"])
_neg("n_counter", "neg_counter", ["Countertop"])
_neg("n_sink", "neg_sink", ["Sink"])
_neg("n_fridge", "neg_fridge", ["Refrigerator"])
_neg("n_pan", "neg_frying_pan", ["Frying pan", "Wok"])
_neg("n_stove", "neg_stove", ["Gas stove"])
_neg("n_kettle", "neg_kettle", ["Kettle", "Teapot"])
_neg("n_can", "neg_can", ["Tin can"])
_neg("n_hand", "neg_hand", ["Human hand"])
_neg("n_person", "neg_person", ["Person", "Human face"])
_neg("n_napkin", "neg_napkin", ["Towel"])
_neg("n_plant", "neg_plant", ["Houseplant"])
_neg("n_pet", "neg_pet", ["Cat", "Dog"])
_neg("n_outside", "neg_outside", ["Car", "Building", "Window"])
_neg("n_object", "neg_object", ["Book", "Laptop", "Mobile phone", "Computer keyboard"])
_neg("n_bag", "neg_bag", ["Handbag"])
_neg("n_packaging", "neg_packaging", ["Box"])
