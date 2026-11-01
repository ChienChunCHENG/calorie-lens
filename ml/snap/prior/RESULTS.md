# Commonness prior + clean display names: results

Chosen params (`params.json`): lambda 0.5, temper 1.0, alpha 20, counts = representative FNDDS code only (`agg: rep`), gate computed on the prior-free softmax.
Files: `out/vocab_prior.json` (2,322 rows aligned with vocab.json: `prior`, `lp`, optional `display` on 788 rows), `build_prior.py`, `prior_lib.py`, `names.py`, `tune_prior.py`, `report.py`, `cache_embeds.py`, `build_members.py`.

## Method
- Counts: NHANES 2021-2023 WWEIA individual-food files DR1IFF_L + DR2IFF_L (CDC, public domain, ~188k food lines; unweighted number of reports per FNDDS food code).
  The 1,735 FNDDS rows map via `#code` in `source`; 243 of them have 0 reports. `build_members.py` re-runs the vocab dedup to recover which codes were merged into each row; `agg=sum` (all merged codes), `rep` (representative code only) and `mix` were compared, `rep` won narrowly on OI-val.
- Non-FNDDS rows (Taiwan FDA / estimated, 343): default count 4 (FNDDS median is 10). 200 non-FNDDS rows that are common in the US/EU (apple, banana, ketchup, croissant, sushi, ramen, hummus, ...) get a hand tier floor A/B/C = 800/250/80 reports (lists in `prior_lib.py`).
- European boost: 85 vocab rows matched by 70 hand patterns get a floor of 250 reports (paella, bratwurst, pierogi, quiche, crepe, lasagna, ravioli, gnocchi, focaccia, brioche, danish, strudel, gelato, tiramisu, brie, feta, ...).
  **Not in the vocab (no row, not added):** schnitzel, risotto, fish and chips, moussaka, muesli, carbonara, ciabatta, goulash, cassoulet, panna cotta, smoked salmon, sausage roll, cottage pie, yorkshire pudding, tzatziki, caesar salad, mayonnaise. (Note: `us_french_or_vienna_bread` stands in for baguette; "french fries" exists only as the cluster row `us_home_fries_potato`, displayed "French fries".)
- prior_i = (count_i + alpha) / (sum + alpha N) over the 2,278 food rows. Neg rows: prior 0, log term 0.
- Tempering: prior^t renormalised equals multiplying the log prior by t, so it is exactly redundant with lambda (lambda_eff = lambda*t). We keep t=1 and tune lambda only.
- Gate: the neutral-neg-term variant was measured and is NOT safe (below). The gate (p_neg, p_food_min) therefore uses the prior-free softmax, so empty-cup behaviour is identical by construction; the prior only reorders food rows.

## Tuning protocol
Open Images meals+multi GT boxes cropped (pad 0.2, shipped clip_image.onnx), split 50/50 by md5(image id): val (tuned on) / test. Hit = predicted row accepted by an OI-class -> regex/category mapping (`tune_prior.py`). "specific" = Hamburger, Pizza, Egg, Cake, Tomato... ; "all" adds generic classes (Pastry, Snack, Fruit, Dessert, Fast food, ...) judged by category membership. Grid: agg x alpha {1,5,20} x lambda {0.25..8}. Food-101 (300 imgs, 60 classes; 275 imgs in classes with an accepted row after extending FOOD101_MAP with regexes for ~25 more classes) is reported only, not tuned on.
Honest note: OI-val gain is small (+3 pts top-1) and not reproduced on OI-test (-0.6 all, -0.3 specific; within noise, n=655/345). Food-101 improves because the added accepted rows include obscure-vs-common competitors. Larger lambda (>=1.5) hurts accuracy everywhere; lambda 0.5 is the plateau 0.4-0.6.

## Numbers (before = prior off, after = lambda 0.5)
| set (n) | top-1 before | top-1 after | top-3 before | top-3 after |
|---|---|---|---|---|
| OI-val all mapped crops (859) | 49.7 | 52.8 | 64.4 | 65.4 |
| OI-val specific classes (472) | 54.0 | 57.0 | 68.4 | 70.5 |
| OI-test all (655) | 53.3 | 52.7 | 64.4 | 65.0 |
| OI-test specific (345) | 62.3 | 62.0 | 73.0 | 74.5 |
| Food-101 mappable (275) | 52.7 | 57.1 | 73.1 | 73.8 |

| images logging anything | before | after (gate raw) |
|---|---|---|
| user negatives (10) | 0/10 | 0/10 |
| uxset negatives (80) | 12/80 | 12/80 |
| extra negatives (30) | 1/30 | 1/30 |
| drinks (39) | 39/39 | 39/39 |

If instead the gate used the prior-adjusted softmax with neutral (0) neg terms: neg_user 0/10, neg_ux 13/80, neg_extra 2/30, drinks 39/39

| image | region | before (en, top-3) | after (display, top-3) |
|---|---|---|---|
| blurry_burger | box [17, 2, 959, 700] | hamburger; cheeseburger; hamburger slider | hamburger; cheeseburger; hamburger slider |
| burger_fries | box [797, 1, 960, 231] | root beer soft drink; cola soda; tamarind drink | Root beer; Cola; tamarind drink |
| burger_fries | box [98, 66, 425, 153] | turkey or chicken burger; hamburger; hamburger slider | Turkey burger; Hamburger bun; hamburger |
| burger_fries | box [66, 164, 954, 717] | turkey or chicken burger; hamburger; cheeseburger | Turkey burger; hamburger; cheeseburger |
| burger_fries | box [672, 3, 836, 78] | turkey or chicken burger; chicken deli sandwich or sub with cheese restaurant; cheese sandwich | Turkey burger; Camembert; chicken deli sandwich or sub with cheese restaurant |
| burger_fries | box [535, 0, 668, 184] | seafood sauce; other flavors tortilla chips; frito pie | Ketchup; Potato chips; Tortilla chips |
| cake_slice | box [97, 124, 906, 577] | tiramisu; torte cake; chocolate cheesecake | Tiramisu; Black forest cake; torte cake |
| croissant_coffee | box [284, 882, 415, 1017] | scone with fruit; scone; breakfast pastry | Scone with fruit; Scone; breakfast pastry |
| croissant_coffee | box [311, 873, 530, 930] | breakfast pastry; berry strudel; cherry strudel | breakfast pastry; Apple strudel; berry strudel |
| dark_steak | box [6, 308, 232, 558] | grape juice drink; sake; 100% grape juice | Sake; grape juice drink; grape juice |
| dark_steak | box [215, 2, 375, 264] | ginger ale soft drink; iced tea / lemonade juice drink; sugar cane beverage | Ginger ale; Iced tea lemonade; Lemon drink |
| dark_steak | box [298, 195, 549, 312] | corn beverage; corn relish; creamed corn | Sweet corn; Corn; corn beverage |
| dark_steak | box [235, 318, 806, 630] | beef steak; steak; steak teriyaki | Beef steak; Steak; Ground beef |
| dark_steak | box [548, 153, 743, 297] | hamburger bun; knish; dinner roll | Hamburger bun; Dinner roll; knish |
| falafel_hummus | box [0, 16, 959, 650] | falafel; hummus; falafel sandwich | Falafel; Hummus; falafel sandwich |
| fish_chips | box [185, 159, 799, 676] | haddock fish; whiting fish; seafood turnover | haddock fish; whiting fish; cod fish |
| oatmeal | box [532, 144, 907, 526] | dumplings and vegetables beef; meat-filled tortellini with tomato sauce; boiled pork dumplings | dumplings and vegetables beef; Tortellini; boiled pork dumplings |
| oatmeal | box [86, 61, 503, 481] | pan-fried pork bun (sheng jian bao); boiled pork dumplings (shui jiao); boiled pork dumplings | pan-fried pork bun (sheng jian bao); boiled pork dumplings (shui jiao); boiled pork dumplings |
| paella | box [176, 29, 795, 644] | paella; paella with seafood; yellow rice | Paella; Paella with seafood; Couscous |
| pasta | box [21, 24, 947, 717] | pasta with sauce; spaghetti with meat sauce; meat with tomato-based sauce | Spaghetti bolognese; pasta with sauce; Pasta |
| pizza_slice | box [451, 61, 907, 718] | pizza; pizza with pepperoni; pizza with meat other than pepperoni | pizza; Pepperoni pizza; Meat pizza |
| pizza_slice | box [28, 114, 475, 621] | cheese-filled manicotti with meat sauce; eggplant with cheese and tomato sauce; pizza | cheese-filled manicotti with meat sauce; Lasagna; Lasagna |
| salad | box [147, 167, 846, 586] | ramen bowl; cream of wheat; hominy | ramen bowl; Queso fresco; hominy |
| sandwich | box [486, 0, 800, 129] | wild pig; tuna with cream or white sauce; trail mix | wild pig; Chewing gum; Yogurt tube |
| sandwich | box [0, 112, 958, 718] | sardine sandwich; reuben sandwich; grilled cheese sandwich | sardine sandwich; Grilled cheese sandwich; Turkey sandwich |
| sandwich | box [204, 63, 418, 163] | almond butter sandwich; french toast; toasted soy bread | French toast; almond butter sandwich; toasted soy bread |
| schnitzel | box [0, 3, 789, 632] | haddock fish; giant Taiwanese fried chicken cutlet (ji pai); flounder fish | Fried chicken cutlet; haddock fish; Pork cutlet |
| snacks | box [0, 181, 951, 1194] | nori seaweed; pickled seaweed; rice chips | nori seaweed; pickled seaweed; rice chips |
| steak | box [5, 307, 232, 558] | sake; grape juice drink; 5% glucose in water fluid replacement | Sake; grape juice drink; Sports drink |
| steak | box [216, 3, 377, 268] | sugar cane beverage; winter melon tea; soju | Ginger ale; Soju; winter melon tea |
| steak | box [297, 195, 548, 312] | seafood sauce; barbecue beef with sauce; sha cha sauce | Barbecue pork with sauce; seafood sauce; barbecue beef with sauce |
| steak | box [234, 318, 803, 638] | beef steak; steak; steak teriyaki | Beef steak; Steak; steak teriyaki |
| steak | box [547, 153, 743, 298] | knish; meat / poultry-filled pastry; pan-fried pork meat pie | knish; meat / poultry-filled pastry; Dinner roll |
| sushi | box [705, 66, 857, 222] | sake; barley tea; black soybean tea | Sake; Hot tea; Oolong tea |
| sushi | box [616, 69, 689, 148] | brewed coffee; espresso decaffeinated coffee; beer | Black coffee; Beer; Latte |
| sushi | box [244, 207, 766, 535] | nigiri sushi; sushi; assorted sashimi | Sushi (nigiri); Sashimi; sushi |
| sushi | box [580, 133, 664, 208] | soy sauce; sake; black soybean tea | Soy sauce; Sake; Black coffee |
| tacos | box [144, 44, 215, 182] | white wine; red wine; beer | White wine; Red wine; Beer |
| tacos | box [0, 199, 958, 631] | beef or pork fajita; beef taco filling: beef; chicken fajita | Fajita; Fajita; beef taco filling: beef |
| tacos | box [782, 0, 959, 250] | apple juice beverage; fruit juice drink; corn beverage | Beer; Olive oil; White wine |
| tacos | box [225, 52, 319, 156] | beer; hamburger; root beer soft drink | Beer; Ketchup; Root beer |
| us_breakfast | box [0, 56, 958, 717] | sausage griddle/pancake sandwich; gluten free pancakes; pancakes | Pancakes; Gluten free pancakes; sausage griddle/pancake sandwich |
| western_breakfast | box [85, 40, 861, 727] | baked beans from fast food / restaurant; pork hash; beans and franks | Baked beans; Baked beans; pork hash |

## Reading the examples
Fixed: steak plate: "sugar cane beverage" -> "Ginger ale" (still a wrong drink box, but the steak box is "Beef steak"), "seafood sauce" drops to 2nd behind "Barbecue pork with sauce" (steak) / "Ketchup" (burger_fries), "corn relish" -> "Sweet corn", "knish" is still top-1 on steak.jpg (but "Hamburger bun" beats it on dark_steak.jpg); fries box "root beer" stays top-1 (the crop is a cup of cola/root beer; both are drinks), pasta -> "Spaghetti bolognese" first.
Not fixed / broken: sushi-photo tea and sake boxes are locator crops of bottles; the prior reshuffles beverages but cannot recognise them. `fish_chips` stays "haddock fish". `schnitzel` -> "Fried chicken cutlet" (no schnitzel row). `sandwich` side crop "wild pig" is untouched (prior 0.5 is too weak to remove it; lowering such rows needs a vocab prune, which is out of scope).
Display names: "Pizza", "Ketchup", "Black coffee", "French fries", "Beef steak", "Cheeseburger", ... on 788 rows (rules in `names.py` + manual map); other rows keep `en`. Mixed capitalisation in a UI is expected for rows without display (they are lowercase `en`).

## Caveats
- Prior is US consumption frequency, not "how often photographed"; foods rarely eaten but often photographed (sushi, tiramisu) rely on the hand tiers.
- Taiwan dishes are down-weighted (count 4); for the Taiwanese set this lowers them, which is intended for the US/EU product.
- The best lever found while reviewing names is not the prior: many FNDDS cluster `en` strings are misleading ("baked banana", "evaporated milk", "home fries potato") and these are the text the CLIP embedding is built from. Re-embedding `display` (or a cleaned `en`) for the 788 rows is untested here and likely worth a run.
- Proto (image prototype) combination untested; the prior is an additive term, so use score = ls*cos + proto_term + lambda*lp.


# Follow-up: 36 new vocab rows + re-embedded display names

Candidate web files (not copied into web/): `out/vocab.json` (2,358 rows = 2,322 old rows unchanged + 36 appended; every row has `prior` and `lp`, 824 have `display`; new rows without an FNDDS value carry `"est": true`), `out/vocab_emb.bin` (2,358 x 512 fp16 = 2.41 MB, **variant A: en text, current recipe**), `out/new_rows_sources.json` (per-row source note). Rebuild: `python build_prior.py; python extend.py`. Recipe check: re-encoding old `en` strings reproduces the shipped rows (cos 0.99995-1.00003), so old rows are byte-for-byte the shipped embeddings and new rows use the identical `"a photo of {}."` / MobileCLIP-S1 datacompdr / L2-normalised recipe.

## New rows (36)
schnitzel, risotto, fish and chips, moussaka, muesli, spaghetti carbonara, ciabatta, goulash, cassoulet, panna cotta, smoked salmon, sausage roll, cottage pie, yorkshire pudding, tzatziki, caesar salad, mayonnaise (the 17 requested) plus French fries (the old vocab only had a "home fries potato" cluster), apple pie (only "apple pie filling" existed), guacamole, salsa, Greek salad, Cobb salad, fettuccine Alfredo, eggs Benedict, shakshuka, caprese salad, pasta salad, baguette, bagel with cream cheese, BLT sandwich, spaghetti and meatballs, avocado toast, chicken tikka masala, shawarma, croque monsieur.
Provenance (details per row in `newrows.py` / `new_rows_sources.json`). Only the offline FNDDS 2021-2023 table was available, not a live FoodData Central query, and SR Legacy was not consulted live:
- FNDDS value copied (12): smoked salmon, cottage pie, tzatziki, mayonnaise, French fries, guacamole, salsa, pasta salad, baguette, apple pie, BLT, moussaka (FNDDS "Eggplant and meat casserole"; 96 kcal/100 g is probably low versus a restaurant moussaka).
- FNDDS proxy: ciabatta (Bread, Italian), goulash (Stew, beef), Yorkshire pudding (Popover).
- Composed from FNDDS components: fish and chips, Caesar salad, Greek salad, Cobb salad, bagel with cream cheese.
- Estimated (`est: true`, no FNDDS entry under that name; typical recipe composition, same convention as the 204 existing estimated rows): schnitzel, risotto, muesli (SR Legacy recalled 340-370, not re-fetched), carbonara, cassoulet, panna cotta, sausage roll, Alfredo, eggs Benedict, shakshuka, caprese, spaghetti and meatballs, avocado toast, tikka masala, shawarma, croque monsieur.
Portions are FNDDS weights where the entry has a matching one, otherwise typical servings. Prior floors: French fries and apple pie tier A (800 reports), the rest B (250), cassoulet/shakshuka/shawarma C (80). The old cluster row `us_home_fries_potato` is now displayed "Home fries" (the new row is "French fries").

## Text re-embedding experiment
A = `en` (current), B = "a photo of {display}." for the 824 rows with display (others keep en), C = L2-normalised mean of the A and B embeddings. Mean cos(en, display) = 0.959. lambda = 0.5 prior from `lp`; gate unchanged (prior-free). Top-1 / top-3 %, OI-test and Food-101 are not tuned on; "old" = baseline vocab (2,278 rows, shipped text).

| config | OI-val all | OI-test all | OI-test specific | Food-101 old classes (275) | Food-101 all incl. new-row classes |
|---|---|---|---|---|---|
| text only, old vocab (current) | 49.7 / 64.4 | 53.3 / 64.4 | 62.3 / 73.0 | 52.7 / 73.1 | 52.7 / 73.1 |
| A text only + 36 new rows | 48.4 / 64.0 | 53.6 / 64.1 | 62.6 / 72.8 | 50.5 / 71.3 | 57.1 / 75.7 |
| A + prior | 51.6 / 65.7 | 53.0 / 64.7 | 62.3 / 74.5 | 54.2 / 71.6 | 61.8 / 76.8 |
| B display-only text, no prior | 49.6 / 63.8 | 54.0 / 62.6 | 62.0 / 71.6 | 48.7 / 68.7 | 55.7 / 73.6 |
| B + prior | 51.7 / 65.2 | 53.6 / 64.6 | 62.6 / 75.1 | 52.7 / 69.1 | 60.7 / 74.6 |
| C avg text, no prior | 48.5 / 63.6 | 53.9 / 61.8 | 62.0 / 71.0 | 47.3 / 70.9 | 54.3 / 75.4 |
| C + prior | 52.4 / 64.5 | 53.4 / 63.5 | 62.6 / 74.5 | 52.0 / 70.9 | 60.0 / 76.4 |

Findings: re-embedding the clean names does not help. Display-only text lowers Food-101 on the old classes by 4 points (no prior) and the averaged embedding is within noise of A on OI-test while costing top-3 (61.8 vs 64.4). The 36 new rows slightly cost old-class accuracy at lambda 0 (they are extra distractors, Food-101 old -2.2) and the prior recovers it (+3.7 at 0.5). Recommendation: ship variant A (en text) with the prior; use `display` for the label only. A prior-only change gives OI-test all 53.0 vs 53.3 text-only (flat); OI-test specific top-3 +1.5; Food-101 +9 (61.8 vs 52.7, partly because apple pie, fish and chips, risotto, croque monsieur etc. now exist: on old classes only, +1.5 top-1).

## Negatives (gate counts of images that log anything; gate = prior-free softmax over the full 2,358-row matrix)
| set | current | A + 36 rows (+prior) | B | C |
|---|---|---|---|---|
| user negatives (10) | 0 | 0 | 0 | 0 |
| uxset negatives (80) | 12 | 12 | 12 | 11 |
| extra negatives (30) | 1 | 1 | 1 | 1 |
| drinks (39) | 39 | 39 | 39 | 39 |
The shipped A candidate leaves every gate count identical; the new rows do not capture any empty-cup/wok/glass crop.

## Spot checks (ux images, A + prior 0.5, display names)
fish_chips: "Fish and chips" (was haddock fish); schnitzel: "Schnitzel" (was a Taiwanese cutlet); burger_fries main crop: "French fries; Turkey burger"; steak plate: "Beef steak"; pasta: "Spaghetti bolognese; pasta with sauce; Spaghetti and meatballs"; cake: "Tiramisu; Black forest cake". Still wrong: steak-photo drink and bun crops ("Sake", "Ginger ale", "knish"), salad photo ("ramen bowl"), sandwich side crop ("wild pig"): those are recognition errors the prior cannot fix; the vocab still contains odd rows (wild pig, knish) that could be pruned.
