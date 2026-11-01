# Portion from a single photo, plate/bowl as ruler: results

Verdict against the agreed bar (median |error| <= 30% AND p10-p90 coverage >= 75%) on the 60-dish test set: **FAIL**. Coverage passes (80-82%), median error does not (42.6%).
The method does help (median error 48.7% -> 42.6%, Pearson 0.18 -> 0.57, coverage 58% -> 82%), but a 30% median error is out of reach from area alone on this data (see ceiling below).

## Data and protocol
- Nutrition5k overhead RGB (CC BY 4.0), 640x480. 800 dev dishes (kcal >= 30, mass >= 30 g, from the official `rgb_train_ids` list, test ids excluded) in
  `$SP/eval/n5k_dev/`, split 70/30 by `md5(dish id) % 10 < 7` -> fit 567 / validation 233. Test = the 60 dishes in `$SP/eval/n5k/labels.json`, never used for fitting or choices.
- Items come from the shipped pipeline unchanged (locator.onnx rows 0-13 inside `loc17.onnx`, MobileCLIP-S1 crop classifier, shipped gate and mixture k3/cum0.7).
- Scripts: `feat.py` (items + container candidates), `prep.py` (ellipse + areas), `fit.py` (variant table), `select_params.py` (bowl diameter, sigma), `final_eval.py`, `demo.py`.
- Test-set protocol caveat: the test set was evaluated once with frozen parameters (v1: per-image single container). Afterwards I changed the code so each item uses the
  container that holds its box (needed for multi-plate phone photos), refit on dev (no material change on dev) and re-ran the test as a disclosed second number (v2, below).
  The v1 run is the official PASS/FAIL number; v2 is what `portion_params.json` and `portion.py` now contain. They are within 0.1 point on the median.

## Final: 60-dish test set
7 of 60 dishes log no item at all in the shipped pipeline (gate); for every method those dishes get the fit-split median (249 kcal, range 64-562) so the table covers all 60.
Constants use the same 64-562 range, which has 80% coverage on the fit split by construction.

| method | median abs err | within +-25% | p10-p90 coverage | Pearson (raw) | Pearson (log) | median p90/p10 |
|---|---|---|---|---|---|---|
| **portion (plate ruler), v1 frozen, official** | **42.6%** | 30.0% | **81.7%** | 0.57 | 0.62 | 5.6x |
| portion (plate ruler), v2 code now in repo | 42.7% | 28.3% | 80.0% | 0.56 | 0.62 | 5.6x |
| (i) current typical-portion pipeline | 48.7% | 25.0% | 58.3% | 0.18 | 0.22 | 4.4x |
| (ii) constant 350 kcal | 49.7% | 30.0% | 73.3% | n/a | n/a | 8.8x |
| (iii) constant = fit-split median (249 kcal) | 58.7% | 13.3% | 73.3% | n/a | n/a | 8.8x |

Logged-only (53 dishes with at least one item): portion 42.5%, typical 48.0%, const 350 43.8%, const fit-median 56.9%. Note that on 53 dishes the portion method is
no better than "350 kcal for everything" on the median error; its gain is the correlation (it actually ranks dishes) and the calibrated range. n = 60: one dish moves 1.7 points.
The sampling error of a median at n = 60 is several points (not computed), so 48.7 vs 42.6 is not a statistically firm difference; correlation 0.18 vs 0.57 is.

## Validation split (233 dishes, sigma calibrated here; v2)
| method | median abs err | within +-25% | coverage | Pearson / log |
|---|---|---|---|---|
| portion | 38.2% | 33.0% | 83.3% | 0.57 / 0.57 |
| typical portion | 54.7% | 21.0% | 55.4% | 0.18 / 0.25 |
| const 350 | 45.7% | 32.6% | 78.5% | n/a |
| const fit-median | 47.0% | 24.0% | 78.5% | n/a |
The validation number is optimistic by about 4 points versus test (38.2 vs 42.7): parameters (D_bowl, sigma, variant) were chosen on it.

## Choices (all on fit/validation, never on test)
Variant table (area = food pixels, D plate 26 / bowl 16, container logic v1; median |err| fit | val; all include the fallback constant for no-item dishes):

| model | params | fit | val |
|---|---|---|---|
| global k, beta = 1 | 1 | 46.7 | 48.5 |
| global k, beta = 1, + density gamma | 2 | 43.0 | 41.1 |
| k per group (8), beta = 1 | 8 | 45.3 | 47.5 |
| k per group + gamma | 9 | 41.0 | 41.5 |
| global k, beta, gamma | 3 | 39.0 | **37.5** |
| k, beta per group + gamma | 17 | 37.8 | 40.1 (overfits: val Pearson 0.38) |
Chosen: global k, beta, gamma (3 parameters): k = 1.567, beta = 0.893, gamma = 0.59. Per-group k/beta (8 groups from vocab category plus name regexes for fried/nuts, `portion.group_of`)
did not help on validation, so no group table is proposed. The food-group idea is therefore NOT supported by this data (it may matter for non-Nutrition5k foods; untestable here).
gamma < 1 means "trust the identified food's kcal/100 g only about 60%": Nutrition5k kcal/g has p10-p90 0.53-2.30 and the name-based value correlates only 0.55 (log) with it.

Area source (3 modes, gbeta+gamma, val median |err|): food pixels not plate-coloured **37.5%**, item box clipped to plate 47.6%, YOLOE-seg instance mask clipped to plate 51.5%.
The shipped locator's masks cover whole plates (the box is "plate of food"), so they carry no portion information. Thresholds: Lab distance 16-22 equal (corr of log pixels with log mass
0.54 / 0.50, 10 gives 0.49, 40 gives 0.22); 18 chosen. Ring 0.70-0.97 vs 0.80-0.97 no difference.

Container diameters (fit split): the plate diameter is not identifiable (k absorbs it; fit median err 36.9-39.4% across plate 20-30 cm), so it is FIXED at 26 cm (standard dinner plate, physically plausible),
and only the bowl is fit relative to it: bowl 12/15/18/21/24/27 cm -> 42.0/39.8/39.0/37.7/**36.7**/37.4%. Chosen bowl = 24 cm. That is on the large side for a bowl rim but Nutrition5k "bowls" are big
salad/pasta bowls; a phone user with a 14 cm rice bowl would be wrong by 1.7x in linear scale (3x in area, softened to ~2.7x by beta 0.89). Measured plate major axis in the image: median 494 px of 640
(p5-p95 367-562 px, CV 15%; bowl 419 px, CV 23%), so real plates are not all one size even in this single canteen (about +-15% in diameter, +-30% in area).

Sigma: items independent, log-sd 0.65 on grams (and no shared factor). Validation coverage at sigma 0.2...1.0: 0.49 0.58 0.68 0.75 0.79 (0.6) 0.85 (0.7) 0.87 0.91 0.91. 0.65 chosen for about 80%.
The resulting range is 5.6x wide (p90/p10) (the constant baseline is 8.8x), which is honest but not tight: a range of 120-670 kcal for a 300 kcal dish.

## Why it cannot reach 30% on this data: ceiling analysis (dev, 717 dishes with items)
- Even with the TRUE mass: true mass x median dish density (1.18 kcal/g) gives median error 33.5% (within 25%: 40%); true mass x the pipeline's kcal/100 g gives 37.7%. Dish density varies 0.53-2.30 kcal/g (p10-p90).
- Food area explains mass only partly: corr(log food px, log mass) = 0.48-0.54, corr with log kcal 0.39. Height/thickness is invisible from above.
- Our 37.5-42.7% is close to that 33.5% oracle. A 30% median error would need density information that a single overhead photo plus a food name does not provide here.
- Real-scale check: replacing the plate ruler with the true fixed-camera scale (every image given the same cm/px) gives 43.0% (val) vs 37.7% with the plate ruler. So on Nutrition5k the plate ruler helps
  not because it measures centimetres better (it measures them worse) but because plate size correlates with how much food the canteen served. On a phone that second effect may or may not exist.
  Treat the 38-43% as "relative plate-fill signal", not as validation of absolute scale.

## Failure cases (test set, v1; true kcal -> predicted)
Worst relative errors are all LOW-calorie dishes predicted ~230 kcal (the model regresses toward the typical dish, and the name-based density is overestimated):
- dish_1561574600: eggplant 100 g, 33 kcal -> 234 (item named "fried eggplant"). +610%.
- dish_1568666263: lettuce/broccoli/artichoke 348 g, 93 kcal -> 429 (item "broccoli salad with cauliflower", p 0.73; large bowl, area big). +359%.
- dish_1558640979: cantaloupe/cauliflower/carrot 169 g, 59 kcal -> 192 ("boiled sweet potato"). +224%.
- dish_1558115825: grapes 62 g, 43 kcal -> 131 ("cherry tomatoes"; bowl measured 24 cm). +207%.
- dish_1565981848: carrot/herbs/tofu dish 198 g, 147 kcal -> 328 ("vegetarian fillet"). +122%.
- dish_1563475788: pineapple 78 g, 39 kcal -> 82 ("dried pineapple"; dried-fruit density 3.0 kcal/g shrunk by gamma only partly). +110%.
- No-item dishes (fallback 249 kcal): dish_1565711296 (cantaloupe+grapes, 43 kcal: +473%), dish_1559240908 (orange, 60: +313%), dish_1558640450 (apple, 80: +210%), dish_1560270630 (pineapple/watermelon/sweet potato, 121: +106%).
  Pattern: small fruit/vegetable portions that the classifier gates out or names as dense foods. Fruit dishes are the weak spot of both identification and density.
Validation worst cases are the same pattern (dish_1559843492 broccoli+strawberries 32 kcal; dish_1558722322 carrot 38 kcal read as "dried apricot", 36k px food, pred 266).
The user's almond example (50 g -> 718 kcal) is NOT in this test set; almonds map to "sweet" group density by name regex in `group_of` but the final model has no per-group table, so the fix for it is the
gamma shrink (kcal/g of almonds, 6 kcal/g, shrinks toward 1.2: about 3 kcal/g) plus area (a handful of nuts covers little plate). Not measured; do not claim it.

## Transfer sanity check on phone-like photos (qualitative, `$SP/research/portion/wiki_sheet.jpg`, `wiki_out.txt`)
20 Wikimedia Commons photos (restaurant tables, angled, multi-plate; searches in `$SP/research/portion/wiki.py`; metadata and licences in `$SP/eval/wiki_meals/meta.json`) and the user's 10 negatives.
- Negatives: 0/10 produce an item (portion step never runs), as before.
- Wikimedia: 18/20 photos log at least one item; 2 log none (a menu page, one with no gate-passing item). 29 of the 33 logged items were matched to a container (the other 4 fall back to typical portions).
- Where it works: single dish on a plate/bowl seen mostly from above or ~45 degrees (Japanese curry rice 233 g and 342 g, beef burgundy 312 g, pizza on a plate 269 g, kung pao chicken 315 g): ellipse sits on the rim, food mask is clean, grams are
  plausible within about +-35% by eye.
- Where it breaks: (1) tilted/partial containers give wrong ellipses (tall ellipses on obliquely seen plates, a plate cut by the frame measured at 117% of the image long edge, a bowl measured from a cropped mask),
  and on trays the mask ellipse spans a whole tray; (2) small side dishes (soy sauce dish 277 g predicted, rice wine 125 g, sushi 357 g, waffle 330 g, scrambled eggs 331 g) are implausible because the 26 cm assumption inflates
  small plates and the food-pixel step counts any non-plate colour (table, cutlery, patterned plate: the striped bowl is partly counted as food); (3) many items on one table are assigned the nearest container only by box overlap.
  By eye roughly a third of the grams look plausible; this was not scored carefully and no ground truth exists for these photos, so no number is claimed.
- Limits of the Nutrition5k conclusion: one fixed overhead camera, a few canteen plate types (white round, square, paper, bowls), single-plate dishes, food mostly one item (682 of 800 dishes have one item),
  no oblique views, constant lighting. Nothing here validates the tilt correction, the ellipse fit on angled plates, patterned plates, multiple plates, or plate-size assumptions on real phone photos.

## Honest summary
- Container-as-ruler plus plate-fill food area gives a real signal (corr 0.57 vs 0.18) and a properly calibrated wide range (80%), but median error stays at 43% on held-out data: FAIL vs the 30% bar, and
  statistically indistinguishable from "always 350 kcal" on the median alone (42.5 vs 43.8% on dishes where an item was logged).
- The gain on Nutrition5k comes at least partly from plate size correlating with served portion, not from correct centimetres.
- Recommended before any port: collect 100+ real phone meal photos with weighed food and a known plate diameter and re-run `fit.py`/`final_eval.py` on them (the scripts take any `feat_*.pkl`).
  Cheapest product win in the meantime is probably the wider calibrated range plus a user plate-size setting, not the point estimate.

## Reproduce
```
python export_portion_loc.py                       # loc17.onnx (research locator = shipped 14 prompts + plate, bowl, tray)
python feat.py dev $SP/eval/n5k_dev ; python feat.py test $SP/eval/n5k
python prep.py dev $SP/eval/n5k_dev ; python prep.py test $SP/eval/n5k
python fit.py colour ; python select_params.py gbeta 1      # then set sigma_item 0.65 in portion_params.json by hand
python final_eval.py val ; python final_eval.py test          # test: once
python feat.py wiki $SP/eval/wiki_meals; python prep.py wiki ...; python demo.py wiki $SP/eval/wiki_meals out.jpg
```
Intermediate files (features, analysis caches, logs, sheets, v1 params) are in `$SP/research/portion/`.
