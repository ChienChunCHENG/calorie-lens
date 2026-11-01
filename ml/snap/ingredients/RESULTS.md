# Ingredient/component-level calorie estimate vs dish-level: measured results

Question: does recognising components (rice + broccoli + chicken, like a MyFitnessPal log) beat dish-level recognition for kcal from one photo, fully on-device?
Short answer: **No, not enough.** The shippable variant misses the agreed bar (median |err| 35.5% vs <= 30%; coverage 78% passes). Most of the gain over the shipped dish-level pipeline comes from measuring *area* (portion), not from naming ingredients.

## Setup (what was actually built)
- Segmentation: FastSAM-s class-agnostic masks (imgsz 640, conf 0.3, iou 0.8) inside a plate/bowl region from YOLOE-11s prompts `plate, bowl, tray` (convex hull of its mask, because the mask has holes where food sits). Max 14 masks, IoU>0.85 dupes dropped.
- Labelling: each mask -> MobileCLIP-S1 (shipped `clip_image.onnx`) on two crops (masked-crop and context-crop, averaged 50/50), zero-shot against `ingredients_vocab.json` (87 components, 95 text rows incl. 8 background rows). Soft labels (full softmax) are kept.
- Vocabulary: 87 canonical components. 71 are built by explicit merge of Nutrition5k ingredient names (the 554-name list in ingredients_metadata.csv, non-food/spice names left out) (mapping in `ingr_vocab.py`; kcal/g = gram-weighted mean over the FIT dishes); 16 hand-added Taiwanese/Asian components and mixed dishes (braised pork, fried tofu, tea egg, noodle soup, bubble tea, ...). Unmapped Nutrition5k kcal (spices, milk, cream, "deprecated") = 0.6% of dev kcal.
- Portion: pixels are painted smallest-mask-wins, area_k = sum of soft-assigned area as a fraction of the plate hull; grams_k = K[group] * area_k (b fixed 1); kcal = sum grams_k * kcal/g_k + oil allowance. Fitted parameters: 17 group constants + 1 oil constant (33 kcal/dish; oil is 7.6% of all dev kcal and invisible). Fit with a Huber loss on ln(pred/true), ridge-tied to the mean K.
- Interval: empirical q10/q90 of ln(true/pred) on validation = [-0.79, +0.76], i.e. x0.45 .. x2.1. Honest 2-fold check of that calibration on dev-val: 68% / 86% coverage (mean 77%).
- Data: 420 dev dishes (md5 split: 277 fit / 143 val, none in the test set), 60 test dishes touched once. The dev cache uses every second dev dish (time); caches are in `$SP/research/ingr/seg_*.pkl`.

## Dev validation (143 dishes; median |err| / true kcal)
| variant | val median err |
|---|---|
| dish-level whole image, top-1 typical portion (shipped style) | 52.3% |
| constant (fit median / 350) | 49.1% / 44.4% |
| area only, no labels: kcal = K*area^b + c | 40.2% |
| **component model (shipped config: plate-normalised, soft, b=1)** | **39.2%** (within 25%: 31%, r=0.60) |
| same, area normalised by image instead of plate (fixed camera only) | 37.7% (r=0.66) |
| ablations of the component model (plate norm): hard labels 43.1%, only masked crop 41.0%, only context crop 40.8%, temp 0.5 42.3%, one global K (ungrouped) 41.5%, no oil allowance 43.7% (b fitted ~0.73 gives 39.5%) | |

Findings:
1. Labels add only ~1 point over a label-free area model (40.2 -> 39.2). Naming ingredients mostly fixes the kcal/g (rice 1.3 vs broccoli 0.35 vs bacon 5), which matters less than the mass error.
2. Oracle bound: with the TRUE dish mass and the median density, median error is still 33.5% (dev). With true mass and the predicted composition density: 29.7%. So <= 30% is unreachable unless the mass estimate is much better than ours. Our area->mass estimate has median error 37% and log-corr only 0.40 with true mass (0.49 with absolute image area).
3. The hidden oil allowance helps (43.7% -> 39.2%). Oil is 7.6% of dev calories and is never visible.
4. Recognition itself is decent: kcal-weighted ingredient recall (component area >= 0.4% of plate) = 82% on dev-val, 81% on test; precision is poor (24-28%: many spurious small components, mostly visually similar siblings: sweet_potato/potato, avocado/cucumber, apple/pear).
5. Plate size is not constant in Nutrition5k (square/round/red/bowls), so "plate as a ruler" with one assumed diameter hurts; absolute pixel area from the fixed camera is better (dev r 0.66 vs 0.60). That scale is not available on a phone.

## FINAL one-time test: 60 Nutrition5k dishes (params frozen on dev)
| method | median err | within +-25% | p10-p90 coverage | Pearson r | median err, true >= 300 kcal (n=28) |
|---|---|---|---|---|---|
| shipped dish-level (whole image, top-1 typical; my re-implementation on the same embedding) | 49.4% | 23% | n/a | 0.35 | 49.4% |
| constant 350 | 49.7% | 30% | n/a | -0.21 | **18.3%** |
| **ingredient-level, plate-normalised (shippable)** | **35.5%** | 38% | **78%** | 0.65 | 26.6% (within 25%: 46%, coverage 86%) |
| ingredient-level, image-normalised (fixed camera, NOT shippable) | 28.3% | 40% | 78% | 0.71 | 26.2% |
| plate-ruler agent (ml/snap/portion) | not available when this was written | | | | |

The lead's number for the shipped pipeline on the same 60 dishes is 48% median / 58% coverage; my whole-image re-implementation gives 49.4%, consistent.

**PASS bar (median <= 30% AND coverage >= 75%): FAIL for the shippable plate-normalised variant (35.5%, coverage 78%).** The fixed-camera image-normalised variant reads 28.3% / 78% on test, but its dev-val figure was 37.7%, so that test number is partly luck (n=60; the 95% CI on a median of 60 is roughly +-8 points) and it cannot work on arbitrary phone photos. I do not count it as a pass for the product.
Honest side note: on meal-sized dishes (>= 300 kcal) constant-350 (18.3%) beats every estimator here, because Nutrition5k meals cluster around 350-600 kcal. The component model's advantage is on small dishes and in correlation (r 0.65 vs -0.21).

### Failure examples (test)
Over-predictions (small food on a big plate; all five are the same failure, area is too large for the thin layer of food or components are mislabelled):
- dish_1561574600: true 33, pred 211. Truth is a little squash; labelled sweet_potato/potato_roast/squash.
- dish_1566417442: true 94, pred 404. True: oil 56 kcal + squash; predicted fried_rice + mixed_veg.
- dish_1560543505: true 66, pred 195. Truth salad+tomato; a mask read as `dressing` (sauce group has high kcal/g).
- dish_1562949242: true 140, pred 401. Bacon + sweet potato + melon correctly named, but bacon (5 kcal/g) on a large area.
- dish_1565812146: true 73, pred 204. Salad read as leafy_salad+stirfry_greens, plus invisible-oil truth.
Under-predictions (dense proteins; height unseen from above):
- dish_1565030391: true 609, pred 193. Beef 483 kcal predicted as broccoli/stirfry greens (a beef-with-greens plate).
- dish_1560367715: true 598, pred 234. Pork 364 kcal + rice; pork not found (pred rice+broccoli+chicken).
- dish_1563563289: true 332, pred 101. Chicken 274 kcal under-sized.
- dish_1563567722: true 645, pred 316. 215 kcal of oil, 134 of fish.

## Sanity on non-Nutrition5k photos (qualitative, `sanity.py`)
- 22 Taiwanese photos: 20 log something. Good: beef noodle soup, bubble tea, white rice (85% of the area), scallion pancake, potstickers, stir-fried greens, lu rou fan 2 (ground_pork_sauce + white_rice). Bad: fried egg 1 read as `dressing` (352 kcal), braised tofu logs nothing, lu rou fan 1 read as banana/pork/beef/pear, chicken leg 915 and pork chop 885-907 kcal (probably high; whole bento boxes though). Mixed dishes are where the hybrid is needed: the existing dish-level classifier names 15/22 sensibly and the component model adds little.
- User negatives (10 images): **6/10 log kcal** with the whole-image neg gate p_neg<=0.1 plus >= 1 component. FAIL vs the shipped pipeline's 0/10. This variant lacks the locator food:false classes and the full NEGS rows at mask level; those must be ported (each mask would be classified against shipped vocab + NEGS rows first, and masks whose neg mass is high dropped). Not done here.

## Feasibility (on-device)
- Extra model download: FastSAM-s ONNX (export verified, opset 17, only standard ops: Conv/ConvTranspose/Resize/Softmax/Slice...; output [1,37,8400] + protos [1,32,160,160]) is 47.3 MB fp32, ~23.7 MB with fp16 weights (the repo's own halving trick). Plate/bowl/tray = 3 extra prompts on the existing locator (the portion agent's loc17.onnx already has them), so ~0 MB. Text rows 95x512 fp16 = 0.1 MB. Total extra ~24 MB (budget 40).
- CPU per photo (onnxruntime, 4 threads, machine shared with another job): FastSAM 0.45-0.8 s, YOLOE 0.3-0.7 s, MobileCLIP ~0.115 s per crop x ~25 crops (12 masks x 2 crops + whole image) = ~3 s, total about 3.3 s. On an idle desktop expect ~1.5 s; a phone in wasm is likely 3-5x slower (5-15 s) and would need single-crop labelling or webgpu. The CLIP crops dominate; dropping the context crop halves them for a ~1-point accuracy cost on dev (mix=1.0: 41.0%).

## Verdict
Do not replace dish-level recognition with ingredient-level for kcal. Use area/portion measurement (the plate-ruler work) as the main gain; ingredient naming adds ~1 point on top and costs 24 MB and ~2x latency. If used at all, use it as a hybrid: components for plated/separable foods, whole-dish entries for soups/noodles/curries, and a hidden-oil allowance (+4 points). Caveats: Nutrition5k is a fixed overhead camera, cafeteria food, a small set of plate types and few sauces; every K here is fitted to it and would need re-fitting on Taiwanese meals; n=60 test makes all medians +-8 points.

## Reproduce
`cache_seg.py` (masks+embeddings cache) -> `mkvocab.py` -> `exp.py` (dev variants) -> `freeze.py` / `freeze_image.py` (params.json, params_image.json) -> `ablate.py` -> `final_eval.py` (test, run once) -> `sanity.py`, `test_detail.py`.
