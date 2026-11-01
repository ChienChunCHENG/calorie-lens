# Portion from the photo: what the browser must compute (proposal, NOT shipped)

Status: experiment result is FAIL against the agreed bar (see RESULTS.md). This spec describes the method so it can be ported if the owner
decides the gain (median error 49% -> 43%, correlation 0.2 -> 0.6, p10-p90 coverage 58% -> 80-82%) is worth it. Reference code: `portion.py`
(`containers_of`, `container_for_item`, `fit_ellipse`, `food_pixels`, `assign_to_boxes`, `analyze`, `area_cm2`, `Params`, `sample_dish`).
All constants live in `portion_params.json` (it would become `web/public/models/portion.json`).

## 0. Locator change
Add three prompts to `locator.onnx`: `plate`, `bowl`, `tray` (food:false, so the existing discard rule still applies to them). They are rows 14-16 of
`loc17.onnx` (`export_portion_loc.py`). Read the mask-prototype output (`output1`, [1,32,160,160]) and the 32 mask coefficients that follow the class
rows in `output0` (the current model already exports them; the app ignores them today).
Container candidates = rows 14-16 plus the existing rows 1-3 (`dish of food`, `bowl of food`, `plate of food`, score x 0.8):
class-agnostic NMS at iou 0.5 on those rows only (no `contain` rule), conf >= 0.10, keep the best 4 with score >= `min_container_score` (0.15) whose
longer box side is >= `min_container_frac` (0.12) of the image short side. type = plate (rows 14, 1, 3) | bowl (15, 2) | tray (16).

## 1. Container ruler (per container)
- Mask = sigmoid(coef . protos) at 160x160, cropped to the letterbox content area, resized to the image (half resolution is fine), > 0.5, zeroed outside the box.
- Largest contour after a 5x5 close -> least-squares ellipse (`cv2.fitEllipse`; in TS fit with a small direct-ellipse solver or use the min-area rotated rect of the
  convex hull as a cheaper substitute: not validated). `major`, `minor` in image px (full image resolution), centre, angle.
  Fallback when the mask gives no contour: the box (major = longer side, minor = shorter, angle 0).
- `cm_per_px = diam_cm[type] / major`. Values: plate 26, tray 26, bowl 24 (`diam_cm`). A user setting "my plate is X cm" replaces `diam_cm.plate`
  (and bowl scaled by 24/26). Tilt correction: `tilt = clamp(minor/major, 0.5, 1)`.

## 2. Which container an item belongs to
For each logged item (the unchanged snap items), pick the container with the largest share of the ITEM box inside the container box (`inside_frac`,
needs >= 0.5; among near ties (within 0.25) the higher score). No such container -> the item keeps today's behaviour (vocab typical portion and
vocab sigma, widened x`fallback_sigma_mult` = 1.5). Unvalidated on real data (3/800 Nutrition5k images, 0/10 negatives).

## 3. Food area
For every container used by at least one item (image RGB, any size; the reference runs at <= 1280 px long edge):
1. Blur 5x5, convert to CIE Lab (L scaled to 0-100, a and b centred).
2. Reference colour: pixels in the ring between 0.80 and 0.97 of the ellipse (ellipse scaled about its centre). Quantise Lab with bin sizes (12, 8, 8), take the
   most populated bin (`food_pixels`), mean of its pixels (median of the ring if the bin has < 20 pixels). Need >= 50 ring pixels else no food.
3. Food pixel = inside the 0.97 ellipse and Lab distance to the reference > `food_lab_thr` (18). Then a 3x3 opening.
4. Each food pixel goes to the item of that container whose box contains it (smallest box wins), otherwise to the nearest box (distance to rectangle).
   `area_px_i` = pixel count. (Instance masks and boxes were also tried and were far worse, RESULTS.md.)
5. `area_cm2_i = max(area_px_i * cm_per_px^2 / tilt, area_cm2_min)`, `area_cm2_min` = 0.5.

## 4. Grams and kcal
- `grams = k * area_cm2^beta` with ONE global k = 1.5667 and beta = 0.8933 (`groups.*` are identical in the file; the per-group fit did not beat the global one on
  validation, so there is no food-group table to port).
- Density: `kcal_per_g(c) = max(kcal100_c/100, 0.02)^gamma * rho0^(1-gamma)`, gamma = 0.59 (file value), rho0 = 1.2. This deliberately shrinks the name-based density
  toward a typical dish: Nutrition5k dishes span 0.5-2.3 kcal/g (p10-p90) and the name is only weakly informative.
- Item median kcal = sum over the mixture candidates c of `p_c * grams * kcal_per_g(c)` (same candidates and weights as today's mixture).

## 5. Range
Monte Carlo as in the contract (n = 4000, seeded). For each item and sample: draw candidate c ~ p, `z ~ N(0,1)`,
`kcal = k*area^beta * exp(sigma_item * z) * kcal_per_g(c)`; `sigma_item` = 0.65 (calibrated on the validation split for ~80% p10-p90 coverage; items independent;
`sigma_dish` = 0, i.e. no shared factor). Dish total = sum over items; report p10/p50/p90. Items without a container use the old portion mixture.
Expect the range to be wide: median p90/p10 = 5.6x on Nutrition5k.

## 6. Fallbacks
- No container for the item: today's typical-portion behaviour (1.5x wider sigma). No item logged: log nothing, as today.
- A user-visible "plate size" setting is the cleanest way to make cm meaningful on a phone; without it the plate acts as a relative ruler against a 26 cm assumption.

## 7. Cost
One extra 3-class slice of the locator head (no extra network) plus mask decoding for <= 4 boxes and one Lab conversion of the image (about 0.3 MP at 640x480; ~1 MP at
1280). The reference does it in numpy/OpenCV in tens of ms. The ellipse fit and Lab pass need a small JS implementation (no OpenCV).

## 8. Parity checklist for the TS port
Dump with `feat.py` + `portion.analyze` for a few images: container list (box, type, score, ellipse), per-item `area_px`, `area_cm2`, final p10/p50/p90. Compare with the
browser within 3% area and 5% kcal (ellipse fits differ slightly between implementations).
