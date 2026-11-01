# SPEC: what the browser must compute for the component-level estimate (only if the hybrid is adopted; see RESULTS.md verdict)

Reference: `ingr.py` (+ `cache_seg.py` for steps 1-2, `fit_params.py` for the feature/loss). Parameters: `params.json` (fields `lk`, `groups`, `oil`, `thr`, `resid_q`, `mix`, `temp`). Vocabulary: `ingredients_vocab.json` (87 rows: id, en, group, kcal_g) and `ingredients_vocab_emb.npy` (95 x 512 fp16; 87 components then 8 background rows; text = `"a photo of {en}."` through the MobileCLIP text tower, same template as `clip.json`).

## Inputs / models
- Photo decoded as in `spec/snap_contract.md` (EXIF, long edge <= 1280).
- `locator.onnx` extended with 3 prompts `plate, bowl, tray` (rows with mask coefficients; the portion agent's `loc17.onnx` already has them). FastSAM-s ONNX (`w/FastSAM-s.onnx`, imgsz 640, letterbox, output0 [1,37,8400] = 4 box + 1 score + 32 mask coefs, output1 protos [1,32,160,160]); ship with fp16 weights (~24 MB).
- `clip_image.onnx` (existing).

## Steps
1. Container: run the locator with prompts plate/bowl/tray, conf 0.10, NMS iou 0.5; take the highest-score box's instance mask (proto @ coefs, sigmoid > 0.5, un-letterbox: crop the pad `top=(S-nh)/2`, `left=(S-nw)/2`, resize to image). Plate region = **convex hull** of that mask (fill polygon). If the hull is < 2% of the image or nothing is found, use the whole frame.
2. Masks: FastSAM-s, conf 0.3, class-agnostic NMS iou 0.8, instance masks as in step 1 (threshold 0.5). Keep a mask if: 0.25% <= area <= 70% of the image; >= 50% of the mask inside the plate hull; area <= 92% of the hull; IoU with an already kept (larger-first) mask <= 0.85. Keep at most the 14 largest.
3. Crops (per kept mask): (a) tight bbox + 10% padding made square, pixels outside the mask set to RGB(124,124,124); (b) bbox + 25% padding made square, unmasked, gray 124 where out of frame. Each goes through the shipped CLIP preprocessing (shortest side 256, centre crop, RGB 0-1) and `clip_image.onnx`; L2-normalise. Embedding e = normalise(0.5*e_a + 0.5*e_b).
4. Labels: p = softmax(64.25 * e @ T^T) over the 95 text rows. The `oil` component row is invisible: negate its text embedding (never wins). p_k for the 87 components, the 8 background rows absorb plate/table/utensil masks.
5. Areas: low-res (e.g. 160 px wide) masks. Sort masks by area descending; paint each mask index into an id map restricted to the plate hull (later = smaller masks overwrite), skipping masks that fail step 2. a_i = painted_pixels_i / hull_pixels.
6. Component area: A_k = sum_i a_i * p_ik (soft; BG probability removes area).
7. Estimate: for each k with A_k >= thr (0.004): grams_k = exp(lk[group_k]) * A_k (b = 1); kcal = sum_k grams_k * kcal_g_k + oil (33 kcal; always added). Only the group constants `lk` and `oil` are fitted (18 numbers).
8. Range: median = kcal; p10 = kcal * exp(resid_q[0]) ; p90 = kcal * exp(resid_q[1]) (about x0.45 .. x2.1). Display the dominant components (A_k >= 4% of the plate) with their gram estimates.
9. Gate (needed, not yet solved): every mask must also be scored against the shipped NEGS rows and `p_neg_max` 0.1 / `p_food_min` 0.05 before painting; as implemented here only 4/10 of the user's negatives were rejected.

## Validation vectors
`test_results.json` and `dev_results.json` hold per-dish numbers; for TS parity export the per-image masks and `ea`/`eb` embeddings from `seg_*.pkl` (not committed; see `cache_seg.py`).

## Cost
Extra download ~24 MB (FastSAM-s fp16). Per photo about 12 masks x 2 crops of MobileCLIP (the dominant cost, ~1 s on an idle desktop CPU, 3x or more on a phone). Single-crop (masked only) halves it for about +1.8 points error.
