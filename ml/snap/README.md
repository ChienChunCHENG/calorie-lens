# ml/snap: on-device models for the snap log

Writes `web/public/models/{locator.onnx, locator.json, clip_image.onnx, clip.json, vocab.json, vocab_emb.bin}` and
`web/test/fixtures/parity.json`. Contract: `spec/snap_contract.md`.

## Rebuild
Venv needs torch, open_clip, ultralytics (YOLOE), onnx, onnxruntime, onnxslim, timm. Data paths are the session scratchpad (`common.SP`, env `SP`).

```
python export_clip.py                 # clip_image.onnx (fp16 weights, fp32 I/O) + clip.json (preprocessing read from open_clip)
python export_locator.py --superset   # 30-class YOLOE used only for tuning
python cache_loc.py 640               # cache raw locator outputs for every eval image
python tune.py text; python tune.py embed; python tune.py sweep; python tune.py final   # tune + write chosen params
python export_locator.py              # final locator.onnx (14 classes) from locator_classes.json
python build_vocab.py                 # vocab.json + vocab_emb.bin
python make_parity.py && (cd ../../web && node ../ml/snap/parity_node.mjs)
python e2e_shipped.py                 # sanity run using only the shipped files
```
A YOLOE head scores each class independently with a shared box regression, so a prompt subset equals the matching rows of the superset
output; `tune.py` uses that to compare class sets without re-running the network.

## Files and sizes
| file | size |
|---|---|
| clip_image.onnx (MobileCLIP-S1 image tower, fp16 weights, fp32 compute/IO) | 43.3 MB |
| locator.onnx (YOLOE-11s-seg, 14 prompts, fp16 weights, imgsz 640; output1 = mask protos, unused) | 20.4 MB |
| vocab_emb.bin (2,322 x 512 fp16) | 2.4 MB |
| vocab.json (2,278 foods + 44 negative rows) | 0.28 MB |
| total | about 66 MB |

fp16 check: embeddings from the fp16-weight model vs fp32 have cosine 0.99996-0.99999 on random input.
fp16 is weight storage only (a `Cast` back to fp32 at load); `onnxconverter-common`'s full fp16 conversion produced an invalid graph for this model.

## Chosen parameters
- Locator: imgsz 640, prompts food:true = food, dish of food, bowl of food, plate of food, drink, cup, cup of drink, glass of drink, bubble tea;
  food:false = empty cup, empty glass, paper cup, tumbler, mug. conf 0.15, iou 0.3, contain 0.6, max_det 6, crop_pad 0.2.
- CLIP: resize shortest to 256 (bilinear) + center crop, template `"a photo of {}."` (foods); negative rows use 1-4 hand-written sentences each
  (`snaplib.NEGS`). gate p_neg_max 0.1, p_food_min 0.05 (p_food_min is the top food row's raw softmax probability over ALL rows, before renormalising).
  mix k 3, cum 0.7, sure 0.7 (top-1 renormalised weight; accuracy was about 0.7-0.9 above 0.7 and 0.3-0.5 below, on 185 + 22 images).
- **Whole-image fallback (contract step 3) is only safe with a much stricter gate**: `locator.json.fallback_whole` = used only when the locator
  returns NO box at all (before the food filter), with p_neg_max 0.005 and p_food_min 0.02. With the normal gate the fallback made 1-2 of the
  user's 10 negatives log kcal.

## Measured results (exact pipeline at the chosen params, `results.json`)
Eval sets: user negatives 10; extra negatives 30 (Wikimedia, hand-screened); drink positives 39 (Wikimedia, hand-screened); UEC multi-item 128 images/379 items; Food-101 300 images (60 classes; 185 images in 37 classes have a vocab equivalent); Taiwanese 22 photos.

| objective | result |
|---|---|
| (a) user negatives logging any kcal | 0/10 |
| (a) extra negatives logging any kcal | 1/30 (water bottle read as papaya juice); tune half 0/15, held-out half 1/15 |
| (b) drink positives with any item logged | 39/39; top-1 is a drink-category row in 38/39 (this is not "correct drink") |
| (c) UEC item recall (logged items, IoU>=0.5) | 0.55 (precision 0.62, 2.6 logged items/img vs 2.96 GT) |
| (c) Food-101, any item logged (mappable subset, n=185) | 99.5% (all 300: 99.3%) |
| (c) Food-101 whole-image top-1 / top-5 (mappable) | 59% / 85% |
| (c) Food-101 largest-logged-item top-1 / any of top-3 mixture candidates | 60% / 78% |
| (c) Taiwanese (n=22, lenient regex match) item top-1 / top-3 | 59% / 86% (whole image: same) |
| (d) Food-101 median abs % error of total kcal vs true entry at typical portion (n=184) | mixture median 25.0%, top-1 only 23.9% |

Findings and caveats, stated plainly:
- **The mixture does not improve the point estimate**: median error is 25.0% (mixture median) vs 23.9% (top-1 typical portion), i.e. no gain. Its value, if any, is the width of the range (not scored here).
- Food-101 truth is the vocab entry's typical-portion kcal, not the real portion in the photo, so the 24-25% figure measures name/entry agreement plus nothing about portion size. 37 of 60 Food-101 classes have a vocab entry; others are excluded.
- Taiwanese "match" = regex on the vocab English name per dish (`evalsets.TW_ACCEPT`), deliberately lenient (e.g. any tofu entry for braised tofu, any egg-dish for fried egg). 22 photos is tiny; one image moves 4.5 points.
- Negatives: the neg prompt list was written knowing the user's five objects (glass, tumbler, paper cups, wok), and params were selected with the user's 10 and the 30 extras in the objective. The tune/held split of the extras is by index parity only. 0/30 has a 95% upper bound of about 10%. Real-world false-log rate is unmeasured.
- The locator's own food:false classes alone do not solve negatives (without them: 1/10 of the user's negatives log); the CLIP negative rows plus gate do most of the work.

### Curves at the chosen params (other params fixed)
Gate p_neg_max (p_food_min 0.05): 

| p_neg_max | user neg | extra neg | drinks logged | UEC recall | Food-101 logged | TW logged |
|---|---|---|---|---|---|---|
| 0.005 | 0/10 | 0/30 | 0.77 | 0.47 | 0.98 | 0.95 |
| 0.02 | 0/10 | 1/30 | 0.85 | 0.53 | 0.99 | 1.00 |
| 0.05 | 0/10 | 1/30 | 0.97 | 0.54 | 0.99 | 1.00 |
| **0.1** | 0/10 | 1/30 | 1.00 | 0.55 | 0.99 | 1.00 |
| 0.2 | 0/10 | 1/30 | 1.00 | 0.55 | 0.99 | 1.00 |
| 0.3 | 2/10 | 1/30 | 1.00 | 0.56 | 0.99 | 1.00 |
| 0.5 | 2/10 | 5/30 | 1.00 | 0.56 | 0.99 | 1.00 |
| 1.0 (no neg gate) | 4/10 | 6/30 | 1.00 | 0.57 | 0.99 | 1.00 |

p_food_min (p_neg_max 0.1): 0.0 -> extra neg 2/30, UEC 0.58; 0.02 -> 1/30, 0.58; **0.05** -> 1/30, 0.55; 0.10 -> 0/30 but drinks 0.95, UEC 0.43, TW logged 0.95; 0.15 -> drinks 0.79, UEC 0.31, Food-101 0.85.

Duplicate suppression (contain / iou) trades item recall for kcal accuracy: contain off + iou 0.6 gives UEC recall 0.67 (precision 0.68) but double-counts nested boxes, Food-101 kcal error about 100%. contain 0.6 / iou 0.3 gives 0.55 recall and 25% error.

Other axes tried in `sweep` (all in `$SP/snap/sweep*.log`): CLIP template variants (t1..t4, ensembles) differ by 0-3 points (noise); "squash" resize raised Taiwanese whole-image top-1 (0.68 vs 0.55) but made 2/10 user negatives log; pad 0-0.2 about equal (pad 0 logs 1/10 user negatives); imgsz 512 was NOT evaluated (only 640 was exported/tested).

## Not verified
- Real-device speed (onnxruntime-web wasm/webgpu on iPhone) and whether fp16-weight models run on webgpu there. Only desktop onnxruntime (python CPU and node CPU) was run.
- Browser decode/resize (canvas) vs PIL: `parity_node.mjs` checks a JS bilinear implementation against PIL on upscaled small images (cosine >= 0.9997); canvas `drawImage` downscale of a 1280 px photo to 256 is antialiased differently and is untested.
- Portion sigma and kcal ranges were never checked against real weights.
