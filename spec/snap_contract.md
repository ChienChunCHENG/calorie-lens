# Snap log: build contract (v1, 2026-10-31)

Product: open the site, take or choose one meal photo, and everything runs on the phone. Each food item is found, named
and given a kcal range, then saved to a daily log. **No cloud, no API keys, and no user choices**: the app never asks
the user to pick a food. The user can only delete a logged item or meal. The old live-boxes lens is retired.

Measured basis: `scratchpad/research/r1..r5`.
- MobileCLIP-S1 zero-shot over about 1,850 foods: top-1 55-59%, top-5 85-88% on Food-101, top-1 about 40% on Taiwanese photos.
- YOLOE-11s with generic prompts: 76-84% item recall on UEC meals.
- Empty cups currently read as milk tea.

## Pipeline (runs in a Web Worker, onnxruntime-web; webgpu if available, else wasm)
1. **Decode** the photo. Apply EXIF orientation with `createImageBitmap(file, {imageOrientation: 'from-image'})`, then
   downscale so the long edge is at most 1280.
2. **Locate.** `locator.onnx` (YOLOE with fixed prompts, letterbox to `imgsz`, gray 114 padding, RGB 0-1, CHW).
   - Decode `output0` [1, 4+nc(+32 mask coefs), N] (cx, cy, w, h in letterbox px).
   - Use only the first `nc` class rows, where `nc = locator.json.classes.length`.
   - Class-agnostic NMS at `iou`, then drop a box at least `contain` inside a higher-scoring one. Keep at most `max_det`.
   - A box whose best class has `food: false` (empty cup, cookware...) is discarded.
3. **No boxes:** classify the whole image as a single candidate region.
4. **Classify** each box crop (expanded by `crop_pad` of its size, clamped to the image) with `clip_image.onnx`, using
   the preprocessing in `clip.json`. L2-normalise the embedding.
   - Similarity to every row of `vocab_emb.bin`, times `logit_scale`, then softmax.
   - `p_neg` = probability mass on rows with `neg: true`. Skip the region if `p_neg > gate.p_neg_max` or the top food
     probability is below `gate.p_food_min`.
5. **Kcal per item: mixture, no user choice.**
   - Renormalise over food rows and take the top `K` candidates (cumulative probability up to `mix.cum`, at most
     `mix.k`). Renormalise again.
   - Item kcal is a mixture over candidates. Candidate c is drawn with probability p_c. Grams are log-normal with
     median `portion_g[1]` and sigma = ln(portion_g[2]/portion_g[0])/3.29 (min/max taken as the 5th/95th percentiles,
     the same convention as `sigmaOf` in core/ts/rangeEngine.ts; reuse it). kcal = grams × kcal100 / 100.
   - Report the 10th, 50th and 90th percentiles from deterministic Monte Carlo (n = 4000, seeded).
6. **Display.**
   - Name: top-1 candidate (zh if present, else en). Add "or similar" (或類似) when its probability is below
     `mix.sure`.
   - Meal total: the 10th, 50th and 90th percentiles of the sum of the item samples, treating items as independent.
7. **Log.** Store the meal in IndexedDB: timestamp, a 320 px JPEG thumbnail, and each item with its name, kcal
   percentiles, box and top candidates as {id, p}. Show the day total, a list of meals, and day navigation.
   - Delete a meal or an item.
   - Export and import a JSON backup.

## Files (all under web/public/models/, written by ml/snap/ scripts; the web app only reads them)
- `locator.onnx`: YOLOE export. Input name and shape are given in `locator.json`.
- `locator.json`:
  ```
  {imgsz, input, output, classes: [{name, food: bool}], conf, iou, contain, max_det, crop_pad}
  ```
- `clip_image.onnx`: MobileCLIP image encoder, fp16 weights. fp32 input/output is preferred; otherwise state the dtype
  in `clip.json`.
- `clip.json`:
  ```
  {input, output, dtype, size, resize: "shortest"|"squash", crop: "center"|null, interpolation,
   mean: [r,g,b], std: [r,g,b], dim, logit_scale,
   gate: {p_neg_max, p_food_min}, mix: {k, cum, sure}}
  ```
- `vocab.json`: array aligned row-for-row with `vocab_emb.bin`:
  ```
  {id, en, zh?, kcal100, portion_g: [min, typ, max], cat, neg?: true}
  ```
  Negative rows (empty cup, empty glass, empty bowl or plate, cookware, table, hand...) have `neg: true`, kcal100 0 and
  portion_g [0, 0, 0].
- `vocab_emb.bin`: float16 little-endian, rows × dim, each row L2-normalised. It holds the text embeddings of
  `"a photo of {en}."` (or the tuned template recorded in `clip.json.template`).
- `ATTRIBUTION.md`: USDA FNDDS (public domain), Taiwan FDA open data attribution string, model licences (MobileCLIP:
  Apple research licence, personal non-commercial use).

## Ownership
- **Models agent:** `ml/snap/**`, `web/public/models/**`. Also produces `web/test/fixtures/parity.json`, holding
  expected embeddings and top-3 for a few images, so the web side can check preprocessing parity.
- **Web agent:** everything else under `web/` (index.html, src/**, PWA manifest, icons, tests). It deletes the
  retired lens code.
- Opus (lead) owns this file, commits and pushes.
