# Browser computation: text + image-prototype score (candidate, not wired into web/)

Files (candidate, in `ml/snap/protos/out/`): `protos_emb.bin` (2,304 x 512 float16 little-endian, rows L2-normalised, 2.36 MB) and `protos.json`.

`protos.json`: `{dim:512, n:2304, K:16, a:1.0, c:0.65, m:3, relu:true, concepts:[{name, vocab_ids:[...], neg:bool}], offsets:[o_0..o_C]}`.
Concept k owns prototype rows `offsets[k] .. offsets[k+1]-1`. 144 concepts (124 food, 20 non-food object concepts mapped to `neg_*` rows). `vocab_ids` are vocab ids (not row numbers): resolve with a map id -> row of the shipped `vocab.json`; ids that are missing are skipped.

Per image region (crop), right after the existing L2-normalised embedding `e` (512) is computed:
```
s[i]   = dot(vocab_emb[i], e)                       // existing text cosine, every vocab row (food and neg)
for each concept k with offsets[k+1] > offsets[k]:
    sims   = [dot(P[j], e) for j in offsets[k] .. offsets[k+1]-1]
    q      = mean of the m=3 largest sims            // all of them if the concept has fewer than 3
    boost  = a * max(q - c, 0)                       // a = 1.0, c = 0.65
    for id in concepts[k].vocab_ids: s[row(id)] += boost
z = logit_scale * s                                  // 64.25, unchanged
p = softmax(z)                                       // gate, mixture and display after this are unchanged
```

Notes
- Rows with no concept are untouched. A row belongs to at most one concept. The boost is zero unless the crop is closer than 0.65 to some prototype, so unusual foods and objects keep the pure text behaviour.
- The 20 negative concepts (cutlery, table, hand, person, packaging ...) boost the `neg_*` rows the same way; this keeps `p_neg_max` gating working. Do not drop them.
- Cost per crop: 2,304 x 512 multiply-adds (about 1.2 MFLOP) plus a top-3 per concept, negligible next to the image encoder. Download +2.36 MB (fp16); total models about 68 MB.
- Keep gate parameters `p_neg_max 0.1`, `p_food_min 0.05`, `logit_scale 64.25`, mixture `k 3 cum 0.7`. `mix.sure` (0.7) was calibrated on text-only scores; scores are sharper now, so re-calibrate it.
- Python reference: `protolib.Tip(T, rc, P, pc, a, c, m, relu=True) @ e` in `ml/snap/protos/protolib.py` returns `s` (before the scale). A parity fixture was NOT generated.
- Vocab drift: evaluation used the 2,322-row research vocab. The shipped `vocab.json` now has 2,358 rows (36 new foods); prototypes are keyed by id, so new rows simply get no boost.
- Zero-cost alternative (not recommended, see RESULTS.md): replace `vocab_emb.bin` rows by `normalize(text_i + lam * mean_image_k)`. It gives similar Food-101 accuracy but breaks negatives (6/10 user negatives logged) because every food row is boosted for every image.
