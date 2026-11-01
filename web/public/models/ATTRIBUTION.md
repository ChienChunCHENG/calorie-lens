# Model and data attribution

- **MobileCLIP-S1** (image tower `clip_image.onnx`, text embeddings in `vocab_emb.bin`): Apple, weights `datacompdr`, loaded through
  OpenCLIP. Licensed under the Apple ML Research licence (apple-amlr): research / personal non-commercial use only. Not cleared for
  a commercial release. Image tower re-exported to ONNX with fp16 weight storage (compute fp32).
- **YOLOE-11s-seg** (`locator.onnx`): Ultralytics YOLOE, AGPL-3.0. Text prompts are baked into the head; only the detector is shipped.
  Re-exported to ONNX with fp16 weight storage. AGPL obligations apply if this is ever distributed beyond personal use.
- **Nutrition data** (`vocab.json`, kcal per 100 g and portion sizes): USDA FNDDS (public domain) and the Taiwan FDA food composition
  open data (food nutrient database, 衛生福利部食品藥物管理署 食品營養成分資料庫, Open Government Data licence v1.0), plus
  hand-estimated entries marked "est" in the source vocabulary (`research/r3`). Portion ranges are typical-serving estimates, not measurements.
- **Evaluation images** (not shipped): Food-101 subset, UECFoodPixComplete (via Hugging Face), Wikimedia Commons photos (various
  free licences, used only for local testing), and the user's own phone screenshots.
- **Commonness prior** (`lp` in `vocab.json`): food-frequency counts from the CDC NHANES 2021-2023 "What We Eat in America" dietary recall files (public domain). It ranks common foods above rare ones when the photo is ambiguous.
- **Added staples** (36 rows, e.g. fish and chips, schnitzel, Caesar salad): values come from FNDDS where a clean match exists. The rest are composed from FNDDS components or estimated, and those are marked `est` in `ml/snap/prior/out/new_rows_sources.json`.
