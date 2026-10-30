# Seed datasets for Calorie Lens

Researched 2026-10-30. **Honesty key:** `OPENED` = I fetched the page and read it. `SEARCH-ONLY` = seen in a
search result snippet; the page itself returned HTTP 403 to my fetcher (all `universe.roboflow.com` pages did),
so counts/classes/licences are unverified and must be checked in a browser before use.

**Bottom line:** I found **no verified Taiwanese bento / 滷肉飯 / 雞排 / 鹽酥雞 / 珍珠奶茶 instance-segmentation
dataset**. The Taiwanese-looking Roboflow sets are tiny (about 100 images) and unverified. The big,
verified segmentation sets (FoodSeg103/FoodInsSeg, UECFoodPix) are Japanese/Western/Chinese and cover only
generic items such as rice. Treat all of them as **warm-up / pretraining data**; the real signal must come from your
own labelled meals (PLAN M3 step 1, `ml/LABELING.md`). All sources here that I could verify are non-commercial
or research-only, which is fine for personal use but blocks shipping a model trained on them commercially.

Our target ids (index order, `ml/labels.yaml`): 0 rice_white, 1 lu_rou_fan, 2 chicken_leg, 3 pork_chop,
4 braised_egg, 5 fried_egg, 6 stirfried_greens, 7 braised_tofu, 8 beef_noodle_soup, 9 boiled_dumplings,
10 potstickers, 11 scallion_pancake, 12 fried_chicken_cutlet, 13 popcorn_chicken, 14 bubble_milk_tea,
15 container_bowl, 16 container_plate, 17 container_bento, 18 container_cup.

Mapping rule: a source class is mapped only when it is the same visual thing. Anything weak is **dropped**
(better missing than label noise: 滷肉飯 vs 白飯, 雞腿/排骨/雞排 and 水餃/鍋貼 are exactly the pairs we need clean).
Mappings are applied with `ml/make_dataset_yaml.py` and a `ml/mappings/<name>.yaml` (see `_example.yaml`).

---

## A. Verified (OPENED)

### 1. FoodInsSeg (instance masks, ingredient level) -- **best verified seed**
- URL: https://github.com/jamesjg/FoodInsSeg (download: Google Drive / Baidu links in the README)
- Status: **OPENED** (README read).
- Annotation: **instance segmentation**, COCO JSON polygons (`Train.json`, `Test.json`).
- Images: 7,118 (4,983 train / 2,135 test), 119,048 masks, 103 categories. Same image set as FoodSeg103, re-annotated per instance.
- Licence: **CC BY-NC-SA 4.0**, academic research only, commercial use prohibited (README).
- Class names: not read (they are in the JSON `categories`). Check before writing the mapping.
- Mapping (candidate; confirm names in `categories`): `rice` -> `rice_white`. Everything else dropped: it is ingredient-level
  (fruit, vegetables, sauce, ...), no lu_rou_fan, no bento, no drinks, no containers.
- Notes: needs COCO -> YOLO-seg conversion (e.g. `ultralytics.data.converter.convert_coco(labels_dir=..., use_segments=True)`, or import into Roboflow and re-export).
  Because instances of non-rice classes get dropped, use `--drop-images-with-dropped` when merging, otherwise untouched food becomes "background".

### 2. FoodSeg103 (semantic masks)
- URL: https://huggingface.co/datasets/EduardoPacheco/FoodSeg103 (community mirror of the original FoodSeg103 benchmark)
- Status: **OPENED** (HF page). Class-name list not visible in what I read.
- Annotation: **semantic** segmentation (one label image per photo; instances are not separated, so not directly usable for our instance-seg head).
- Images: 7.12k rows (4.98k train / 2.14k val), 103 classes.
- Licence: **Apache-2.0** tag on this HF mirror. The original dataset is distributed for research; PLAN.md treats it as non-commercial. Do not assume the Apache tag overrides the original terms.
- Mapping: prefer FoodInsSeg (same images, instance level). Search snippets suggested a "wonton dumplings" and a generic "fried meat" class; I did **not** confirm them. Do not map "fried meat" -> 雞排 (it is generic and would poison 雞腿/排骨/雞排).

### 3. UECFoodPix / UECFoodPixComplete (Japanese food masks)
- URL: https://mm.cs.uec.ac.jp/uecfoodpix/
- Status: **OPENED** (official page).
- Annotation: pixel-wise segmentation masks (semantic, label in the R channel, 0 = background); 103 labels. *UECFoodPix* masks were made with box + GrabCut (noisy); *UECFoodPixComplete* (759 MB) is manually annotated.
- Images: 9,000 train + 1,000 test.
- Licence: quote from the page, "can be used only for non-commercial research purpose".
- Mapping: semantic masks need a connected-components pass to become instances; I did not read the label list. Candidate only after you read it: plain `rice` -> `rice_white`; `gyoza`-like class (if present) -> `potstickers` (Japanese gyoza is pan-fried); everything else dropped.
- Mirror: https://huggingface.co/datasets/justinsiow/UECFOODPIXCOMPLETE (**OPENED**; 10k rows, Parquet, JPEG + PNG label maps, **no licence stated**, so use the original page's terms).

### 4. THFOOD-50 subset `Pchanyutt/siamtech-food10` -- **rejected**
- URL: https://huggingface.co/datasets/Pchanyutt/siamtech-food10 -- **OPENED**.
- Classification only (folders by class), 10 Thai dishes, 1,590 images at 256x256, non-commercial research/education only. No boxes or masks, so useless for the detector. Listed so nobody re-investigates it.

---

## B. Seen only in search results (SEARCH-ONLY, unverified; page returned 403 to me)

Open each in a browser and check: task type (Instance Segmentation vs Object Detection), class list, licence badge, image count, and sample images.

### 5. Roboflow `TW_Food`
- URL: https://universe.roboflow.com/twfood/tw_food -- search snippet: about 137 images, Taiwanese food. Annotation type, classes and licence unknown.
- Mapping: unknown until opened. Likely candidates: rice / 滷肉飯 / 雞排 / 水餃 style classes onto 0, 1, 12, 9. Fill in `ml/mappings/tw_food.yaml` from the real `data.yaml` `names`.
- Value: closest thing to our domain; small. If it is box-only, `make_dataset_yaml.py` converts boxes to rectangle polygons (poor masks; use only for class/detector pretraining, not mask quality).

### 6. Roboflow "Taiwanese-Food"
- About 100 images per the same search snippet; I did not get its URL. Search Universe for "Taiwanese-Food". Everything else unknown.

### 7. Roboflow `bento` (sota-seki) and `bento-detection` (ryosei-tsuda)
- URLs: https://universe.roboflow.com/sota-seki/bento , https://universe.roboflow.com/ryosei-tsuda/bento-detection
- Snippet: `bento-detection` has 31 images; `bento` has no description. Origin is unknown (the word bento does not imply Taiwan), task type unknown.
- Mapping if boxes of the whole box: `bento` -> `container_bento`; compartments/contents dropped. 31 images is only a smoke test.

### 8. Roboflow `train_food` (fikree) and `braised_pork_belly` (test-helge)
- URLs: https://universe.roboflow.com/fikree/train_food (14 classes, one is a stewed-pork-leg-on-rice class), https://universe.roboflow.com/test-helge/braised_pork_belly (51 images, single class, detection).
- Mapping: braised pork on rice looks like 滷肉飯 but is a *different dish* (stewed pork leg 滷豬腳飯 / pork belly); mapping it to `lu_rou_fan` is risky. Safest to drop; use only if sample images clearly show minced braised pork over rice.

### 9. Roboflow bubble tea / cups
- `00-wrhv1/other-project` https://universe.roboflow.com/00-wrhv1/other-project : a detection API with a "bubble tea" class among bagel / christmas tree classes. Counts and licence unknown. `bubble_tea` -> `bubble_milk_tea`.
- "Bubble Tea" workspace https://universe.roboflow.com/bubble-tea : projects `final_test2` (55 images) and `test` (73 images); object detection; class contents unknown.
- `knee-prosthesis/cup-ygo7m` https://universe.roboflow.com/knee-prosthesis/cup-ygo7m : snippet says **Instance Segmentation, CC BY 4.0**, generic cups, no description. `cup` -> `container_cup` (drink cups; reject coffee mugs by eye).
- DamnSon workspace (https://universe.roboflow.com/damnson) lists "Cup" (342 images) and "Coffee Cup" (195 images) per the snippet; unverified.

### 10. UEC-Food256 (boxes) and Food-101 (classification)
- UEC-Food256: https://mm.cs.uec.ac.jp/uecfood256/ (page 403; PLAN.md lists it as non-commercial). 256 Japanese categories with bounding boxes; I could not read the class list. Use for detector warm-up only (`rice`, `gyoza`-like classes if present).
- Food-101 (HF `food101`): classification only, has a "dumplings" folder; **cannot** supply boxes/masks. Not usable here.

---

## Recommended use

1. Fine-tune the COCO-pretrained `yolo11n-seg.pt` on **your own** 150-300 instances per class; that is what the M3 exit criterion measures.
2. Optional warm-up: FoodInsSeg `rice` + any verified Taiwanese Roboflow set + a bubble-tea/cup set, merged via `make_dataset_yaml.py` with the held-out meals kept out of the merge.
3. Never put seed data into the held-out evaluation set.
4. Record the licence of every source you actually use in `ml/mappings/<name>.yaml` (`source:` field).
