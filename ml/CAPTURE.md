# CAPTURE.md - photo protocol for the Calorie Lens dataset

Goal: about **1,000-1,500 images**, **150-300 instances per class** for the 15 dish classes in `ml/labels.yaml` (the 4 container classes come for free in the same photos). You shoot with an iPhone; nothing here needs special gear.

## 1. What to shoot

One *meal* = one plate/tray/cup set on one table at one moment. Take **6-10 photos per meal** (see angles), then move on. Many different meals beat many photos of one meal.

| Angle | Camera tilt | Distance | Per meal |
|---|---|---|---|
| Overhead | straight down (90°) | 30-50 cm | 2 |
| 45° | the "diner's eye" view | 30-50 cm | 3 |
| 30° (low) | shows bowl depth, cup contents | 30-50 cm | 2 |
| Free / messy | handheld, off-centre, partly cropped | 25-60 cm | 1-3 |

Rules of thumb:
- Keep the whole container in frame in at least half of the shots (the rim matters for portion estimation). The rest may crop the dish edge, as real use will.
- Re-position the camera between shots (do not zoom: use your feet). Use the 1x lens; no portrait mode, no filters, no HDR "food" modes, no digital zoom.
- Tap to focus on the food; avoid motion blur.
- Shoot some meals **before** eating and a few **half-eaten** (partial portions occur in real use).
- Include 1 shot per few meals with a hand holding the chopsticks/spoon in frame (natural occlusion). Hands are fine; faces are not.

## 2. Variety (this is what makes the model generalise)

- **Lighting:** daylight by a window, shop fluorescent (very common, greenish), warm restaurant light, night street stall, backlit window, harsh overhead spot with glare on soup/oil. Aim for no more than 40% in any one lighting type.
- **Place:** shops/street stalls/night markets and home cooking about 50/50 at the start.
- **Table/background:** wood, steel tray, plastic tablecloth, white table, patterned placemat, paper-lined tray.
- **Format:** bento boxes (便當), 小吃 (bowls, plates, paper trays, paper bags for 鹽酥雞/雞排), and drinks (cup in hand, on table, with straw/lid sealed).
- **Quantity of items:** single-dish shots (needed to learn each class cleanly) AND 2-5 item combos (a typical 自助餐/便當 tray). Roughly 40% single, 60% combos.
- Different shops/cooks: the same dish looks different everywhere, so spread each class across **at least 8-10 different shops/kitchens**.

## 3. Folder layout and the split rule

```
raw/
  20261030_meal01/   IMG_0001.HEIC ... (all photos of ONE meal)
  20261030_meal02/
  20261031_meal01/
```
- Folder name: `YYYYMMDD_mealNN` (NN restarts each day; 01, 02, ...).
- **The train/val/test split happens by meal folder, never by photo.** Photos of the same meal are near-duplicates; if they land in both train and val, the evaluation lies. Keep ~15% of meal *folders* as the held-out test set from the very start and never label-tune on them.
- Never rename or move single photos between meal folders. Keep the folder in the file path through prep, labelling and export (all scripts in `ml/` preserve it).
- Drinks bought with a meal go in the same meal folder.

## 4. Privacy (do this before cloud tools)

- **No faces**, no name tags, no receipts/phone screens, no delivery addresses in the frame. Crop or delete the photo if one slips in.
- **Strip location.** Either:
  1. iPhone: Photos > select > Share > **Options** (top of the share sheet) > turn **Location** off (also turn "All Photos Data" off; choose Format: **Most Compatible** to get JPEG instead of HEIC), then AirDrop/save to files; or
  2. Script on the PC (also converts HEIC to JPEG, applies rotation, resizes to 1600 px, removes all EXIF): `py ml/xanylabeling/prep_photos.py raw clean`
- Or switch off geotagging at the source: Settings > Privacy & Security > Location Services > Camera > **Never**.
- Only the cleaned copy goes to Roboflow (public on the free tier) or to `prelabel.py` (sends images to the Anthropic API). Keep the raw originals local.

## 5. Targets

| Stage | Images | Notes |
|---|---|---|
| **First 300** | 300 | Prioritise the most common: 白飯, 滷肉飯, 雞腿, 排骨, 滷蛋, 炒青菜, 牛肉麵, 珍珠奶茶 (bento and 自助餐 trays hit most of these at once). Train a first model on this and see what fails. |
| Next 300 | 600 | Add 荷包蛋, 滷豆腐, 水餃, 鍋貼, 雞排, 鹽酥雞, 蔥油餅; more shops and lighting. |
| Goal | 1,000-1,500 | 150-300 instances of each of the 15 dishes. Use `prelabel.py` summary.csv or the label counts to find classes that are short. |

Bento trays count instances for several classes at once, which is the cheapest way to reach 150-300 for staples. Rare classes (蔥油餅, 鍋貼, 荷包蛋) need deliberate shots: order them on purpose, and photograph 2-3 servings per shop.

Public seed sets (Roboflow Universe, UEC-Food256, FoodSeg103) are for warm-up only; the held-out test set must be only your own meals.

## 6. One-page checklist

Before leaving:
- [ ] Camera: 1x lens, no zoom, no filters/portrait/Live (turn Live Photo off), lens wiped.
- [ ] Location off at source (Camera > Never), or I will strip it later.

Per meal:
- [ ] New folder name `YYYYMMDD_mealNN` noted (or sort photos into it tonight, by time).
- [ ] 2 overhead, 3 at 45°, 2 at 30°, 1-3 free shots; 30-50 cm.
- [ ] Container fully visible in at least half the shots; cup shots show the contents.
- [ ] No faces / receipts / screens.
- [ ] Different shop or lighting than the last meal if possible.

Weekly:
- [ ] Copy to PC, group by meal folder, run `prep_photos.py` (strips GPS/EXIF).
- [ ] Count per class; pick the short classes for the next outing.
- [ ] Mark ~15% of meal folders as test; never train on them.
- [ ] Only the `clean/` folder is ever uploaded anywhere.
