# LABELING.md - polygon/mask labelling rules and tools

Output: one **instance segmentation** polygon per item, using exactly the 19 class ids in `ml/labels.yaml` (15 dishes + 4 containers). The yaml order is the model class index; never reorder or add classes here.

## 1. Rules

Label what the portion engine needs: the **food surface** for dishes, the **outline of the vessel** for containers.

| Situation | Rule |
|---|---|
| Mixed bowl (牛肉麵, 滷肉飯) | ONE instance of the dish class covering the **food surface only** (noodles, meat, broth, toppings together). Do not trace the bowl and do not split toppings. |
| Containers | Every bowl, plate, bento box and cup gets its OWN instance (`container_bowl`, `container_plate`, `container_bento`, `container_cup`) tracing the vessel **outline including the rim** (and the full body for a cup, excluding the straw and lid dome only if clearly separate). Overlap with the food instance is expected and correct. |
| Bento | One `container_bento` for the whole box (outer edge). Each compartment's food is its own dish instance (rice, pork chop, egg, greens...). Do not label the dividers. |
| Bubble tea | Two instances: `bubble_milk_tea` over the **visible cup contents** (liquid + pearls region, not the lid, straw or empty top) AND `container_cup` over the **whole cup** (rim to base, lid included). |
| Several same items, separate | One instance each (3 braised eggs = 3 instances; 2 chicken legs = 2). |
| Piles | 鹽酥雞 = **one instance per pile/bag/tray**, not per piece. 水餃 and 鍋貼 = one instance per plate pile (a touching cluster is one). A bag of 鹽酥雞: trace the visible pile surface. |
| Rice | `rice_white` for plain rice only. Rice covered by 滷肉飯 sauce/meat is `lu_rou_fan`, as one instance. Plain rice visible beside another dish in the same bowl or box is its own `rice_white` instance. |
| Partial occlusion | Trace only the **visible** part. If something is hidden behind another item, do not guess the hidden edge. Skip instances that are less than ~10% visible or you cannot identify. |
| Cropped by frame edge | Label the visible part, running along the image border. |
| Garnish and sauce | Exclude small garnish (scallion sprinkles, cilantro, pickled mustard greens, pepper, a spoon of chilli oil) from the mask: do not chase them; a slightly loose edge is fine. Sauce that is part of the dish surface (滷肉 sauce on rice) is included. A separate sauce dish is not labelled. |
| Soup | The broth surface is part of the food instance (牛肉麵). |
| Utensils, hands, straw, lids, paper bag | Never part of a food mask. |
| Unknown or out-of-list food | **Leave it unlabeled.** Do not use the closest class and do not invent a class. Do label its container if it is a bowl/plate/cup. (Note the shot in a `unknown.txt` list if the dish is common; this feeds the "next 25" classes.) |
| Ambiguous call | e.g. 滷肉飯 vs 肉燥飯, 雞腿 vs 排骨: label what the shop calls it if known; otherwise skip the item rather than guess. Flag the image for review. |
| Precision | Polygons should be tight to within a few pixels at 1600 px width; do not spend time on perfect corners. Aim for 30-60 s per image after a few dozen. |

## 2. Tools (Windows)

### 2a. X-AnyLabeling (recommended, local, private)

Verified 2026-10-30: latest release **v4.2.3** (published 2026-10-29), PyPI package **`x-anylabeling-cvhub`** 4.2.3, Python >= 3.11, console command `xanylabeling`.

Install, either:
- **Installer (easiest):** GitHub Releases > `X-AnyLabeling-v4.2.3-Windows-CPU-Setup.exe` (CUDA 11/12/13 variants also exist; use CPU unless you have an NVIDIA GPU. SAM runs fine on CPU for one click at a time).
- **pip** (recommended in a venv):
  ```
  py -3.13 -m venv .venv-label
  .venv-label\Scripts\activate
  pip install "x-anylabeling-cvhub[cpu]==4.2.3"
  ```
  (`[gpu]` instead of `[cpu]` for NVIDIA.)

Launch with our classes:
```
py ml/xanylabeling/make_classes.py          # regenerates classes.txt from labels.yaml
xanylabeling --labels ml\xanylabeling\classes.txt --autosave --nodata --filename D:\photos\clean
```
(`ml\xanylabeling\launch.bat D:\photos\clean` does both.) `--nodata` keeps JSON files small; `classes.txt` has one id per line in labels.yaml order.

SAM auto-segmentation: in the left toolbar open **Auto Labeling** (Ctrl+A) and choose a model; first use downloads it. Models offered in v4.2.3: Segment Anything 1 variants (SAM-HQ ViT-B/L/H, MobileSAM, EdgeSAM, EfficientViT-SAM L0/L1, SAM-Med2D), **Segment Anything 2.1 (Tiny / Small / Base / Large)**, and **Segment Anything 3 (ViT-H)**, plus video variants. Suggested: **SAM 2.1 Small or Base** for speed on CPU; SAM-HQ ViT-B if edges are rough; SAM 3 ViT-H is the heaviest. Workflow per item:
1. Open the Auto Labeling panel (Ctrl+A), pick the SAM model, and use the point prompt tool.
2. Add a **+ point** on the item (`Q`, or click with the add-point tool), **- points** with `E` to subtract the bowl, garnish or a neighbour; `F` finishes the object.
3. Pick the class from the list (ids from classes.txt). Containers work the same way, clicking the vessel including its rim.
4. Next/previous image: `D` / `A`. Autosave is on. Shortcuts are from the v4.2.3 default config; check Settings (Ctrl+0) if they differ.

Export YOLO-seg: menu **Export > YOLO Seg**, select `ml/xanylabeling/classes.txt` when asked for the classes file, choose an export folder (default `labels/` next to the images). Output is one `.txt` per image: `class_index x1 y1 x2 y2 ...` normalised, where index = line number of classes.txt minus 1 = labels.yaml order. To pre-load labels instead (e.g. to fix them later): **Upload > YOLO Seg**. Keep the per-image `.json` files: they are the lossless source.

Keep `YYYYMMDD_mealNN/` folders in the path. Training scripts split by that folder name.

### 2b. Roboflow Smart Polygon (alternative)

Browser-based, click-to-segment with SAM-class models; faster to start, no install. **Caveat: free-tier (Public plan) projects and their images are public.** Meal photos can include your table, home and belongings, so use Roboflow only for copies of `clean/` (GPS stripped, no faces), or pay for a private workspace. Otherwise prefer X-AnyLabeling. Create the project as *Instance Segmentation*, add the classes with exactly the ids from `classes.txt`, label with Smart Polygon, then export as **YOLOv8/YOLO11 segmentation**; check that the class order in `data.yaml` matches labels.yaml (re-map by name if not).

### 2c. Optional pre-labelling with Claude (`ml/prelabel.py`)

Suggests which classes each photo contains and writes a class-count `summary.csv`. It is a labelling aid; confirm everything yourself, and the app stays offline.
```
pip install -r ml/requirements-prelabel.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."        # PowerShell; never put it in a file
py ml/prelabel.py D:\photos\clean --out D:\photos\prelabel --dry-run
py ml/prelabel.py D:\photos\clean --out D:\photos\prelabel
```
Use the JSON to decide what to click; ignore low-confidence suggestions. It sends images off-device: only use it on cleaned photos. It cannot produce polygons.

## 3. QA pass checklist

Do after every ~100 images and before every training run:
- [ ] Class ids come only from the 19 in labels.yaml; classes.txt was regenerated from the current yaml.
- [ ] Every bowl/plate/bento/cup in the image has a container instance with the rim included; none traced on food only.
- [ ] Mixed bowls are one instance, food only, no bowl wall in the mask.
- [ ] Bubble tea has both `bubble_milk_tea` (contents) and `container_cup` (whole cup).
- [ ] Piles: one instance per pile/plate; not one polygon per piece.
- [ ] No duplicate instances on the same item; no instance with an area under ~0.1% of the image unless the item is truly tiny.
- [ ] No mask bleeds onto the table, chopsticks, hands or lid.
- [ ] Unknown foods left unlabeled; no "closest guess".
- [ ] Spot-check 10% of images by re-opening them in the tool; fix the systematic errors first.
- [ ] Count instances per class (export, then count first tokens of label files, or `summary.csv`); note classes below 150.
- [ ] The held-out test meal folders are not mixed into the training export.
- [ ] 滷肉飯 vs 肉燥飯 and 雞腿 vs 排骨 images re-checked against the shop menu.
