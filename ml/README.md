# ml/: dataset, training, export, evaluation

The loop on Windows + Colab. Class order is `ml/labels.yaml` (19 classes: 15 dishes + 4 containers; index = order, append-only).

| File | Purpose |
|---|---|
| `CAPTURE.md`, `LABELING.md`, `prelabel.py`, `xanylabeling/` | photographing and labelling (other owner) |
| `datasets.md` | seed datasets I researched, with licences, class mappings and what is verified |
| `make_dataset_yaml.py` | builds `data.yaml`; remaps / merges exports; splits by meal |
| `mappings/<name>.yaml` | per-export class remap (`_example.yaml` is the template) |
| `train_seg.ipynb` | Colab notebook: train YOLO11n-seg, validate, export |
| `export_onnx.py` | ONNX export at imgsz 320, opset 17, simplified, plus `detector.labels.json` |
| `eval.py` | held-out evaluation: box/mask mAP50, per-class AP, top-1 dish accuracy, confusion PNG |

## The loop

1. **Capture** meals per `CAPTURE.md`: one folder per meal, `YYYYMMDD_mealNN/IMG_0001.jpg`. Reserve about 15% of the meal folders as the **held-out test set** from day one.
2. **Label** per `LABELING.md` (X-AnyLabeling + SAM, export "YOLO Seg"). Labels sit next to the images or in a `labels/` sibling; the class index is the `labels.yaml` order.
3. **(Optional) Seed data.** Read `datasets.md`. For each seed export create `ml/mappings/<name>.yaml` from `mappings/_example.yaml` using the class names in the export's own `data.yaml`. Unmapped classes are dropped.
4. **Preview locally** (needs only `pip install pyyaml`):
   ```
   py ml/make_dataset_yaml.py --out D:\ds --dry-run ^
       --source D:\photos\train_meals ^
       --source D:\exports\tw_food::ml\mappings\tw_food.yaml
   ```
   Read the per-class table: every class needs data in both train and val (the script flags `NO DATA` / `not in val`). Aim for 150-300 instances per class (PLAN M3). The split is by **session**: a `YYYYMMDD_mealNN` folder, a `<meal>__<n>.jpg` file prefix, or a Roboflow original (augmented copies stay together). Use `--session-regex` for anything else. **Do not** pass the held-out meals as a `--source`.
5. **Zip and upload to Drive** under `MyDrive/calorie-lens/`: `ml/` (the folder), `mine.zip` (your labelled meals), `exports/*.zip` (seed exports), `heldout.zip` (held-out meals, YOLO-seg).
6. **Train in Colab.** Open `ml/train_seg.ipynb` in Colab, choose a T4 runtime, run all cells. It mounts Drive, installs `ultralytics==8.4.193`, builds the dataset, trains `yolo11n-seg.pt` (imgsz 640, 100 epochs, patience 20, mosaic on, fliplr 0.5, default HSV aug), validates, copies `best.pt` to `MyDrive/calorie-lens/models/`, and exports ONNX to `MyDrive/calorie-lens/export/`. Runs live on Drive, so after a disconnect use the `resume=True` line in cell 5.
7. **Web model.** Copy `export/detector.onnx` and `export/detector.labels.json` into `web/public/models/`. `detector.onnx` has the **detect** head (`[1, 23, 2100]`), which is what the current web decoder (`web/src/detector.ts`) reads. The labels file is a JSON array of our class ids in index order, so the web app auto-assigns dishes.
   Manual run (any Python with torch): `py ml/export_onnx.py best.pt --out web/public/models [--also-seg]`.
8. **Core ML.** This is **not** done in Colab. Run the GitHub Actions workflow `.github/workflows/coreml-export.yml` (macOS runner, Python 3.11/3.12 pinned, written by another agent). It takes `best.pt` as input and produces the `.mlpackage` (`yolo export format=coreml nms=True quantize=16`). Check the workflow file for how `best.pt` is supplied (upload / path input).
9. **Evaluate on the held-out meals** (cell 9 of the notebook, or locally):
   ```
   py ml/eval.py --model best.pt --data D:\heldout --imgsz 320 --out eval_out
   ```
   `--imgsz 320` mimics the web model's input size; run once at 640 too. Look at `eval_out/top1_confusion.png` and `metrics.json`. Exit criterion (PLAN M3): at least 85% top-1 dish accuracy on held-out meals.
10. **Iterate.** Find the weak classes (low AP50, or key-pair confusion below), shoot and label more meals of those, retrain.

## Confusions to watch

`eval.py` prints how often each pair is mixed up. Fix them with more varied meals, not with augmentation tweaks.
- 滷肉飯 `lu_rou_fan` vs 白飯 `rice_white`: sauce-covered rice vs plain rice.
- 雞腿 `chicken_leg` / 排骨 `pork_chop` / 雞排 `fried_chicken_cutlet`: all fried or braised brown meat.
- 水餃 `boiled_dumplings` vs 鍋貼 `potstickers`: same filling; the cue is the browned flat bottom.
- PLAN.md also lists 滷肉飯 vs 肉燥飯, which is not a separate class yet.

## ONNX export notes (verified)

Checked against the Ultralytics v8.4.193 source (`ultralytics/nn/modules/head.py`) and by actually exporting a `yolo11n-seg` with 19 classes on CPU:
- Seg ONNX: `output0 = [1, 4+nc+32, N]` (here `[1, 55, 2100]`, N = 40²+20²+10² at imgsz 320) and `proto = [1, 32, 80, 80]` (imgsz/4).
- Detect ONNX: `[1, 4+nc, N]` (here `[1, 23, 2100]`).
- Ultralytics has no built-in "export seg weights as detect" switch. `Segment` subclasses `Detect` with the same box/class branches, so `export_onnx.py` loads the seg weights into a `DetectionModel`, asserts that every tensor of the detect model came from the checkpoint (only `cv4.*` and `proto.*` are ignored), and exports that. Test: for the same input, `detector.onnx` output equals `detector_seg.onnx` output channels `[:4+nc]` exactly (max abs diff 0.0 on a random-weight model).
- If you would rather ship one seg file, run `export_onnx.py --head seg` and change the web decoder to read score channels `4..4+nc` and ignore the trailing 32 coefficients and the proto output.
- `nms=False`: the web code does its own NMS.

## Windows notes

- Local scripts need only `py -m pip install pyyaml` (`make_dataset_yaml.py`). `eval.py` and `export_onnx.py` need `ultralytics`, `onnx`, `onnxslim` and a CPU torch; fine locally but slow for validation, so use Colab.
- A local NVIDIA GPU is not assumed.
