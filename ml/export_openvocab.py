"""Export an open-vocabulary detector (Ultralytics YOLOE, text prompts) for the web lens, no training data needed.

YOLOE takes class names as text, embeds them once with its MobileCLIP text encoder, and folds the embeddings into the
head, so the exported ONNX is a plain YOLO model with our dish prompts as its classes. Several prompts can map to one
dish id (PROMPTS below); the web app merges them (detector.labels.json lists one dish id per model class).

    pip install ultralytics onnx onnxslim
    python ml/export_openvocab.py --out web/public/models          # writes detector.onnx + detector.labels.json
    python ml/export_openvocab.py --test path/to/photos            # also prints top detections per photo

Once a trained model exists (ml/train_seg.ipynb), export that instead; it beats zero-shot prompts on Taiwanese dishes.
"""
import argparse
import json
import shutil
from pathlib import Path

# dish id -> English prompts, tuned on 22 Wikimedia photos (yoloe-11s, imgsz 416, conf 0.35: top-1 dish right on 13/22,
# 11 wrong-dish boxes). Background prompts (plate, bowl, ...) and the m/l checkpoints scored no better.
# Ids must exist in data/nutrition.json; containers are not prompted (no depth on web).
PROMPTS: dict[str, list[str]] = {
    "rice_white": ["bowl of white rice", "steamed white rice"],
    "lu_rou_fan": ["braised pork rice", "rice topped with minced pork sauce"],
    "chicken_leg": ["fried chicken drumstick", "braised chicken leg"],
    "pork_chop": ["fried pork chop"],
    "braised_egg": ["braised soy sauce egg", "brown boiled egg"],
    "fried_egg": ["fried egg", "sunny side up egg"],
    "stirfried_greens": ["stir-fried green vegetables", "cooked leafy greens"],
    "braised_tofu": ["braised tofu"],
    "beef_noodle_soup": ["beef noodle soup", "bowl of noodle soup"],
    "boiled_dumplings": ["boiled dumplings"],
    "potstickers": ["potstickers", "pan-fried dumplings", "gyoza"],
    "scallion_pancake": ["scallion pancake", "green onion pancake"],
    "fried_chicken_cutlet": ["large fried chicken cutlet"],
    "popcorn_chicken": ["popcorn chicken", "bite-sized fried chicken pieces"],
    "bubble_milk_tea": ["bubble milk tea", "bubble tea"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yoloe-11s-seg.pt", help="YOLOE text-prompt checkpoint (auto-downloads)")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--out", default="web/public/models")
    ap.add_argument("--test", help="folder of photos to run the prompted model on (sanity check)")
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()

    from ultralytics import YOLOE

    names, ids = [], []
    for dish, ps in PROMPTS.items():
        for p in ps:
            names.append(p)
            ids.append(dish)
    model = YOLOE(a.weights)
    try:
        model.set_classes(names, model.get_text_pe(names))
    except TypeError:  # newer ultralytics computes the text embeddings itself
        model.set_classes(names)

    if a.test:
        for img in sorted(Path(a.test).glob("*.*")):
            r = model.predict(str(img), imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
            dets = sorted(((float(b.conf), ids[int(b.cls)], names[int(b.cls)]) for b in r.boxes), reverse=True)[:4]
            print(f"{img.name:32s} " + ", ".join(f"{d}({p}) {c:.2f}" for c, d, p in dets) if dets else f"{img.name:32s} -")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx = Path(model.export(format="onnx", imgsz=a.imgsz, simplify=True, opset=17, dynamic=False))
    shutil.copy(onnx, out / "detector.onnx")
    (out / "detector.labels.json").write_text(json.dumps(ids) + "\n", encoding="utf-8")
    print(f"wrote {out / 'detector.onnx'} ({(out / 'detector.onnx').stat().st_size / 1e6:.1f} MB), {len(ids)} classes")


if __name__ == "__main__":
    main()
