"""Export YOLOE-11s-seg with FIXED text prompts baked in -> web/public/models/locator.onnx + locator.json.

    python ml/snap/export_locator.py                     # reads ml/snap/locator_classes.json (written by tune.py), imgsz from it
    python ml/snap/export_locator.py --superset          # big class list into $SP/snap/superset_<imgsz>.onnx, used by tune.py to cache raw outputs

Classes marked food:false (empty cup, pan, ...) are detected on purpose so empty containers are claimed by a non-food box; the app discards
them. Because the YOLOE head scores every class independently and shares one box regression, a model exported with a subset of classes
is exactly the sub-rows of the superset output; tune.py exploits that to evaluate prompt subsets without re-running the network.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from common import MODELS, SP

HERE = Path(__file__).parent
SUPERSET = [
    # food:true
    ("food", True), ("dish of food", True), ("bowl of food", True), ("plate of food", True), ("drink", True), ("cup", True),
    ("cup of drink", True), ("glass of drink", True), ("bubble tea", True), ("meal on a tray", True),
    ("sandwich", True), ("fruit", True), ("dessert", True),
    # food:false
    ("empty cup", False), ("empty glass", False), ("empty bowl", False), ("empty plate", False), ("frying pan", False), ("wok", False),
    ("cooking pot", False), ("bottle", False), ("mug", False), ("paper cup", False), ("tumbler", False), ("table", False), ("hand", False),
    ("chopsticks", False), ("spoon", False), ("kitchen utensil", False), ("tray", False),
]


def export(classes, imgsz, out_onnx, half_weights=True):
    from ultralytics import YOLOE
    tmp = SP / "snap" / "yoloe_export"; tmp.mkdir(parents=True, exist_ok=True)
    pt = tmp / "yoloe-11s-seg.pt"
    if not pt.exists(): shutil.copy(SP / "export" / "yoloe-11s-seg.pt", pt)
    m = YOLOE(str(pt))
    names = [c[0] for c in classes]
    m.set_classes(names, m.get_text_pe(names))
    p = Path(m.export(format="onnx", imgsz=imgsz, simplify=True, opset=17, dynamic=False))
    if half_weights: halve(p, out_onnx)
    else: shutil.copy(p, out_onnx)


def halve(src, dst):
    """Weight-only fp16 (fp16 initialisers + Cast to fp32 at use). Compute stays fp32 -> works on wasm and webgpu."""
    import onnx
    from onnx import numpy_helper, helper, TensorProto
    m = onnx.load(str(src)); casts, keep = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims or [1])) >= 64:
            new = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__f16")
            keep.append(new); casts.append(helper.make_node("Cast", [new.name], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name))
        else: keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(casts + nodes)
    onnx.save(m, str(dst))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--superset", action="store_true"); ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    if a.superset:
        d = SP / "snap"; d.mkdir(exist_ok=True)
        export(SUPERSET, a.imgsz, d / f"superset_{a.imgsz}.onnx", half_weights=False)
        (d / "superset.json").write_text(json.dumps(SUPERSET), encoding="utf-8")
        print("superset ->", d)
    else:
        cfg = json.loads((HERE / "locator_classes.json").read_text(encoding="utf-8"))
        imgsz = cfg["imgsz"]; classes = [(c["name"], c["food"]) for c in cfg["classes"]]
        MODELS.mkdir(parents=True, exist_ok=True)
        export(classes, imgsz, MODELS / "locator.onnx")
        import onnxruntime as ort
        s = ort.InferenceSession(str(MODELS / "locator.onnx"), providers=["CPUExecutionProvider"])
        i0 = s.get_inputs()[0]; o = s.get_outputs()
        out = dict(imgsz=imgsz, input=i0.name, output=o[0].name, classes=cfg["classes"], conf=cfg["conf"], iou=cfg["iou"],
                   contain=cfg["contain"], max_det=cfg["max_det"], crop_pad=cfg["crop_pad"])
        for k in ("fallback_whole", "min_area_frac"):
            if k in cfg: out[k] = cfg[k]
        (MODELS / "locator.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("outputs", [(x.name, x.shape) for x in o], "input", i0.shape, (MODELS / "locator.onnx").stat().st_size / 1e6, "MB")
