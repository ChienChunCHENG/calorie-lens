#!/usr/bin/env python3
"""Export a trained Calorie Lens YOLO11-seg checkpoint to ONNX for the web app.

    py ml/export_onnx.py runs/segment/train/weights/best.pt --out web_models
    -> web_models/detector.onnx          (DETECT head, output [1, 4+nc, N])   <- web app default
       web_models/detector.labels.json   (JSON array of our class ids, index order)
       web_models/detector_seg.onnx      (SEG head, only with --also-seg)

Output tensor facts (verified against ultralytics v8.4.193 source, nn/modules/head.py):
  * Detect.forward in export mode returns the decoded tensor y = cat(box[4], cls[nc]) -> [1, 4+nc, N]
    (boxes are xywh in input pixels, class scores already sigmoided). N = 2100 at imgsz 320
    (40*40 + 20*20 + 10*10).
  * Segment._inference does torch.cat([detect_preds, mask_coefficient], dim=1), and
    Segment.forward in export mode returns (that_tensor, proto).  So the SEG ONNX has
      output0 = [1, 4 + nc + 32, N]     (channels 4..4+nc are class scores, the last 32 are mask coeffs)
      output1 = [1, 32, 80, 80]         (mask prototypes; imgsz/4 -> 80x80 at imgsz 320)
  * The current web decoder (web/src/detector.ts) only handles detection [1, 4+nc, N].

DECISION: the web app gets a DETECT-head export built from the SEG weights.
  Segment subclasses Detect and shares the exact same box/class branches (cv2/cv3, layer 23 in
  yolo11*.yaml), so a plain DetectionModel(nc=...) can take every tensor except cv4.* and proto.*.
  That is the same key-intersection Ultralytics itself does when it transfers weights between
  tasks (ultralytics.utils.torch_utils.intersect_dicts / BaseModel.load). Ultralytics does NOT have a
  one-line "export seg weights as detect" option, so we do it by hand here and then ASSERT that every
  cv2/cv3 tensor was loaded (so a silent half-random model is impossible).
  Fallback if you prefer a single file: export --head seg and make the web decoder slice channels
  4..4+nc for scores and ignore the trailing 32 coefficients (and the proto output).

Needs: pip install ultralytics onnx onnxslim (simplify) -- run in Colab / any Python with torch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def class_ids(labels_yaml: Path) -> list[str]:
    return [c["id"] for c in yaml.safe_load(labels_yaml.read_text(encoding="utf-8"))["classes"]]


def make_detect_from_seg(seg_pt: Path, ids: list[str]):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    seg = YOLO(str(seg_pt))
    head_cls = type(seg.model.model[-1]).__name__  # most reliable: look at the real last layer
    if not head_cls.startswith("Segment"):
        sys.exit(f"{seg_pt} has a {head_cls} head, expected a Segment head")
    nc = len(seg.model.names)
    if nc != len(ids):
        sys.exit(f"checkpoint has {nc} classes but labels.yaml has {len(ids)}; retrain or fix labels.yaml")
    scale = seg.model.yaml.get("scale") or "n"
    # Same architecture YAML family as the seg model, but with the Detect head.
    cfg = dict(seg.model.yaml)
    cfg["head"] = list(cfg["head"])
    last = list(cfg["head"][-1])
    if last[2] not in ("Segment", "Segment26"):
        sys.exit(f"unexpected last layer {last[2]}")
    last[2] = "Detect"
    last[3] = [nc]  # Detect args: [nc]  (Segment args were [nc, 32, 256])
    cfg["head"][-1] = last
    cfg["nc"] = nc
    det = DetectionModel(cfg, nc=nc, verbose=False)
    sd = {k: v.float() for k, v in seg.model.state_dict().items()}
    missing, unexpected = det.load_state_dict(sd, strict=False)
    # every tensor of the Detect model (backbone, neck, cv2 box, cv3 cls) must come from the checkpoint
    if missing:
        sys.exit(f"detect model has tensors not found in the seg checkpoint: {missing[:8]}")
    skipped = sorted({k.split(".")[2] for k in unexpected if k.startswith("model.")})  # cv4, proto
    print(f"loaded seg weights into Detect model; ignored seg-only tensors: {skipped}")
    det.names = {i: n for i, n in enumerate(ids)}
    det.args = getattr(seg.model, "args", {})
    det.task = "detect"  # the exporter reads model.task / model.pt_path (set by the loader for real checkpoints)
    det.pt_path = str(seg_pt)
    y = YOLO(str(seg_pt))  # cheap wrapper to reuse the exporter plumbing
    y.model = det
    y.task = "detect"
    y.overrides["task"] = "detect"
    return y


def verify_onnx(path: Path, nc: int, imgsz: int, head: str) -> None:
    import onnx

    m = onnx.load(str(path))
    outs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
    n = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    want0 = [1, 4 + nc + (32 if head == "seg" else 0), n]
    first = list(outs.values())[0]
    print(f"{path.name}: outputs {outs}")
    if first != want0:
        sys.exit(f"unexpected output0 shape {first}, wanted {want0}")
    if head == "detect" and len(outs) != 1:
        sys.exit("detect export should have exactly one output")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="best.pt from yolo11n-seg training")
    ap.add_argument("--out", default=".", help="output dir")
    ap.add_argument("--labels", default=str(HERE / "labels.yaml"))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--head", choices=["detect", "seg"], default="detect", help="head for detector.onnx (default detect)")
    ap.add_argument("--also-seg", action="store_true", help="additionally write detector_seg.onnx (full seg head)")
    a = ap.parse_args(argv)

    ids = class_ids(Path(a.labels))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Path(a.weights)

    from ultralytics import YOLO

    kw = dict(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True, dynamic=False, half=False, nms=False, batch=1)

    def export(model, dest: Path, head: str):
        res = Path(model.export(**kw))
        shutil.move(str(res), dest)
        verify_onnx(dest, len(ids), a.imgsz, head)
        print(f"wrote {dest}")

    if a.head == "detect":
        export(make_detect_from_seg(w, ids), out / "detector.onnx", "detect")
        if a.also_seg:
            export(YOLO(str(w)), out / "detector_seg.onnx", "seg")
    else:
        export(YOLO(str(w)), out / "detector.onnx", "seg")

    # Class names in INDEX ORDER. The web app auto-assigns dishes when these are dish ids.
    (out / "detector.labels.json").write_text(json.dumps(ids, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out / 'detector.labels.json'} ({len(ids)} ids)")
    print("copy detector.onnx + detector.labels.json to web/public/models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
