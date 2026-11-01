"""Export the MobileCLIP-S1 IMAGE encoder to web/public/models/clip_image.onnx (fp16 weights, fp32 I/O) + clip.json.

    python ml/snap/export_clip.py
Preprocessing is read from open_clip's own transform (not hand-typed). gate/mix are placeholders until tune.py overwrites them.
"""
import json
import numpy as np
import torch
from common import MODELS, load_clip

model, pre, _ = load_clip()
MODELS.mkdir(parents=True, exist_ok=True)


class Img(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m.encode_image(x)  # un-normalised; the app L2-normalises


tf = {type(t).__name__: t for t in pre.transforms}
rs, cc, nm = tf["Resize"], tf["CenterCrop"], tf["Normalize"]
size = rs.size if isinstance(rs.size, int) else rs.size[0]
x = torch.rand(1, 3, size, size)
fp32 = MODELS / "_clip32.onnx"
with torch.no_grad():
    ref = Img(model)(x)
    torch.onnx.export(Img(model).eval(), x, str(fp32), input_names=["image"], output_names=["embedding"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, dynamo=False)
import onnx
from onnx import numpy_helper, helper, TensorProto
m = onnx.load(str(fp32))
# Weight-only fp16: store big float initialisers as fp16 and Cast them back to fp32 at the point of use. (onnxconverter-common's
# full fp16 conversion produced an invalid graph for this model: mismatched Cast types in the attention blocks.) Compute stays fp32,
# so it is portable to onnxruntime-web wasm and webgpu, and the file is half size. Cast of a constant is folded at load.
casts, keep = [], []
for init in m.graph.initializer:
    if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims or [1])) >= 64:
        a = numpy_helper.to_array(init)
        new = numpy_helper.from_array(a.astype(np.float16), init.name + "__f16")
        keep.append(new)
        casts.append(helper.make_node("Cast", [new.name], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name))
    else:
        keep.append(init)
del m.graph.initializer[:]
m.graph.initializer.extend(keep)
nodes = list(m.graph.node)
del m.graph.node[:]
m.graph.node.extend(casts + nodes)
onnx.save(m, str(MODELS / "clip_image.onnx"))
import onnxruntime as ort
def run(p, a):
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]); return s.run(None, {"image": a})[0]
xs = np.random.RandomState(0).rand(4, 3, size, size).astype(np.float32)
a, b = run(fp32, xs), run(MODELS / "clip_image.onnx", xs)
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print("torch vs onnx fp32 maxabs", float(np.abs(a[:1] - ref.numpy()).max()), "| fp16 vs fp32 cosine", cos)
assert cos.min() > 0.999
fp32.unlink()
cfg = dict(input="image", output="embedding", dtype="float32", size=size,
           resize="shortest", crop="center", interpolation=str(rs.interpolation).split(".")[-1].lower() if hasattr(rs.interpolation, "name") else "bilinear",
           mean=list(map(float, nm.mean)), std=list(map(float, nm.std)), dim=int(ref.shape[1]),
           logit_scale=float(model.logit_scale.exp()), template="a photo of {}.",
           gate=dict(p_neg_max=0.5, p_food_min=0.05), mix=dict(k=5, cum=0.9, sure=0.5))
cfgp = MODELS / "clip.json"
if cfgp.exists():  # keep tuned values on re-export
    old = json.loads(cfgp.read_text(encoding="utf-8"))
    for k in ("gate", "mix", "template", "resize", "crop", "templates"):
        if k in old: cfg[k] = old[k]
cfgp.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
print("wrote", MODELS / "clip_image.onnx", (MODELS / "clip_image.onnx").stat().st_size / 1e6, "MB;", cfg)
