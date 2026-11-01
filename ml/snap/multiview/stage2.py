"""Stage 2 (per video): Depth Anything V2 small relative inverse depth for each stage-1 frame.
usage: python stage2.py <name>  -> $SP/research/mv/s2_<name>/<dish>_<cam>.npy   float16 (n, 322, 574) disparity-like (bigger = closer)"""
import sys, time
from pathlib import Path
import numpy as np, torch, pickle
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import SP
torch.set_num_threads(3)
MV = SP / "research/mv"; IN_H, IN_W = 322, 574
_m = None


def model():
    global _m
    if _m is None:
        from transformers import AutoModelForDepthEstimation
        _m = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()
    return _m


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def depth_frames(frames):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    x = torch.nn.functional.interpolate(x, size=(IN_H, IN_W), mode="bilinear", antialias=True, align_corners=False)
    x = (x - MEAN) / STD
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 4): out.append(model()(pixel_values=x[i:i + 4]).predicted_depth.numpy())
    return np.concatenate(out).astype(np.float16)


if __name__ == "__main__":
    name = sys.argv[1]; src = MV / f"s1_{name}"; dst = MV / f"s2_{name}"; dst.mkdir(exist_ok=True)
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    for p in sorted(src.glob("*.pkl"))[:lim]:
        o = dst / (p.stem + ".npy")
        if o.exists(): continue
        t = time.time(); d = pickle.load(open(p, "rb")); np.save(o, depth_frames(d["frames"])); print(p.stem, f"{time.time()-t:.1f}s", flush=True)
