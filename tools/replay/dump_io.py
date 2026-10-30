"""Read / write the sweep dump format (see DUMP_FORMAT.md). numpy + Pillow only."""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile

import numpy as np

from cl_types import Frame, Plane

FORMAT_VERSION = 1
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def save_dump(frames: list[Frame], out_dir: str, meta: dict | None = None, rgb_jpegs: list | None = None,
              include_depth: bool = False) -> str:
    """Write frames to out_dir. meta (all optional): device_model, ios_version, image_size [w,h], depth_model,
    plane {"n":[3],"offset":x,"source":str}, containers {item_key: dict}. rgb_jpegs: per-frame JPEG bytes or None.
    include_depth writes depth_NN.f32 (metric depth; only for debugging/synthetic ground truth)."""
    from PIL import Image
    meta = dict(meta or {})
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, f in enumerate(frames):
        e = {"index": i, "t": float(f.t),
             "T_wc": [float(x) for x in np.asarray(f.T_wc, np.float64).reshape(16)],
             "K": [float(x) for x in np.asarray(f.K, np.float64).reshape(9)]}
        if f.rel_depth is None:
            raise ValueError(f"frame {i} has no rel_depth")
        H, W = f.rel_depth.shape
        name = f"rel_{i:02d}.f16"
        with open(os.path.join(out_dir, name), "wb") as fh:
            fh.write(np.ascontiguousarray(f.rel_depth, dtype="<f2").tobytes())
        e["rel"] = {"file": name, "shape": [H, W]}
        e["masks"] = {}
        for key, m in f.masks.items():
            if not _KEY_RE.match(key):
                raise ValueError(f"item key {key!r} must match [A-Za-z0-9_-]+")
            if m.shape != (H, W):
                raise ValueError(f"mask {key} shape {m.shape} != rel shape {(H, W)}")
            mname = f"mask_{i:02d}_{key}.png"
            Image.fromarray((m.astype(np.uint8) * 255), mode="L").save(os.path.join(out_dir, mname))
            e["masks"][key] = {"file": mname, "label": f.labels.get(key)}
        fp = np.zeros((0, 3), np.float32) if f.feature_points_w is None else np.asarray(f.feature_points_w, np.float32).reshape(-1, 3)
        fname = f"fp_{i:02d}.f32"
        with open(os.path.join(out_dir, fname), "wb") as fh:
            fh.write(np.ascontiguousarray(fp, dtype="<f4").tobytes())
        e["feature_points"] = {"file": fname, "count": int(len(fp))}
        jpg = rgb_jpegs[i] if rgb_jpegs else None
        if jpg is not None:
            rname = f"rgb_{i:02d}.jpg"
            with open(os.path.join(out_dir, rname), "wb") as fh:
                fh.write(jpg)
            e["rgb"] = rname
        else:
            e["rgb"] = None
        if include_depth and f.depth_m is not None:
            dname = f"depth_{i:02d}.f32"
            with open(os.path.join(out_dir, dname), "wb") as fh:
                fh.write(np.ascontiguousarray(f.depth_m, dtype="<f4").tobytes())
            e["depth"] = {"file": dname, "shape": [H, W]}
        entries.append(e)
    manifest = {"version": FORMAT_VERSION,
                "device_model": meta.get("device_model"), "ios_version": meta.get("ios_version"),
                "image_size": meta.get("image_size"), "depth_model": meta.get("depth_model"),
                "depth_semantics": "disparity_larger_is_nearer",
                "plane": meta.get("plane"), "containers": meta.get("containers") or {},
                "frames": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return out_dir


def _resolve_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    if zipfile.is_zipfile(path):
        tmp = tempfile.mkdtemp(prefix="sweepdump_")
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        if os.path.exists(os.path.join(tmp, "manifest.json")):
            return tmp
        subs = [d for d in os.listdir(tmp) if os.path.exists(os.path.join(tmp, d, "manifest.json"))]
        if len(subs) == 1:
            return os.path.join(tmp, subs[0])
        raise FileNotFoundError("manifest.json not found in zip")
    raise FileNotFoundError(path)


def load_dump_with_meta(path: str) -> tuple[list[Frame], dict]:
    """Load a dump directory or .zip. Returns (frames, manifest dict). plane in the manifest is also exposed as
    manifest['plane_obj'] (Plane or None)."""
    from PIL import Image
    d = _resolve_dir(path)
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported dump version {man.get('version')}")
    frames = []
    for e in man["frames"]:
        H, W = e["rel"]["shape"]
        rel = np.fromfile(os.path.join(d, e["rel"]["file"]), dtype="<f2")
        if rel.size != H * W:
            raise ValueError(f"{e['rel']['file']}: {rel.size} values, expected {H * W}")
        rel = rel.reshape(H, W).astype(np.float32)
        masks, labels = {}, {}
        for key, me in e["masks"].items():
            arr = np.array(Image.open(os.path.join(d, me["file"])).convert("L"))
            if arr.shape != (H, W):
                raise ValueError(f"{me['file']}: shape {arr.shape} != {(H, W)}")
            masks[key] = arr > 127
            if me.get("label") is not None:
                labels[key] = me["label"]
        fp = np.fromfile(os.path.join(d, e["feature_points"]["file"]), dtype="<f4").reshape(-1, 3)
        depth = None
        if e.get("depth"):
            depth = np.fromfile(os.path.join(d, e["depth"]["file"]), dtype="<f4").reshape(H, W)
        frames.append(Frame(t=float(e["t"]), T_wc=np.array(e["T_wc"], np.float64).reshape(4, 4),
                            K=np.array(e["K"], np.float64).reshape(3, 3), rel_depth=rel, depth_m=depth,
                            masks=masks, labels=labels, feature_points_w=fp))
    pl = man.get("plane")
    man["plane_obj"] = Plane(n=np.array(pl["n"], np.float64), offset=float(pl["offset"])) if pl else None
    return frames, man


def load_dump(path: str) -> list[Frame]:
    return load_dump_with_meta(path)[0]
