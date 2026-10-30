"""Print the Core ML spec (inputs/outputs) of an .mlpackage.

Usage: inspect_model.py PKG [OUT_TXT]
"""
import sys

import coremltools as ct
from coremltools.proto import FeatureTypes_pb2 as ft


def describe(f):
    t = f.type
    kind = t.WhichOneof("Type")
    if kind == "multiArrayType":
        m = t.multiArrayType
        return (f"{f.name}: multiArray shape={list(m.shape)} "
                f"dtype={ft.ArrayFeatureType.ArrayDataType.Name(m.dataType)}")
    if kind == "imageType":
        i = t.imageType
        cs = ft.ImageFeatureType.ColorSpace.Name(i.colorSpace)
        return f"{f.name}: image {i.width}x{i.height} {cs}"
    return f"{f.name}: {kind}"


def main():
    pkg = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = m.get_spec()
    lines = [f"model: {pkg}", f"spec type: {spec.WhichOneof('Type')}", "INPUTS"]
    lines += ["  " + describe(f) for f in spec.description.input]
    lines.append("OUTPUTS")
    lines += ["  " + describe(f) for f in spec.description.output]
    md = dict(m.user_defined_metadata)
    lines.append("METADATA KEYS: " + ", ".join(sorted(md)))
    for k in ("task", "imgsz", "names", "args"):
        if k in md:
            lines.append(f"  {k}: {md[k][:300]}")
    text = "\n".join(lines)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
