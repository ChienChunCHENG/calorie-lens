"""Generate X-AnyLabeling class files from ml/labels.yaml (the single source of truth).

    py ml/xanylabeling/make_classes.py

Writes next to this script:
  classes.txt  one class id per line, in labels.yaml order (line N = YOLO class index N-1).
               Used for (a) `--labels` at launch and (b) the classes file in Export > YOLO-Seg.
  classes_zh.txt  "id  zh  en" cheat sheet for humans.
Uses a tiny parser (no PyYAML needed). Do not hand-edit the outputs.
"""
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
src = here.parent / "labels.yaml"
rows = []
for line in src.read_text(encoding="utf-8").splitlines():
    m = re.match(r"\s*-\s*\{id:\s*(\w+),\s*zh:\s*([^,]+),\s*en:\s*([^,]+?),\s*type:\s*(\w+)\}", line)
    if m:
        rows.append(m.groups())
if len(rows) != 19:
    sys.exit(f"expected 19 classes in {src}, parsed {len(rows)}")
(here / "classes.txt").write_text("\n".join(r[0] for r in rows) + "\n", encoding="utf-8", newline="\n")
(here / "classes_zh.txt").write_text(
    "\n".join(f"{i:2d}  {r[0]:<22} {r[1].strip():<6} {r[2].strip()}  [{r[3]}]" for i, r in enumerate(rows)) + "\n",
    encoding="utf-8", newline="\n")
print(f"wrote {len(rows)} classes to {here}")
