"""Research export: shipped 14 locator prompts + container prompts (plate, bowl, tray) -> $SP/research/portion/loc17.onnx (fp32)."""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from common import MODELS, SP
import export_locator as E, shutil
L = json.load(open(MODELS / "locator.json"))
classes = [(c["name"], c["food"]) for c in L["classes"]] + [("plate", False), ("bowl", False), ("tray", False)]
out = SP / "research/portion/loc17.onnx"
E.export(classes, 640, out, half_weights=False)
json.dump(classes, open(SP / "research/portion/loc17.json", "w"))
print(out)
