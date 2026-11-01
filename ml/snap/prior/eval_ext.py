"""One configuration per process (module-level vocab in tune_prior comes from env PRIOR_VOCAB / PRIOR_EMB). Prints JSON: results for lambda in LAMS (prior from vocab.json `lp`)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_prior import *
lams = [float(x) for x in sys.argv[1:]] or [0.0, 0.5]
term = np.array([v.get("lp", 0.0) for v in V]); D = Data(); out = {}
for l in lams: out[str(l)] = evaluate(D, term, l, "raw")
print("RESULT" + json.dumps(out))
