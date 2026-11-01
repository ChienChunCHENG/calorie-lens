"""Build web/public/models/vocab.json + vocab_emb.bin from the r3 food vocabulary plus hand-written negative rows.

    python ml/snap/build_vocab.py
Food rows: text embedding = L2-normalised mean of the normalised embeddings of clip.json "templates" formatted with the English name.
Negative rows (neg:true): mean of the hand-written prompt sentences in snaplib.NEGS ("a cup with nothing in it." ...), because
"a photo of an empty cup" alone is a weak anchor. vocab_emb.bin: float16 little-endian, rows x dim, each row L2-normalised.
"""
import json
import numpy as np
from common import MODELS, load_clip
from snaplib import build_vocab, load_foods, text_matrix

cfg = json.loads((MODELS / "clip.json").read_text(encoding="utf-8"))
tpls = cfg.get("templates") or [cfg["template"]]
neg_mode = cfg.get("neg_prompts", "custom")
model, _, tok = load_clip()
M = text_matrix(model, tok, load_foods(), tpls, neg_mode)
V = build_vocab()
assert len(V) == len(M) and M.shape[1] == cfg["dim"]
M = M / np.linalg.norm(M, axis=1, keepdims=True)
(MODELS / "vocab_emb.bin").write_bytes(M.astype("<f2").tobytes())
(MODELS / "vocab.json").write_text(json.dumps(V, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
print(len(V), "rows;", sum(1 for v in V if v.get("neg")), "negative;", (MODELS / "vocab_emb.bin").stat().st_size / 1e6, "MB;",
      (MODELS / "vocab.json").stat().st_size / 1e6, "MB; templates", tpls, neg_mode)
