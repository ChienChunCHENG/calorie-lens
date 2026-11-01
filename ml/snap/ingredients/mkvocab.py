import json, torch
torch.set_num_threads(2)
from common_i import *
import ingr_vocab as V, ingr
d = dev_labels(); fit = [x for x in d if split(x["dish"]) == "fit"]; print(len(d), len(fit))
rows, n2k = V.build_vocab(fit, str(SP / "ingr_meta.csv"), str(Path(__file__).parent / "ingredients_vocab.json"))
print(len(rows)); E = ingr.text_embs(rows, V.BG); print(E.shape)
unm = {}
for x in d:
    for i in x["ingr"]:
        if i["name"] not in n2k: unm[i["name"]] = unm.get(i["name"], 0) + i["kcal"]
tot = sum(i["kcal"] for x in d for i in x["ingr"]); print("unmapped kcal share", sum(unm.values()) / tot, sorted(unm.items(), key=lambda t: -t[1])[:15])
for r in rows[:70]: print(r["id"], r["kcal_g"], end=" | ")
