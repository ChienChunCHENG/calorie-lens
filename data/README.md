# Calorie Lens nutrition data

## Sources and licences

| Source | Use | Licence |
|---|---|---|
| 衛福部食藥署 食品營養成分資料集 (TFDA), data.gov.tw/dataset/8543, export `https://data.fda.gov.tw/data/opendata/export/20/csv` (zip containing `20_2.csv`, UTF-8 with BOM, 230,152 long-format rows, 2,213 foods, downloaded 2026-10-30, file dated 2026-10-05) | all kcal and macro anchors | 政府資料開放授權條款-第1版 (Open Government Data License v1.0). Attribution required. |
| 國健署 食物代換表 | rice cross-check (recalled from memory, not re-verified) | public government document |
| `data/priors/*.yaml` | densities, oil adders, portions, containers, drinks | see each file's `source` fields (owned by the priors workstream) |

Raw downloads live in `data/sources/` (`tfda_export_20.zip`, `tfda_raw/20_2.csv`) and are git-ignored by contract.

## TFDA pivot

`data/build_tfda_pivot.py` turns the long CSV into `data/tfda_pivot.json` (2,213 foods; 2,160 have all four macros).
Each object: `tfda_id, name, alias, english, category, description, per100g{kcal,kcal_modified,protein,fat,carb}, n{...}, sd{...}`.

* Energy analyte: **熱量** is used as `kcal`. 熱量 equals 4*P + 9*F + 4*total-carb within about 5 kcal on average; 修正熱量 (fibre-adjusted) averages about 10 kcal lower and differs from 熱量 in about half the foods. It is kept as `kcal_modified`. Choosing 熱量 is the conservative (higher) option, in line with the project's bias against under-estimation.
* Protein = 粗蛋白, fat = 粗脂肪, carb = 總碳水化合物.
* TFDA reports `樣本數` and `標準差` only for the measured analytes, never for energy. `n.kcal` is the minimum n across P/F/C and `sd.kcal` is propagated from the macro sds (`sqrt((4 sd_P)^2 + (9 sd_F)^2 + (4 sd_C)^2)`); it is null when no macro sd exists.
* Rows whose `tfda_id` has 5 digits (for example `E30001`, `K01121`, `I04044`) are TFDA group averages (名稱 ends with 平均值).
* Almost all rows are raw ingredients; prepared dishes are rare.

## Dishes

`data/dishes.yaml` holds the 15 contract classes (ids and kinds from `spec/data_contract.md`).

* `kcal_100g` is per 100 g of the dish as served (cooked, edible portion). Rice is COOKED rice.
* For fried or stir-fried dishes it **excludes absorbed cooking oil**: `oil.yaml` supplies the oil adder, so including it here would double count. (Example: fried egg uses TFDA 荷包蛋 *without* oil, 162, rather than the 1 tsp oil row, 192.)
* min/typ/max is a 90% interval. Rule for direct TFDA matches: typ = TFDA mean; if n > 1, min/max = typ +/- 1.645 sd (sd propagated from macro sds), otherwise +/-15%. Exceptions are documented in each dish's `method_notes` (rice min lowered to 140; braised egg min lowered to 175).
* Composed dishes list their `components` (weight fraction, kcal/100 g, range) and assumptions (cooking yields, coating share, broth share); min/max combine component ranges with fully correlated weights, which is conservative.
* `origin` is `direct` or `composed`; `confidence` is high/med/low.

| id | zh | origin | TFDA basis | kcal/100 g typ (min-max) | confidence |
|---|---|---|---|---|---|
| rice_white | 白飯 | TFDA direct | A0550601 | 183 (140-200) | med |
| lu_rou_fan | 滷肉飯 | composed | rice 80% + 肉燥 P1000301 20% | 226 (172-250) | low |
| chicken_leg | 雞腿 | composed | raw 雞腿 I0404302/I0404102, yield 0.75 | 232 (185-285) | low |
| pork_chop | 排骨 | composed | raw 豬大排 I0304201 + starch coating | 287 (245-346) | low |
| braised_egg | 滷蛋 | TFDA direct | K0120101 (n=3) | 191 (175-202) | high |
| fried_egg | 荷包蛋 | TFDA direct | K0123101 no-oil (n=2) | 162 (155-169) | high |
| stirfried_greens | 炒青菜 | composed | raw cabbage E30001 + 青江菜, yield 0.75 | 24.5 (16-36.5) | med |
| braised_tofu | 滷豆腐 | composed | 傳統豆腐 R4700901/R4700902 | 105 (80-140) | low |
| beef_noodle_soup | 牛肉麵 | composed | noodles + broth + beef | 94 (71-125) | low |
| boiled_dumplings | 水餃 | TFDA direct (similar product) | R2200801, R2201501 (cooked, n=1 each) | 217 (184-250) | med |
| potstickers | 鍋貼 | TFDA direct | R2201401 (n=1) | 238 (202-274) | med |
| scallion_pancake | 蔥油餅 | TFDA direct | R2700101 (n=1) | 305 (259-351) | med |
| fried_chicken_cutlet | 雞排 | composed | raw 雞排 I04044 + coating | 253 (198-309) | low |
| popcorn_chicken | 鹽酥雞 | composed | raw 雞排 I04044 + 25% coating | 259 (204-315) | low |
| bubble_milk_tea | 珍珠奶茶 | TFDA direct | O0702101 半糖 (n=4); indicative only | 84 (71-97) | med |

Drinks: the app computes drink kcal from `priors/drinks.yaml` (sugar_g * 4 + base + toppings); the TFDA 84 kcal/100 g (about 590 kcal for 700 mL, half sugar) is a sanity check, not an input.

## Rebuild

```
py -m pip install --user pyyaml
py -I -X utf8 data/build_tfda_pivot.py     # needs data/sources/tfda_export_20.zip
py -X utf8 data/build_nutrition.py         # dishes.yaml + priors/*.yaml -> nutrition.json
```

Re-download: `curl -L -o data/sources/tfda_export_20.zip https://data.fda.gov.tw/data/opendata/export/20/csv`.

`build_nutrition.py` tolerates missing or partial priors (nulls plus warnings). It exits 1 if any of the 15 ids is missing, a `kind` differs from the contract, or any `min <= typ <= max` ordering is violated. `data/dishes.yaml` was produced by a one-off generator script; edit the YAML directly (or regenerate consistently) and keep the `components` and `method_notes` in step with the numbers.

## Output: `nutrition.json`

`{version, generated, schema_note, dishes[], containers, drinks}`. Each dish: `id, zh, en, kind, density_g_ml, kcal_100g, macros_100g, oil_g_per_100g (+oil_method), default_portion_g, container_prior, tfda_refs, origin, source, confidence, method_notes`. `containers` and `drinks` are passed through from `priors/containers.yaml` and `priors/drinks.yaml`.
