# Data contract (Wave 1)

This file is the single agreement between the nutrition-data, priors, range-engine and app workstreams. Change it only through Opus QC.

## Class ids (first 15). Every workstream must use these exact ids.

| id | zh | en | kind |
|---|---|---|---|
| rice_white | 白飯 | Steamed white rice | solid |
| lu_rou_fan | 滷肉飯 | Braised pork rice | solid |
| chicken_leg | 雞腿 | Chicken drumstick (fried or braised) | solid |
| pork_chop | 排骨 | Pork chop (fried) | solid |
| braised_egg | 滷蛋 | Braised egg | solid |
| fried_egg | 荷包蛋 | Fried egg | solid |
| stirfried_greens | 炒青菜 | Stir-fried greens | solid |
| braised_tofu | 滷豆腐 | Braised tofu | solid |
| beef_noodle_soup | 牛肉麵 | Beef noodle soup (whole bowl) | solid |
| boiled_dumplings | 水餃 | Boiled dumplings | solid |
| potstickers | 鍋貼 | Potstickers | solid |
| scallion_pancake | 蔥油餅 | Scallion pancake | solid |
| fried_chicken_cutlet | 雞排 | Fried chicken cutlet | solid |
| popcorn_chicken | 鹽酥雞 | Taiwanese popcorn chicken | solid |
| bubble_milk_tea | 珍珠奶茶 | Bubble milk tea | drink |

## Range triple

Every uncertain number is written `{min, typ, max}`, with `min ≤ typ ≤ max`. The bounds are read as a **90% interval** (5th to 95th percentile).

## File ownership

| File | Owner | Content |
|---|---|---|
| `data/sources/*` | nutrition + priors agents | Raw downloads. Git-ignored. |
| `data/tfda_pivot.json` | nutrition agent | TFDA long → wide: one object per food, `{tfda_id, name, alias, category, per100g:{kcal,protein,fat,carb}, n, sd:{...}}` |
| `data/dishes.yaml` | nutrition agent | Per class id: `zh, en, kind, kcal_100g{min,typ,max}, macros_100g{p,f,c}, tfda_refs[], method_notes, source[], confidence(high/med/low)`. Drinks: macros may be null, because kcal comes from `drinks.yaml`. |
| `data/priors/density.yaml` | priors agent | Per solid id: `density_g_ml{min,typ,max}, bulk:bool, source[], confidence` |
| `data/priors/oil.yaml` | priors agent | `methods: {stir_fry, deep_fry_breaded, pan_fry, braise, boil, steam, none}` → `oil_g_per_100g{min,max}`, plus `dish_method: {id: method}` |
| `data/priors/portions.yaml` | priors agent | Per id: `default_portion_g{min,typ,max}, container_prior[]` |
| `data/priors/containers.yaml` | priors agent | Container id → `capacity_ml{min,typ,max}, rim_diameter_cm, depth_cm, shape(bowl/plate/box/cup)`. Ids: `bowl_small, bowl_large, bento_box, plate_26cm, cup_500, cup_700` |
| `data/priors/drinks.yaml` | priors agent | `sugar_levels: {full, less, half, light, none}` → fraction of full, plus `full_sugar_g_per_700ml`. Per drink id: `base_kcal_per_700ml{min,typ,max}, toppings_kcal{min,typ,max}`. Also cites the MOHW table. |
| `data/build_nutrition.py` + `data/nutrition.json` | nutrition agent | Merges dishes and priors into the per-dish schema in PLAN.md §5, plus top-level `containers` and `drinks` |
| `spec/range_engine.md`, `spec/vectors.json` | range-engine agent | Math spec and golden vectors (self-contained fake inputs, not tied to real data) |
| `core/py/range_engine.py` | range-engine agent | Reference implementation, which generates the vectors |
| `core/ts/rangeEngine.ts` (+ test) | range-engine agent | TypeScript port, which must pass the vectors |
| `web/**` | web agent | Vite app. Must not edit `core/`. |
| `ios/**`, `.github/workflows/**` | iOS agent | |

## Range engine API (TS; Swift later mirrors it)

```ts
type Triple = { min: number; typ: number; max: number };
interface DishEntry { id: string; kind: 'solid'|'drink'; density_g_ml?: Triple; kcal_100g: Triple;
  oil_g_per_100g?: { min: number; max: number }; default_portion_g?: Triple; }
interface ItemEstimate { id: string; lo: number; med: number; hi: number; flags: string[] } // kcal
estimateSolid(dish: DishEntry, volumeMl: Triple | null): ItemEstimate   // null → default portion, flag 'portion_default'
estimateDrink(drink: DrinkEntry, sizeMl: number, sugarLevel: 'full'|'less'|'half'|'light'|'none', fullSugarGPer700: number, sugarFraction: Record<string,number>): ItemEstimate
combineMeal(items: ItemEstimate[]): { lo: number; med: number; hi: number }
```

## Wave 1 QC decisions (Opus, 2026-10-30)

1. **Oil convention:** `dishes.yaml` kcal for fried, stir-fried and pan-fried dishes **excludes** cooking oil, and `oil.yaml` supplies it. `adder_applies: true` is set for chicken_leg, pork_chop, fried_egg, stirfried_greens, potstickers, scallion_pancake, fried_chicken_cutlet and popcorn_chicken. Braised, boiled and steamed dishes already include their fat, so the flag is `false` for those. `build_nutrition.py` emits `oil_g_per_100g: null` when the adder does not apply.
2. **As-served basis:** kcal_100g is per 100 g **as served, including bone**, because portions and measured volumes include bone. chicken_leg uses an edible fraction of 0.70/0.80/1.00 and pork_chop 0.85/0.90/1.00. (pork_chop macros are still on an edible basis; they are not displayed yet.)
3. **Drinks:** `drinks.full_sugar_g_per_700ml` is a triple. Callers pass `.typ` to `estimateDrink`, because the engine applies its own ±25% shop-variance band. `sugar_levels` are *total-sugar* ratios from the MOHW table (milk and pearl sugars remain at 無糖), not "fraction of added sugar".
4. **Range engine decisions accepted:**
   - Oil uses the median mass for both bounds, and its midpoint is added to med.
   - Toppings are not size-scaled.
   - `low_confidence` is computed before oil and the floor.
5. **Sanity results:**
   - 700 mL 珍奶: 全糖 651 and 半糖 584 kcal median, against MOHW 658 and 588.
   - 雞腿便當 + 700 半糖珍奶: range 1311–2106 kcal, median 1634. The 國健署 example of ≈1375 is inside the range, in its lower part.
   - Default portions may run high; recalibrate in M5.
