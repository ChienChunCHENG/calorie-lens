# Priors (density, oil, portions, containers, drinks)

Validate: `py data/priors/check_priors.py` (parses all YAML, checks ids against spec/data_contract.md and min <= typ <= max).
Range triples are 90% intervals per the contract. Every number carries a `source` and `confidence`; "reasoned" means no measurement behind it.

## Sources
| Source | Used for | Local copy | Licence / notes |
|---|---|---|---|
| FAO/INFOODS Density DB v2.0 (https://www.fao.org/fileadmin/templates/food_composition/documents/density_DB_v2_0_final-1__1_.xlsx) | rice, soup, noodle, bread/crumb densities; cited by sheet "Density DB" Excel row | `data/sources/priors_fao_density.xlsx` | FAO/INFOODS terms: free for non-commercial use with attribution; check before commercial redistribution. Few cooked/prepared Asian foods, so most solids are analogues. |
| USDA FoodData Central (api.nal.usda.gov, DEMO_KEY) | cup-weight cross checks | none | Public domain (CC0). DEMO_KEY hit HTTP 429 so **USDA figures are recalled from memory and marked unverified**. |
| MOHW 手搖飲 sugar table PDF (https://mohw.gov.tw/dl-38434-1f700ea4-1843-4a74-a23c-91b98158d2b0.html) | 珍珠奶茶 sugar/kcal, 700 mL | `data/sources/priors_mohw_sugar.pdf` (5 pages, table on p.5) | Taiwan government open-data licence (attribute MOHW). Its own basis is TFDA food DB. |
| Pu et al. 2019, IJERPH 16(18):3367, doi:10.3390/ijerph16183367 (PMC6765806) | oil used per 100 g in Chinese dishes | none | Open access (CC BY 4.0 per MDPI). |
| Pan et al. 1993, J Am Diet Assoc 93:1442-4 (ir.sinica.edu.tw/handle/201000000A/23518) | Taiwan stir-fry oil absorption (qualitative only) | none | Abstract-level only; not read. |
| NC A&T oleogel frying study; oil-uptake review (HAL hal-01194942) | fried chicken ~15% fat | none | Snippet-level only, full text not read. |
| 董氏基金會 reports via PTS / UDN / Taiwan News | 半糖 reality, labelling rule | none | Press coverage, years not confirmed. |
| 國健署 食物代換表 / 每日飲食指南 | exchange portions | none | Values recalled, **not re-fetched**; confirm at hpa.gov.tw. |
| cw.com.tw/article/5127474, TVBS supertaste 345348 | 50嵐 large cup 668-783 kcal, 62.5 g sugar | none | Secondary press; low confidence. No official 50嵐/CoCo/清心 table was downloaded. |

## Findings worth knowing
- MOHW verified: 珍珠奶茶 700 mL 全糖 61.6 g / 半糖 44.8 g / 微糖 39.2 g sugar; 658 / 588 / 525 kcal.
- Sugar does not scale with the menu percentage: ~28 g stays in at 0% (pearl syrup, creamer). So `sugar_levels` are empirical total-sugar ratios (half = 0.727, light = 0.636), and base + toppings kcal are the MOHW residual after removing 4 kcal/g sugar.

## Assumptions (weakest first)
1. Oil ranges for pan_fry, braise, boil, steam are judgement. Stir-fry and deep-fry have one weak anchor each; Pu 2019 measures oil used, not retained, and has no per-method breakdown.
2. `adder_applies` flags are guesses made without dishes.yaml; all are `review_with_dishes: true`. True only for fried_egg, stirfried_greens, potstickers.
3. Densities for 11 of 14 solids are reasoned analogues (low confidence); only rice, soup and egg have a real number behind them. Bulk densities (greens, dumplings, popcorn chicken) are the widest.
4. chicken_leg/pork_chop volumes include bone: dishes.yaml kcal must be per whole piece as served.
5. Portion sizes (other than egg and the project's 雞排 200-300 g) are estimates of common servings; no Taiwan portion survey found.
6. Container dimensions are typical retail sizes from general knowledge, unmeasured. Triples are used for rim/depth (contract unspecified); bento_box adds length/width.
7. The base vs toppings split of the ~405 kcal residual is unsourced; only the sum is data-backed. The 700 mL basis is assumed to include ice.
8. `none` sugar level (0.455) is an extrapolation with no MOHW datapoint.
9. beef_noodle_soup density assumes whole bowl with broth; kcal_100g must match that basis.
