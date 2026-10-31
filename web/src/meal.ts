// Pure meal logic: portion -> volume triple, per-item estimate, meal total. No Vite/DOM imports, so node can test it.
import { estimateSolid, estimateDrink, combineMeal } from '../../core/ts/rangeEngine.ts';
import type { Triple, DishEntry, DrinkEntry, ItemEstimate, SugarLevel, MealRange } from '../../core/ts/rangeEngine.ts';

export type { Triple, ItemEstimate, SugarLevel, MealRange };
export type Ice = 'normal' | 'less' | 'none';

export interface DishFull extends DishEntry {
  zh: string; en: string; container_prior: string[];
}
export interface ContainerInfo { shape: string; zh: string; capacity_ml: Triple }
export interface MealData {
  dishes: Record<string, DishFull>;
  dishOrder: string[];
  containers: Record<string, ContainerInfo>;
  drinks: Record<string, DrinkEntry>;
  fullSugarGPer700: number; // drinks.full_sugar_g_per_700ml.typ (the engine adds its own +-25% band)
  sugarLevels: Record<string, number>;
}

/** Shape of data/nutrition.json, only the parts we read. */
export interface RawNutrition {
  dishes: Array<Record<string, any>>;
  containers: Record<string, ContainerInfo>;
  drinks: { full_sugar_g_per_700ml: Triple; sugar_levels: Record<string, number>; drinks: Record<string, any> };
}

export function buildMealData(raw: RawNutrition): MealData {
  const dishes: Record<string, DishFull> = {};
  const dishOrder: string[] = [];
  for (const d of raw.dishes) {
    dishes[d.id] = { ...d, oil_g_per_100g: d.oil_g_per_100g ?? undefined, density_g_ml: d.density_g_ml ?? undefined,
      container_prior: d.container_prior ?? [] } as DishFull;
    dishOrder.push(d.id);
  }
  const drinks: Record<string, DrinkEntry> = {};
  for (const [id, v] of Object.entries(raw.drinks.drinks)) drinks[id] = { id, kind: 'drink', ...v } as DrinkEntry;
  return { dishes, dishOrder, containers: raw.containers, drinks,
    fullSugarGPer700: raw.drinks.full_sugar_g_per_700ml.typ, sugarLevels: raw.drinks.sugar_levels };
}

export interface Item { dishId: string; containerId: string; sugar: SugarLevel; ice: Ice }
export const UNKNOWN_CONTAINER = 'unknown';
export const DEFAULT_SUGAR: SugarLevel = 'half';
export const FILL = { min: 0.6, typ: 0.8, max: 1.0 };
export const CUP_ML: Record<string, number> = { cup_500: 500, cup_700: 700 };

/**
 * Portion rule (v1, web has no depth):
 *  - bowls and cups: volume = container capacity_ml triple x fill fraction (min 0.6, typ 0.8, max 1.0), element-wise.
 *  - plate_26cm, bento_box and "unknown": null. A plate has no brim capacity, and a bento box holds several dishes in
 *    compartments, so a whole-box capacity would overstate any single dish. null makes the engine use default_portion_g
 *    and flag 'portion_default'.
 */
export function volumeFor(containerId: string, containers: Record<string, ContainerInfo>): Triple | null {
  const c = containers[containerId];
  if (!c || (c.shape !== 'bowl' && c.shape !== 'cup')) return null;
  const cap = c.capacity_ml;
  return { min: cap.min * FILL.min, typ: cap.typ * FILL.typ, max: cap.max * FILL.max };
}

/** Container chip choices: cups for drinks, everything else for solids. "unknown" is added by the UI. */
/** Containers offered for a dish: only its plausible ones (container_prior), so rice can't land in a 900 mL noodle bowl. */
export function containerChoices(data: MealData, dishId: string, current?: string): string[] {
  const ids = (data.dishes[dishId]?.container_prior ?? []).filter(id => id in data.containers);
  if (current && current in data.containers && !ids.includes(current)) ids.push(current);
  return ids;
}

/** New item for a dish, container preselected from container_prior[0] (or a bowl when the box was a COCO bowl). */
export function defaultItem(data: MealData, dishId: string, prefer?: 'bowl'): Item {
  const d = data.dishes[dishId];
  let containerId = d?.container_prior[0] ?? UNKNOWN_CONTAINER;
  if (prefer === 'bowl' && d?.kind !== 'drink' && data.containers[containerId]?.shape !== 'bowl')
    containerId = d.container_prior.find(id => data.containers[id]?.shape === 'bowl') ?? 'bowl_small';
  return { dishId, containerId, sugar: DEFAULT_SUGAR, ice: 'normal' };
}

export function estimateItem(data: MealData, item: Item): ItemEstimate {
  const dish = data.dishes[item.dishId];
  if (!dish) throw new Error('unknown dish ' + item.dishId);
  if (dish.kind === 'drink') {
    const size = CUP_ML[item.containerId] ?? CUP_ML[dish.container_prior[0]] ?? 700;
    return estimateDrink(data.drinks[dish.id], size, item.sugar, data.fullSugarGPer700, data.sugarLevels);
  }
  return estimateSolid(dish, volumeFor(item.containerId, data.containers));
}

export function totalOf(estimates: ItemEstimate[]): MealRange { return combineMeal(estimates); }

/** Item estimate that never throws (the UI should still show the name). */
export function safeEstimate(data: MealData, item: Item): ItemEstimate | null {
  try { return estimateItem(data, item); } catch { return null; }
}

/** If a detector label name is a dish id (future custom model), return it. */
export function autoDishId(label: string | undefined, data: MealData): string | null {
  return label && data.dishes[label] ? label : null;
}

/** Dishes served as one bowl whose toppings are part of the dish: boxes inside them are not extra items. */
export const WHOLE_DISHES = new Set(['beef_noodle_soup', 'lu_rou_fan']);
