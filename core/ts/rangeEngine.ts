// Calorie Lens range engine, TS port of core/py/range_engine.py. See spec/range_engine.md.
import type { Triple, DishEntry, DrinkEntry, ItemEstimate, SugarLevel, MealRange } from './types.ts';

export type { Triple, DishEntry, DrinkEntry, ItemEstimate, SugarLevel, MealRange };

const Z90 = 1.645;
const SPAN90 = 3.29;
const FLOOR_LO = 0.85;
const FLOOR_HI = 1.15;
const LOW_CONF_SIGMA = 0.35;
const SUGAR_VARIANCE = 0.25;
const KCAL_PER_G_SUGAR = 4;
const KCAL_PER_G_OIL = 9;

function num(x: unknown, what: string): number {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(what + ' must be a finite number');
  return x;
}

function triple(t: Triple | null | undefined, what: string, positive = true): [number, number, number] {
  if (t === null || t === undefined || typeof t !== 'object') throw new Error(what + ' missing');
  const mn = num(t.min, what + '.min');
  const ty = num(t.typ, what + '.typ');
  const mx = num(t.max, what + '.max');
  if (!(mn <= ty && ty <= mx)) throw new Error(what + ' must satisfy min<=typ<=max');
  if (positive && mn <= 0) throw new Error(what + '.min must be > 0');
  if (!positive && mn < 0) throw new Error(what + '.min must be >= 0');
  return [mn, ty, mx];
}

export function sigmaOf(mn: number, mx: number): number {
  if (mn <= 0) throw new Error('min must be > 0');
  if (mn === mx) return 0;
  return Math.log(mx / mn) / SPAN90;
}

function applyFloor(lo: number, med: number, hi: number): [number, number] {
  return [Math.min(lo, FLOOR_LO * med), Math.max(hi, FLOOR_HI * med)];
}

export function estimateSolid(dish: DishEntry, volumeMl: Triple | null): ItemEstimate {
  const flags: string[] = [];
  const [eMin, eTyp, eMax] = triple(dish.kcal_100g, 'kcal_100g');
  const sE = sigmaOf(eMin, eMax);
  let massMed: number;
  let sig2: number;
  if (volumeMl === null || volumeMl === undefined) {
    const [pMin, pTyp, pMax] = triple(dish.default_portion_g, 'default_portion_g');
    massMed = pTyp;
    sig2 = sigmaOf(pMin, pMax) ** 2 + sE ** 2;
    flags.push('portion_default');
  } else {
    const [vMin, vTyp, vMax] = triple(volumeMl, 'volumeMl');
    const [dMin, dTyp, dMax] = triple(dish.density_g_ml, 'density_g_ml');
    massMed = vTyp * dTyp;
    sig2 = sigmaOf(vMin, vMax) ** 2 + sigmaOf(dMin, dMax) ** 2 + sE ** 2;
  }
  let med = (massMed * eTyp) / 100;
  const sigma = Math.sqrt(sig2);
  let lo = med * Math.exp(-Z90 * sigma);
  let hi = med * Math.exp(Z90 * sigma);
  const oil = dish.oil_g_per_100g;
  if (oil !== undefined && oil !== null) {
    const oMin = num(oil.min, 'oil.min');
    const oMax = num(oil.max, 'oil.max');
    if (oMin < 0 || oMax < oMin) throw new Error('oil must satisfy 0<=min<=max');
    lo += (oMin * massMed * KCAL_PER_G_OIL) / 100;
    hi += (oMax * massMed * KCAL_PER_G_OIL) / 100;
    med += (((oMin + oMax) / 2) * massMed * KCAL_PER_G_OIL) / 100;
  }
  [lo, hi] = applyFloor(lo, med, hi);
  if (sigma > LOW_CONF_SIGMA) flags.push('low_confidence');
  return { id: dish.id, lo, med, hi, flags };
}

export function estimateDrink(
  drink: DrinkEntry,
  sizeMl: number,
  sugarLevel: SugarLevel,
  fullSugarGPer700: number,
  sugarFraction: Record<string, number>,
): ItemEstimate {
  const size = num(sizeMl, 'sizeMl');
  if (size <= 0) throw new Error('sizeMl must be > 0');
  const full = num(fullSugarGPer700, 'fullSugarGPer700');
  if (full < 0) throw new Error('fullSugarGPer700 must be >= 0');
  if (!sugarFraction || !Object.prototype.hasOwnProperty.call(sugarFraction, sugarLevel)) {
    throw new Error('unknown sugar level');
  }
  const frac = num(sugarFraction[sugarLevel], 'sugar fraction');
  if (frac < 0) throw new Error('sugar fraction must be >= 0');
  const b = triple(drink.base_kcal_per_700ml, 'base_kcal_per_700ml', false);
  const t = triple(drink.toppings_kcal, 'toppings_kcal', false);
  const sKcal = ((full * frac * size) / 700) * KCAL_PER_G_SUGAR;
  const k = size / 700;
  const lo0 = sKcal * (1 - SUGAR_VARIANCE) + b[0] * k + t[0];
  const med = sKcal + b[1] * k + t[1];
  const hi0 = sKcal * (1 + SUGAR_VARIANCE) + b[2] * k + t[2];
  const [lo, hi] = applyFloor(lo0, med, hi0);
  return { id: drink.id, lo, med, hi, flags: ['drink_sugar_assumed'] };
}

export function combineMeal(items: ItemEstimate[]): MealRange {
  let med = 0;
  let dl = 0;
  let du = 0;
  for (const i of items) {
    med += i.med;
    dl += (i.med - i.lo) ** 2;
    du += (i.hi - i.med) ** 2;
  }
  return { lo: med - Math.sqrt(dl), med, hi: med + Math.sqrt(du) };
}

export function round5(x: number): number {
  return Math.floor(x / 5 + 0.5) * 5;
}

export function formatRange(lo: number, hi: number): string {
  return round5(lo) + '–' + round5(hi);
}
