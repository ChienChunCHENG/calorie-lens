// Log data model and pure aggregation (day grouping, totals, 7-day medians, backup validation).
import { sumRange } from './mixture.ts';
import type { Range, Summable } from './mixture.ts';

export interface LogItem extends Range {
  id: string;
  /** Display name without the "or similar" suffix (zh if present, else en). */
  name: string;
  nameEn: string;
  /** True when top-1 probability < mix.sure -> show "或類似". */
  unsure: boolean;
  /** Box as fractions of the decoded image: [x1, y1, x2, y2]. */
  box: [number, number, number, number];
  /** 64 quantiles of the item's kcal distribution (see mixture.quantileArray), enough to recombine into totals. */
  q: number[];
  cands: { id: string; p: number }[];
}
export interface Meal<T = Blob | null> { id: string; ts: number; thumb: T; items: LogItem[] }

const pad = (n: number) => String(n).padStart(2, '0');
export function dayKey(ts: number): string {
  const d = new Date(ts);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function parseDayKey(k: string): Date {
  const [y, m, d] = k.split('-').map(Number);
  return new Date(y, m - 1, d, 12);
}

export function shiftDay(k: string, delta: number): string {
  const d = parseDayKey(k);
  d.setDate(d.getDate() + delta);
  return dayKey(d.getTime());
}

export function mealsOfDay<T>(meals: readonly Meal<T>[], key: string): Meal<T>[] {
  return meals.filter(m => dayKey(m.ts) === key).sort((a, b) => a.ts - b.ts);
}

export function mealTotal(m: Meal<unknown>): Range | null { return sumRange(m.items as Summable[], 'meal:' + m.id); }

export function dayTotal(meals: readonly Meal<unknown>[]): Range | null {
  const items = meals.flatMap(m => m.items);
  return sumRange(items as Summable[], 'day:' + meals.map(m => m.id).sort().join(','));
}

/** Median kcal per day for the 7 days ending at `endKey` (null = nothing logged). */
export function weekMedians<T>(meals: readonly Meal<T>[], endKey: string): { key: string; median: number | null }[] {
  const out: { key: string; median: number | null }[] = [];
  for (let i = 6; i >= 0; i--) {
    const key = shiftDay(endKey, -i);
    const t = dayTotal(mealsOfDay(meals, key));
    out.push({ key, median: t ? t.p50 : null });
  }
  return out;
}

export function removeItem<T>(m: Meal<T>, itemId: string): Meal<T> | null {
  const items = m.items.filter(i => i.id !== itemId);
  return items.length ? { ...m, items } : null; // a meal without items is deleted
}

// ---- backup ----
export const BACKUP_VERSION = 1;
export interface Backup { app: 'calorie-lens'; version: number; exported: string; meals: Meal<string | null>[] }

const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

export function validateItem(x: any): LogItem {
  if (!x || typeof x.id !== 'string' || typeof x.name !== 'string' || !isNum(x.p10) || !isNum(x.p50) || !isNum(x.p90) ||
    !Array.isArray(x.q) || x.q.length < 2 || !x.q.every(isNum)) throw new Error('bad item');
  const box = Array.isArray(x.box) && x.box.length === 4 && x.box.every(isNum) ? x.box : [0, 0, 1, 1];
  return {
    id: x.id, name: x.name, nameEn: typeof x.nameEn === 'string' ? x.nameEn : x.name, unsure: !!x.unsure,
    box: box as [number, number, number, number], p10: x.p10, p50: x.p50, p90: x.p90, q: x.q,
    cands: Array.isArray(x.cands) ? x.cands.filter((c: any) => c && typeof c.id === 'string' && isNum(c.p)) : [],
  };
}

/** Parse + validate a backup; throws a readable Error. Thumbnails stay as data-URL strings. */
export function parseBackup(text: string): Meal<string | null>[] {
  let j: any;
  try { j = JSON.parse(text); } catch { throw new Error('不是有效的 JSON / not valid JSON'); }
  if (!j || j.app !== 'calorie-lens' || !Array.isArray(j.meals)) throw new Error('不是 Calorie Lens 備份 / not a Calorie Lens backup');
  return j.meals.map((m: any) => {
    if (!m || typeof m.id !== 'string' || !isNum(m.ts) || !Array.isArray(m.items)) throw new Error('bad meal');
    const items = m.items.map(validateItem);
    return { id: m.id, ts: m.ts, thumb: typeof m.thumb === 'string' ? m.thumb : null, items };
  }).filter((m: Meal<string | null>) => m.items.length > 0);
}
