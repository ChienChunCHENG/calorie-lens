// Display helpers (English-only UI). Pure and unit-tested; no DOM.
export const round10 = (x: number): number => Math.round(x / 10) * 10;

/** Best estimate, e.g. "~520" (rounded to 10). */
export const fmtBest = (r: { p50: number }): string => `~${round10(r.p50)}`;

/** "380–690" (10th–90th percentile rounded to 10); collapses to "≈380" when both ends round to the same value. */
export function fmtRange(r: { p10: number; p90: number }): string {
  const a = round10(r.p10), b = round10(r.p90);
  return a === b ? `≈${a}` : `${a}–${b}`;
}

/** "~520 kcal" */
export const fmtKcal = (r: { p50: number }): string => `${fmtBest(r)} kcal`;

/** English display name of a vocab row: `display` if present, else `en`; never zh. */
export function itemName(row: { en: string; display?: string }): string { return row.display || row.en; }

/** Second name is shown only when its probability is >= 0.6 x the first's. Returns the index-1 candidate or null. */
export function maybeSecond<T extends { p: number }>(cands: readonly T[] | undefined): T | null {
  if (!cands || cands.length < 2 || !(cands[0].p > 0)) return null;
  return cands[1].p >= 0.6 * cands[0].p ? cands[1] : null;
}
