export const round10 = (x: number): number => Math.round(x / 10) * 10;

/** "310–520" (10th–90th percentile rounded to 10); collapses to "≈310" when both ends round the same. */
export function fmtRange(r: { p10: number; p90: number }): string {
  const a = round10(r.p10), b = round10(r.p90);
  return a === b ? `≈${a}` : `${a}–${b}`;
}

export const fmtKcal = (r: { p10: number; p90: number }): string => `${fmtRange(r)} kcal`;

export function itemName(row: { en: string; zh?: string }): string { return row.zh || row.en; }

/** Displayed name: top-1 name plus 或類似 when the top-1 probability is below `mix.sure`. */
export function displayName(name: string, unsure: boolean): string { return unsure ? `${name} 或類似` : name; }

export function fmtTime(ts: number): string {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}
