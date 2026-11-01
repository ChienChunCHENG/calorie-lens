// Seeded Monte Carlo: per-item kcal mixture, compact quantile storage, and sums of independent items.
import { sigmaOf } from '../../core/ts/rangeEngine.ts';

export const N_SAMPLES = 4000;
export const N_QUANTILES = 64;
export interface Range { p10: number; p50: number; p90: number }
export interface MixCand { p: number; kcal100: number; portion: [number, number, number] }

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashSeed(s: string): number { // FNV-1a
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

export function gauss(rng: () => number): number {
  const u = 1 - rng(), v = rng(); // u in (0,1]
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** log-normal sigma for a portion triple (min/max = 5th/95th percentile); degenerate/invalid -> moderate default. */
export function portionSigma(p: readonly number[]): number {
  try { return p[0] > 0 && p[2] >= p[0] ? sigmaOf(p[0], p[2]) : 0.3; } catch { return 0.3; }
}

/** Sorted kcal samples of the candidate mixture. Candidate probabilities must sum to ~1. */
export function mixtureSamples(cands: readonly MixCand[], n: number, seed: number): Float32Array {
  const rng = mulberry32(seed), out = new Float32Array(n);
  const cum: number[] = []; let s = 0;
  for (const c of cands) { s += c.p; cum.push(s); }
  const sig = cands.map(c => portionSigma(c.portion));
  for (let i = 0; i < n; i++) {
    const u = rng() * s;
    let k = 0; while (k < cands.length - 1 && u > cum[k]) k++;
    const c = cands[k], grams = c.portion[1] * Math.exp(sig[k] * gauss(rng));
    out[i] = grams * c.kcal100 / 100;
  }
  return out.sort();
}

export function percentile(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const x = Math.min(1, Math.max(0, q)) * (n - 1), i = Math.floor(x), f = x - i;
  return i + 1 < n ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[n - 1];
}

export function summarize(sorted: ArrayLike<number>): Range {
  return { p10: percentile(sorted, 0.1), p50: percentile(sorted, 0.5), p90: percentile(sorted, 0.9) };
}

/**
 * Compact storage of an item's distribution: the quantiles at probabilities (i+0.5)/m, i<m, of the sorted samples
 * (m = 64). Whole-kcal rounding. Resampling = inverse-CDF with linear interpolation (clamped at the ends), which is
 * enough to recombine items into meal/day totals without keeping 4000 samples per item.
 */
export function quantileArray(sorted: ArrayLike<number>, m = N_QUANTILES): number[] {
  const q: number[] = [];
  for (let i = 0; i < m; i++) q.push(Math.round(percentile(sorted, (i + 0.5) / m)));
  return q;
}

export function sampleQuantiles(q: readonly number[], u: number): number {
  const m = q.length;
  if (m === 1) return q[0];
  const x = u * m - 0.5;
  if (x <= 0) return q[0];
  if (x >= m - 1) return q[m - 1];
  const i = Math.floor(x), f = x - i;
  return q[i] * (1 - f) + q[i + 1] * f;
}

export interface Summable { p10: number; p50: number; p90: number; q: readonly number[] }

/** Sorted samples of the sum of independent items (each drawn from its stored quantile array). */
export function sumSamples(items: readonly Summable[], n: number, seed: number): Float32Array {
  const rng = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let t = 0;
    for (const it of items) t += sampleQuantiles(it.q, rng());
    out[i] = t;
  }
  return out.sort();
}

/** Range of a sum of items; zero items -> null; one item -> its own range (exact). */
export function sumRange(items: readonly Summable[], seedKey: string): Range | null {
  if (items.length === 0) return null;
  if (items.length === 1) { const { p10, p50, p90 } = items[0]; return { p10, p50, p90 }; }
  return summarize(sumSamples(items, N_SAMPLES, hashSeed(seedKey)));
}
