// Pure zero-shot classification maths: L2 norm, similarity, softmax, gate, top-K candidate selection.
export interface VocabRow {
  id: string; en: string; display?: string; zh?: string; kcal100: number; portion_g: [number, number, number]; cat?: string; neg?: true;
  /** Centred log commonness prior (ml/snap/prior); 0 or absent = neutral. */
  lp?: number;
}
export interface Gate { p_neg_max: number; p_food_min: number }
/** lambda weights the commonness prior when ranking food candidates (0 = pure zero-shot). */
export interface Mix { k: number; cum: number; sure: number; lambda?: number }
export interface Cand { idx: number; p: number }
export type Skip = 'negative' | 'low_food';
export interface Classified {
  skip: Skip | null; pNeg: number; pTopFood: number;
  /** Top-K candidates renormalised to sum 1 (empty when skipped). */
  cands: Cand[];
  /** Top-1 probability after renormalising over food rows, before truncation (compared with mix.sure). */
  top1: number;
}

export function l2normalize(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1, out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

/** Cosine similarities of a (normalised) embedding with every row of a rows x dim matrix. */
export function similarities(emb: Float32Array, mat: Float32Array, rows: number, dim: number): Float32Array {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0; const o = r * dim;
    for (let d = 0; d < dim; d++) s += emb[d] * mat[o + d];
    out[r] = s;
  }
  return out;
}

export function softmax(logits: Float32Array | number[], scale = 1): Float32Array {
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) m = Math.max(m, logits[i] * scale);
  const out = new Float32Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] * scale - m); s += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}

export function classify(sims: Float32Array, vocab: readonly VocabRow[], logitScale: number, gate: Gate, mix: Mix): Classified {
  const p = softmax(sims, logitScale);
  let pNeg = 0, pFoodSum = 0, pTopFood = 0;
  const food: Cand[] = [];
  const lambda = mix.lambda ?? 0;
  for (let i = 0; i < vocab.length; i++) {
    if (vocab[i].neg) { pNeg += p[i]; continue; }
    pTopFood = Math.max(pTopFood, p[i]);
    // Ranking adds lambda * lp to the logit, i.e. multiplies the probability by exp(lambda * lp).
    const w = lambda ? p[i] * Math.exp(lambda * (vocab[i].lp ?? 0)) : p[i];
    pFoodSum += w;
    food.push({ idx: i, p: w });
  }
  const none = (skip: Skip): Classified => ({ skip, pNeg, pTopFood, cands: [], top1: 0 });
  // The gate uses the prior-free probabilities so the prior never changes what counts as food.
  if (pNeg > gate.p_neg_max) return none('negative');
  if (pTopFood < gate.p_food_min || pFoodSum <= 0) return none('low_food');
  food.sort((a, b) => b.p - a.p);
  const top1 = food[0].p / pFoodSum;
  const kept: Cand[] = []; let cum = 0;
  for (const c of food) {
    kept.push({ idx: c.idx, p: c.p / pFoodSum });
    cum += c.p / pFoodSum;
    if (kept.length >= mix.k || cum >= mix.cum) break;
  }
  const t = kept.reduce((s, c) => s + c.p, 0);
  for (const c of kept) c.p /= t;
  return { skip: null, pNeg, pTopFood, cands: kept, top1 };
}
