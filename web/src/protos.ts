// Image-prototype boost (ml/snap/protos/SPEC.md): real example photos per food concept lift the text scores of that
// concept's vocab rows when a crop is close to them. Below the threshold c nothing changes, so unusual foods keep the
// pure text behaviour; 20 non-food concepts (cutlery, table, hand...) lift the not-food rows so the gate still works.
export interface ProtosFile {
  dim: number; n: number; a: number; c: number; m: number;
  concepts: { name: string; vocab_ids: string[]; neg: boolean }[]; offsets: number[];
}
export interface Protos { emb: Float32Array; dim: number; a: number; c: number; m: number; concepts: { start: number; end: number; rows: number[] }[] }

/** Resolve concept vocab ids to row numbers of the shipped vocab once at load (missing ids are skipped). */
export function prepareProtos(f: ProtosFile, emb: Float32Array, vocabIds: readonly string[]): Protos {
  if (emb.length !== f.n * f.dim) throw new Error(`protos_emb has ${emb.length} values, expected ${f.n}x${f.dim}`);
  const row = new Map(vocabIds.map((id, i) => [id, i]));
  const concepts = f.concepts.map((k, i) => ({
    start: f.offsets[i], end: f.offsets[i + 1],
    rows: k.vocab_ids.map(id => row.get(id)).filter((r): r is number => r !== undefined),
  })).filter(k => k.end > k.start && k.rows.length);
  return { emb, dim: f.dim, a: f.a, c: f.c, m: f.m, concepts };
}

/** Adds a * max(mean of the m best prototype cosines - c, 0) to every row of each concept, in place. */
export function protoBoost(sims: Float32Array, e: Float32Array, P: Protos): Float32Array {
  const { emb, dim, a, c, m } = P;
  for (const k of P.concepts) {
    const best: number[] = [];
    for (let j = k.start; j < k.end; j++) {
      let s = 0; const o = j * dim;
      for (let d = 0; d < dim; d++) s += emb[o + d] * e[d];
      best.push(s);
    }
    best.sort((x, y) => y - x);
    const top = best.slice(0, Math.min(m, best.length));
    const boost = a * Math.max(top.reduce((x, y) => x + y, 0) / top.length - c, 0);
    if (boost > 0) for (const r of k.rows) sims[r] += boost;
  }
  return sims;
}
