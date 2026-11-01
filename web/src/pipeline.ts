// Live lens engine: locate every frame, IoU-track boxes, classify new tracks and re-classify each every ~2 s, vote labels.
// Model runners are injected, so it is testable without ONNX.
import { protoBoost } from './protos.ts';
import type { Protos } from './protos.ts';
import { clipTensor, cropRect, cropRGBA, letterboxTensor } from './imageops.ts';
import type { Rgba } from './imageops.ts';
import { selectBoxes } from './yolo.ts';
import { classify, l2normalize, similarities } from './classify.ts';
import type { Gate, Mix, VocabRow } from './classify.ts';
import { N_SAMPLES, hashSeed, mixtureSamples, quantileArray, summarize } from './mixture.ts';
import { itemName } from './format.ts';
import { area, interArea, iou } from './yolo.ts';
import type { Box } from './yolo.ts';

export interface LocatorCfg {
  imgsz: number; input: string; output: string; classes: { name: string; food: boolean }[];
  conf: number; iou: number; contain: number; max_det: number; crop_pad: number;
  /** Stricter gate for the whole-image fallback (an empty frame must not log a drink). */
  fallback_whole?: { when: 'no_boxes'; p_neg_max: number; p_food_min: number };
}
export interface ClipCfg {
  input: string; output: string; dtype?: string; size: number; resize: 'shortest' | 'squash'; crop: 'center' | null;
  interpolation?: string; mean: number[]; std: number[]; dim: number; logit_scale: number;
  gate: Gate; mix: Mix; template?: string;
}
export interface Assets { locator: LocatorCfg; clip: ClipCfg; vocab: VocabRow[]; emb: Float32Array; protos?: Protos }
export interface Runners {
  locate(data: Float32Array, size: number): Promise<{ data: ArrayLike<number>; dims: readonly number[] }>;
  embed(data: Float32Array, size: number): Promise<Float32Array>;
}

export interface LensCand { id: string; name: string; p: number }
export interface LensTrack {
  id: number; name: string; nameEn: string; box: [number, number, number, number];
  p10: number; p50: number; p90: number; q: number[]; cands: LensCand[];
}
export interface LensOpts { matchIou: number; maxMissed: number; showMissed: number; minHits: number; reclassifyMs: number; maxClassify: number; hysteresis: number; decay: number }
export const LENS_DEFAULTS: LensOpts = { matchIou: 0.25, maxMissed: 4, showMissed: 2, minHits: 2, reclassifyMs: 2000, maxClassify: 2, hysteresis: 1.3, decay: 0.9 };

interface Tr {
  id: number; box: Box; hits: number; missed: number; lastCls: number;
  votes: Map<string, number>; food: number; neg: number; label: string | null;
  /** Top-1 probability of the latest classification (a confident first sighting may show at once). */
  conf: number;
  /** Region is the whole frame (no box found): classified with the stricter fallback gate. */
  whole?: boolean;
  out: Omit<LensTrack, 'id' | 'box'> | null;
}

export class LensEngine {
  tracks: Tr[] = [];
  private nextId = 1;
  stats = { nBoxes: 0, classified: 0, prepMs: 0, locMs: 0, clsMs: 0 };
  private a: Assets; private run: Runners; private o: LensOpts;
  constructor(a: Assets, run: Runners, o: LensOpts = LENS_DEFAULTS) {
    this.a = a; this.run = run; this.o = o;
    const rows = a.vocab.length;
    if (a.emb.length !== rows * a.clip.dim) throw new Error(`vocab_emb has ${a.emb.length} values, expected ${rows}x${a.clip.dim}`);
    this.byId = new Map(a.vocab.map(r => [r.id, r]));
  }
  private byId: Map<string, VocabRow>;
  reset(): void { this.tracks = []; }

  /** Analyse one frame; returns the tracks worth drawing (confirmed, label won the food-vs-not-food vote). */
  async step(img: Rgba, now: number): Promise<LensTrack[]> {
    const { locator: L, clip: C } = this.a, o = this.o;
    const p0 = performance.now();
    const { data, lb } = letterboxTensor(img, L.imgsz);
    const p1 = performance.now();
    const out = await this.run.locate(data, L.imgsz);
    this.stats.prepMs = p1 - p0; this.stats.locMs = performance.now() - p1; this.stats.clsMs = 0;
    const all = selectBoxes(out.data, out.dims, lb, img.w, img.h,
      { nc: L.classes.length, conf: L.conf, iou: L.iou, contain: L.contain, maxDet: L.max_det });
    const dets: (Box & { whole?: boolean })[] = all.filter(d => L.classes[d.cls]?.food !== false);
    this.stats.nBoxes = dets.length;
    // Nothing located at all (not even an empty cup): treat the whole frame as one region, judged by the stricter
    // fallback gate, so a close-up of a single food (a bowl filling the screen) still gets a label.
    if (!all.length && L.fallback_whole) dets.push({ x1: 0, y1: 0, x2: img.w, y2: img.h, whole: true });

    // greedy IoU association against each track's latest box (class-agnostic)
    const pairs: { t: number; d: number; v: number }[] = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => { const v = iou(t.box, d); if (v >= o.matchIou) pairs.push({ t: ti, d: di, v }); }));
    pairs.sort((x, y) => y.v - x.v);
    const usedT = new Set<number>(), usedD = new Set<number>();
    for (const p of pairs) {
      if (usedT.has(p.t) || usedD.has(p.d)) continue;
      usedT.add(p.t); usedD.add(p.d);
      const t = this.tracks[p.t], d = dets[p.d];
      t.box = { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 }; t.hits++; t.missed = 0; t.whole = !!d.whole;
    }
    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.missed++; });
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      this.tracks.push({ id: this.nextId++, box: { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 }, hits: 1, missed: 0, lastCls: -1,
        votes: new Map(), food: 0, neg: 0, label: null, conf: 0, out: null, whole: !!d.whole });
    });
    this.tracks = this.tracks.filter(t => t.missed <= o.maxMissed);

    // classify: new tracks first, then the stalest; a few per frame so the loop stays responsive
    const due = this.tracks.filter(t => t.missed === 0 && (t.lastCls < 0 || now - t.lastCls >= o.reclassifyMs))
      .sort((x, y) => (x.lastCls < 0 ? -1e12 : x.lastCls) - (y.lastCls < 0 ? -1e12 : y.lastCls)).slice(0, o.maxClassify);
    for (const t of due) {
      const c0 = performance.now();
      const rect = cropRect(t.box, img.w, img.h, t.whole ? 0 : L.crop_pad);
      const emb = l2normalize(await this.run.embed(clipTensor(cropRGBA(img, rect), C), C.size));
      const sims = similarities(emb, this.a.emb, this.a.vocab.length, C.dim);
      if (this.a.protos) protoBoost(sims, emb, this.a.protos);
      const gate = t.whole && L.fallback_whole ? { p_neg_max: L.fallback_whole.p_neg_max, p_food_min: L.fallback_whole.p_food_min } : C.gate;
      const cl = classify(sims, this.a.vocab, C.logit_scale, gate, C.mix);
      t.lastCls = now; this.stats.classified++; this.stats.clsMs += performance.now() - c0;
      this.vote(t, cl);
    }

    const shown = this.tracks.filter(t => t.label && t.out && (t.hits >= o.minHits || t.conf >= this.a.clip.mix.sure) && t.missed <= o.showMissed && t.food > t.neg);
    // a duplicate box (>= 80% inside a better track with the same label) would double-count calories: drop it
    const vis = shown.filter(t => !shown.some(u => u !== t && u.label === t.label && (u.hits > t.hits || (u.hits === t.hits && u.id < t.id))
      && interArea(t.box, u.box) >= 0.8 * area(t.box)));
    return vis.map(t => ({ id: t.id, box: [t.box.x1 / img.w, t.box.y1 / img.h, t.box.x2 / img.w, t.box.y2 / img.h] as [number, number, number, number], ...t.out! }));
  }

  private vote(t: Tr, cl: ReturnType<typeof classify>): void {
    const o = this.o, C = this.a.clip;
    for (const [k, v] of t.votes) t.votes.set(k, v * o.decay);
    t.food *= o.decay; t.neg *= o.decay;
    if (cl.skip) { t.neg += 1; t.conf = 0; return; }
    t.food += 1; t.conf = cl.top1;
    for (const c of cl.cands) { const id = this.a.vocab[c.idx].id; t.votes.set(id, (t.votes.get(id) ?? 0) + c.p); }
    let top = '', tv = -1;
    for (const [k, v] of t.votes) if (v > tv) { top = k; tv = v; }
    if (!t.label || (top !== t.label && tv > o.hysteresis * (t.votes.get(t.label) ?? 0))) t.label = top;

    // kcal mixture over the voted candidates (same truncation as classify: cumulative mix.cum, at most mix.k)
    const ranked = [...t.votes].sort((x, y) => y[1] - x[1]);
    const tot = ranked.reduce((s, r) => s + r[1], 0) || 1;
    const kept: [string, number][] = []; let cum = 0;
    for (const [id, v] of ranked) { kept.push([id, v / tot]); cum += v / tot; if (kept.length >= C.mix.k || cum >= C.mix.cum) break; }
    const ks = kept.reduce((s, r) => s + r[1], 0);
    const rows = kept.map(([id, p]) => ({ p: p / ks, row: this.byId.get(id)! })).filter(r => r.row);
    const sorted = mixtureSamples(rows.map(r => ({ p: r.p, kcal100: r.row.kcal100, portion: r.row.portion_g })), N_SAMPLES,
      hashSeed(`${t.id}|${rows.map(r => r.row.id).join('|')}`));
    const r = summarize(sorted), lab = this.byId.get(t.label)!;
    const others = rows.filter(x => x.row.id !== t.label);
    const lp = rows.find(x => x.row.id === t.label)?.p ?? 0;
    t.out = {
      name: itemName(lab), nameEn: lab.en, p10: Math.round(r.p10), p50: Math.round(r.p50), p90: Math.round(r.p90), q: quantileArray(sorted),
      cands: [{ id: lab.id, name: itemName(lab), p: lp }, ...others.map(x => ({ id: x.row.id, name: itemName(x.row), p: x.p }))],
    };
  }
}
