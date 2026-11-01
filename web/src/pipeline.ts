// The analysis pipeline (contract steps 2-6) with injected model runners, so it is testable without ONNX.
import { clipTensor, cropRect, cropRGBA, letterboxTensor } from './imageops.ts';
import type { Rgba } from './imageops.ts';
import { selectBoxes } from './yolo.ts';
import { classify, l2normalize, similarities } from './classify.ts';
import type { Gate, Mix, VocabRow } from './classify.ts';
import { N_SAMPLES, hashSeed, mixtureSamples, quantileArray, summarize, sumRange } from './mixture.ts';
import type { Range } from './mixture.ts';
import { itemName } from './format.ts';
import type { LogItem } from './log.ts';

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
export interface Assets { locator: LocatorCfg; clip: ClipCfg; vocab: VocabRow[]; emb: Float32Array }
export interface Runners {
  locate(data: Float32Array, size: number): Promise<{ data: ArrayLike<number>; dims: readonly number[] }>;
  embed(data: Float32Array, size: number): Promise<Float32Array>;
}
export interface Analysis {
  items: LogItem[]; total: Range | null; whole: boolean; nBoxes: number; skipped: { negative: number; low_food: number };
}

export async function analyze(img: Rgba, a: Assets, run: Runners, stage?: (s: 'locate' | 'classify') => void): Promise<Analysis> {
  const { locator: L, clip: C, vocab } = a;
  const rows = vocab.length, dim = C.dim;
  if (a.emb.length !== rows * dim) throw new Error(`vocab_emb has ${a.emb.length} values, expected ${rows}x${dim}`);

  stage?.('locate');
  const { data, lb } = letterboxTensor(img, L.imgsz);
  const out = await run.locate(data, L.imgsz);
  const dets = selectBoxes(out.data, out.dims, lb, img.w, img.h,
    { nc: L.classes.length, conf: L.conf, iou: L.iou, contain: L.contain, maxDet: L.max_det });

  // step 3: nothing located -> the whole image is one candidate region. Non-food boxes are discarded outright.
  const whole = dets.length === 0;
  const regions = whole
    ? [{ x1: 0, y1: 0, x2: img.w, y2: img.h }]
    : dets.filter(d => L.classes[d.cls]?.food !== false);

  const gate: Gate = whole && L.fallback_whole
    ? { p_neg_max: L.fallback_whole.p_neg_max, p_food_min: L.fallback_whole.p_food_min } : C.gate;

  stage?.('classify');
  const items: LogItem[] = [];
  const skipped = { negative: 0, low_food: 0 };
  for (const box of regions) {
    const rect = cropRect(box, img.w, img.h, whole ? 0 : L.crop_pad);
    const emb = l2normalize(await run.embed(clipTensor(cropRGBA(img, rect), C), C.size));
    const cl = classify(similarities(emb, a.emb, rows, dim), vocab, C.logit_scale, gate, C.mix);
    if (cl.skip) { skipped[cl.skip]++; continue; }
    const cands = cl.cands.map(c => ({ p: c.p, row: vocab[c.idx] }));
    const key = cands.map(c => c.row.id).join('|') + `@${Math.round(box.x1)},${Math.round(box.y1)}`;
    const sorted = mixtureSamples(cands.map(c => ({ p: c.p, kcal100: c.row.kcal100, portion: c.row.portion_g })), N_SAMPLES, hashSeed(key));
    const r = summarize(sorted), top = cands[0].row;
    items.push({
      id: `i${items.length}`, name: itemName(top), nameEn: top.en, unsure: cl.top1 < C.mix.sure,
      box: [box.x1 / img.w, box.y1 / img.h, box.x2 / img.w, box.y2 / img.h],
      p10: Math.round(r.p10), p50: Math.round(r.p50), p90: Math.round(r.p90), q: quantileArray(sorted),
      cands: cands.map(c => ({ id: c.row.id, p: Math.round(c.p * 1000) / 1000 })),
    });
  }
  return { items, total: sumRange(items, 'new'), whole, nBoxes: dets.length, skipped };
}
