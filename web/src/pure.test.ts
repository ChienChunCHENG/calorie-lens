// Run: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { f16ArrayToF32, f32ArrayToF16, f16ToF32 } from './f16.ts';
import { decodeYolo, iou, letterboxParams, nms, selectBoxes, suppressContained } from './yolo.ts';
import type { Detection } from './yolo.ts';
import { clipTensor, cropRect, cropRGBA, fitSize, letterboxTensor, resizeRGBA } from './imageops.ts';
import { classify, l2normalize, softmax, similarities } from './classify.ts';
import type { VocabRow } from './classify.ts';
import { hashSeed, mixtureSamples, mulberry32, percentile, quantileArray, sampleQuantiles, sumRange, summarize } from './mixture.ts';
import { displayName, fmtRange, round10 } from './format.ts';
import { dayKey, dayTotal, mealTotal, parseBackup, removeItem, shiftDay, weekMedians, mealsOfDay } from './log.ts';
import type { LogItem, Meal } from './log.ts';
import { analyze } from './pipeline.ts';
import type { Assets } from './pipeline.ts';

const rgba = (w: number, h: number, f: (x: number, y: number) => [number, number, number]) => {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const [r, g, b] = f(x, y); d.set([r, g, b, 255], (y * w + x) * 4); }
  return { data: d, w, h };
};

test('f16 round trip', () => {
  const v = new Float32Array([0, 1, -1, 0.5, 0.1234, -3.75, 1000, 6e-5]);
  const back = f16ArrayToF32(f32ArrayToF16(v));
  v.forEach((x, i) => assert.ok(Math.abs(back[i] - x) <= Math.abs(x) * 1e-3 + 1e-7, `${x} vs ${back[i]}`));
  assert.equal(f16ToF32(0x3c00), 1);
  assert.equal(f16ToF32(0xc000), -2);
});

test('letterbox params and tensor', () => {
  const lb = letterboxParams(1280, 960, 640);
  assert.deepEqual([lb.scale, lb.newW, lb.newH, lb.padX, lb.padY], [0.5, 640, 480, 0, 80]);
  const img = rgba(40, 20, () => [255, 0, 0]);
  const { data, lb: l2 } = letterboxTensor(img, 32);
  const n = 32 * 32;
  assert.equal(l2.newH, 16); assert.equal(l2.padY, 8);
  assert.ok(Math.abs(data[0] - 114 / 255) < 1e-6);               // padding gray
  const c = (16 * 32) + 16;                                       // centre pixel is red
  assert.ok(Math.abs(data[c] - 1) < 1e-6 && data[n + c] === 0 && data[2 * n + c] === 0);
});

test('decodeYolo maps letterbox px back to image and reads only nc class rows', () => {
  const N = 64, C = 4 + 2 + 32;                                   // 2 classes + 32 mask coefficients
  const out = new Float32Array(C * N);
  const set = (i: number, cx: number, cy: number, w: number, h: number, s0: number, s1: number) => {
    out[i] = cx; out[N + i] = cy; out[2 * N + i] = w; out[3 * N + i] = h; out[4 * N + i] = s0; out[5 * N + i] = s1;
  };
  set(0, 320, 320, 200, 100, 0.2, 0.9);
  set(1, 100, 100, 50, 50, 0.1, 0.1);
  set(2, 10, 10, 50, 50, 0, 0); out[6 * N + 2] = 5;              // mask-coef row must be ignored
  const lb = letterboxParams(1280, 1280, 640);
  const d = decodeYolo(out, [1, C, N], lb, 1280, 1280, { nc: 2, conf: 0.25 });
  assert.equal(d.length, 1);
  assert.equal(d[0].cls, 1);
  assert.deepEqual([d[0].x1, d[0].y1, d[0].x2, d[0].y2], [440, 540, 840, 740]);
});

test('nms is class-agnostic; contained boxes are dropped', () => {
  const D = (x1: number, y1: number, x2: number, y2: number, score: number, cls = 0): Detection => ({ x1, y1, x2, y2, score, cls });
  const dets = [D(0, 0, 100, 100, 0.9, 0), D(5, 5, 100, 100, 0.8, 1), D(200, 200, 300, 300, 0.7), D(220, 220, 260, 260, 0.6)];
  const n = nms(dets, 0.5);
  assert.equal(n.length, 3);                                     // 2nd merged into 1st despite other class
  const c = suppressContained(n, 0.9);
  assert.deepEqual(c.map(x => x.score), [0.9, 0.7]);
  assert.ok(Math.abs(iou(D(0, 0, 10, 10, 1), D(5, 0, 15, 10, 1)) - 1 / 3) < 1e-9);
  assert.equal(suppressContained(n, 1.01).length, 3);
});

test('selectBoxes honours max_det', () => {
  const N = 5, out = new Float32Array(5 * N);
  for (let i = 0; i < N; i++) { out[i] = 50 + i * 100; out[N + i] = 50; out[2 * N + i] = 40; out[3 * N + i] = 40; out[4 * N + i] = 0.9 - i * 0.1; }
  const d = selectBoxes(out, [1, 5, N], letterboxParams(640, 640, 640), 640, 640, { nc: 1, conf: 0.25, iou: 0.5, contain: 0.9, maxDet: 3 });
  assert.equal(d.length, 3);
});

test('resize: constant image stays constant; downscale averages', () => {
  const flat = rgba(50, 30, () => [10, 200, 90]);
  for (const mode of ['bilinear', 'bicubic'] as const) {
    const r = resizeRGBA(flat, 20, 12, mode);
    for (let i = 0; i < r.w * r.h; i++) assert.deepEqual([r.data[i * 4], r.data[i * 4 + 1], r.data[i * 4 + 2]], [10, 200, 90]);
  }
  const chk = rgba(8, 8, (x, y) => ((x + y) % 2 ? [255, 255, 255] : [0, 0, 0]));
  const r = resizeRGBA(chk, 2, 2);
  assert.ok(Math.abs(r.data[0] - 127.5) <= 1);
});

test('clipTensor: shortest+center crop and normalisation', () => {
  const img = rgba(60, 30, (x) => (x < 30 ? [255, 0, 0] : [0, 0, 255]));
  const t = clipTensor(img, { size: 16, resize: 'shortest', crop: 'center', interpolation: 'bicubic', mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] });
  assert.equal(t.length, 3 * 16 * 16);
  assert.ok(Math.abs(t[8 * 16 + 0] - 1) < 1e-6);                 // left column of the centre crop is red -> R=+1
  assert.ok(Math.abs(t[8 * 16 + 0 + 256 * 2] - -1) < 1e-6);      // ... B=-1
  const sq = clipTensor(img, { size: 16, resize: 'squash', crop: null, mean: [0, 0, 0], std: [1, 1, 1] });
  assert.ok(sq[8 * 16 + 1] > 0.99 && sq[8 * 16 + 14] < 0.01);    // squash keeps both halves
});

test('crop rect pads and clamps', () => {
  assert.deepEqual(cropRect({ x1: 10, y1: 10, x2: 50, y2: 30 }, 100, 100, 0.1), { x: 6, y: 8, w: 48, h: 24 });
  assert.deepEqual(cropRect({ x1: 0, y1: 0, x2: 100, y2: 100 }, 100, 100, 0.2), { x: 0, y: 0, w: 100, h: 100 });
  const c = cropRGBA(rgba(4, 4, (x, y) => [x, y, 0]), { x: 1, y: 2, w: 2, h: 1 });
  assert.deepEqual([c.data[0], c.data[1], c.data[4], c.data[5]], [1, 2, 2, 2]);
  assert.deepEqual(fitSize(4000, 3000, 1280), { w: 1280, h: 960 });
  assert.deepEqual(fitSize(500, 300, 1280), { w: 500, h: 300 });
});

const vocab: VocabRow[] = [
  { id: 'cup', en: 'empty cup', kcal100: 0, portion_g: [0, 0, 0], neg: true },
  { id: 'rice', en: 'rice', zh: '白飯', kcal100: 130, portion_g: [100, 200, 300] },
  { id: 'noodle', en: 'noodles', zh: '麵', kcal100: 140, portion_g: [150, 250, 400] },
  { id: 'soup', en: 'soup', kcal100: 30, portion_g: [150, 250, 400] },
];

test('softmax / classify gate and candidates', () => {
  const s = softmax([1, 2, 3]);
  assert.ok(Math.abs(s.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  const mix = { k: 2, cum: 0.9, sure: 0.5 }, gate = { p_neg_max: 0.5, p_food_min: 0.3 };
  const neg = classify(new Float32Array([0.9, 0.1, 0.1, 0.1]), vocab, 20, gate, mix);
  assert.equal(neg.skip, 'negative');
  const low = classify(new Float32Array([0, 0.3, 0.3, 0.3]), vocab, 1, { p_neg_max: 0.9, p_food_min: 0.5 }, mix);
  assert.equal(low.skip, 'low_food');
  const ok = classify(new Float32Array([0, 0.9, 0.85, 0.1]), vocab, 20, gate, mix);
  assert.equal(ok.skip, null);
  assert.equal(ok.cands.length, 2);                              // k=2
  assert.deepEqual(ok.cands.map(c => vocab[c.idx].id), ['rice', 'noodle']);
  assert.ok(Math.abs(ok.cands[0].p + ok.cands[1].p - 1) < 1e-6);
  assert.ok(ok.top1 < 0.8 && ok.top1 > 0.5);
  const e = l2normalize(new Float32Array([3, 4]));
  assert.ok(Math.abs(e[0] - 0.6) < 1e-6);
  assert.deepEqual(Array.from(similarities(e, new Float32Array([0.6, 0.8, 1, 0]), 2, 2)).map(v => +v.toFixed(3)), [1, 0.6]);
});

test('PRNG is deterministic; mixture percentiles are sane and ordered', () => {
  assert.equal(mulberry32(42)(), mulberry32(42)());
  assert.notEqual(hashSeed('a'), hashSeed('b'));
  const cands = [{ p: 1, kcal100: 130, portion: [100, 200, 400] as [number, number, number] }];
  const a = mixtureSamples(cands, 4000, 1), b = mixtureSamples(cands, 4000, 1);
  assert.deepEqual(Array.from(a.slice(0, 5)), Array.from(b.slice(0, 5)));
  const r = summarize(a);
  assert.ok(r.p10 < r.p50 && r.p50 < r.p90);
  assert.ok(Math.abs(r.p50 - 260) < 12, `median ${r.p50}`);      // 200 g * 1.3
  // 5th/95th percentiles of grams ~ 100/400 -> kcal 130/520
  assert.ok(Math.abs(percentile(a, 0.05) - 130) < 20 && Math.abs(percentile(a, 0.95) - 520) < 60);
  const mixed = summarize(mixtureSamples([
    { p: 0.5, kcal100: 100, portion: [100, 100, 100] }, { p: 0.5, kcal100: 300, portion: [100, 100, 100] }], 4000, 2));
  assert.ok(mixed.p10 < 120 && mixed.p90 > 280);                 // bimodal mixture spreads the range
});

test('quantile arrays recombine into sums', () => {
  const s = mixtureSamples([{ p: 1, kcal100: 100, portion: [100, 200, 400] }], 4000, 5);
  const q = quantileArray(s);
  assert.equal(q.length, 64);
  assert.ok(q.every((v, i) => i === 0 || v >= q[i - 1]));
  assert.ok(Math.abs(sampleQuantiles(q, 0.5) - percentile(s, 0.5)) < 3);
  const it = (id: string): LogItem => ({ id, name: id, nameEn: id, unsure: false, box: [0, 0, 1, 1], ...summarize(s), q, cands: [] });
  const two = sumRange([it('a'), it('b')], 'k')!;
  const mean = Array.from(s).reduce((a, b) => a + b, 0) / s.length; assert.ok(two.p50 > 2 * percentile(s, 0.5) && two.p50 < 2 * mean);
  assert.ok(two.p90 - two.p10 < 2 * (percentile(s, 0.9) - percentile(s, 0.1)));   // independence: sum narrower than 2x
  assert.deepEqual(sumRange([it('a'), it('b')], 'k'), two);       // deterministic
  assert.equal(sumRange([], 'x'), null);
});

test('format', () => {
  assert.equal(round10(314), 310); assert.equal(round10(315), 320);
  assert.equal(fmtRange({ p10: 312, p90: 523 }), '310–520');
  assert.equal(fmtRange({ p10: 301, p90: 304 }), '≈300');
  assert.equal(displayName('白飯', true), '白飯 或類似');
  assert.equal(displayName('白飯', false), '白飯');
});

test('log aggregation: days, totals, week, delete item, backup', () => {
  const q = Array.from({ length: 64 }, (_, i) => 100 + i);
  const item = (id: string): LogItem => ({ id, name: id, nameEn: id, unsure: false, box: [0, 0, 1, 1], p10: 105, p50: 132, p90: 158, q, cands: [] });
  const noon = (d: string) => new Date(`${d}T12:00:00`).getTime();
  const meals: Meal[] = [
    { id: 'm1', ts: noon('2026-10-30'), thumb: null, items: [item('a')] },
    { id: 'm2', ts: noon('2026-10-31') + 1, thumb: null, items: [item('b'), item('c')] },
    { id: 'm3', ts: noon('2026-10-31') + 2, thumb: null, items: [item('d')] },
  ];
  assert.equal(dayKey(noon('2026-10-31')), '2026-10-31');
  assert.equal(shiftDay('2026-10-31', 1), '2026-11-01');
  assert.equal(shiftDay('2026-03-01', -1), '2026-02-28');
  assert.equal(mealsOfDay(meals, '2026-10-31').length, 2);
  assert.equal(mealTotal(meals[0])!.p50, 132);
  const dt = dayTotal(mealsOfDay(meals, '2026-10-31'))!;
  assert.ok(Math.abs(dt.p50 - 3 * 131.5) < 8);
  const wk = weekMedians(meals, '2026-10-31');
  assert.equal(wk.length, 7); assert.equal(wk[6].key, '2026-10-31');
  assert.ok(wk[5].median === 132 && wk[0].median === null);
  assert.equal(removeItem(meals[1], 'b')!.items.length, 1);
  assert.equal(removeItem(meals[0], 'a'), null);
  const text = JSON.stringify({ app: 'calorie-lens', version: 1, meals: meals.map(m => ({ ...m, thumb: 'data:image/jpeg;base64,AAAA' })) });
  const back = parseBackup(text);
  assert.equal(back.length, 3); assert.equal(back[1].items.length, 2); assert.equal(back[0].thumb, 'data:image/jpeg;base64,AAAA');
  assert.throws(() => parseBackup('{"app":"x"}'));
  assert.throws(() => parseBackup('nope'));
});

test('pipeline with fake runners: boxes, negatives, whole-image fallback', async () => {
  const dim = 4;
  const emb = new Float32Array(4 * dim); for (let i = 0; i < 4; i++) emb[i * dim + i] = 1; // one-hot rows
  const assets: Assets = {
    locator: { imgsz: 64, input: 'images', output: 'output0', classes: [{ name: 'food', food: true }, { name: 'cup', food: false }],
      conf: 0.25, iou: 0.5, contain: 0.9, max_det: 10, crop_pad: 0.1 },
    clip: { input: 'x', output: 'y', size: 8, resize: 'shortest', crop: 'center', interpolation: 'bilinear', mean: [0, 0, 0], std: [1, 1, 1],
      dim, logit_scale: 100, gate: { p_neg_max: 0.5, p_food_min: 0.3 }, mix: { k: 3, cum: 0.95, sure: 0.6 } },
    vocab, emb,
  };
  const img = rgba(128, 64, (x) => (x < 64 ? [255, 0, 0] : [0, 255, 0]));
  const N = 8, out = new Float32Array(6 * N);
  const box = (i: number, cx: number, cy: number, w: number, h: number, s0: number, s1: number) => {
    out[i] = cx; out[N + i] = cy; out[2 * N + i] = w; out[3 * N + i] = h; out[4 * N + i] = s0; out[5 * N + i] = s1;
  };
  box(0, 16, 32, 20, 20, 0.9, 0); box(1, 48, 32, 20, 20, 0.8, 0); box(2, 32, 20, 6, 6, 0, 0.7);
  const embed = async (t: Float32Array) => { // red crop -> rice, green crop -> noodle
    const n = 8 * 8, e = new Float32Array(dim);
    let r = 0, g = 0; for (let i = 0; i < n; i++) { r += t[i]; g += t[n + i]; }
    if (r > g) e[1] = 1; else e[2] = 1;
    return e;
  };
  const res = await analyze(img, assets, { locate: async () => ({ data: out, dims: [1, 6, N] }), embed });
  assert.equal(res.whole, false);
  assert.equal(res.items.length, 2);                              // cup box discarded
  assert.deepEqual(res.items.map(i => i.name), ['白飯', '麵']);
  assert.ok(res.items.every(i => i.p10 < i.p50 && i.p50 < i.p90 && i.q.length === 64 && i.cands[0].p > 0.9));
  assert.ok(res.items[0].box[0] >= 0 && res.items[0].box[2] <= 1);
  assert.ok(res.total && res.total.p50 > res.items[0].p50);
  const empty = { locate: async () => ({ data: new Float32Array(6 * N), dims: [1, 6, N] }) };
  const none = await analyze(img, assets, { ...empty, embed });
  assert.equal(none.whole, true); assert.equal(none.items.length, 1);
  assert.deepEqual(none.items[0].box, [0, 0, 1, 1]);
  const cup = await analyze(img, assets, { ...empty, embed: async () => Float32Array.from([1, 0, 0, 0]) });
  assert.equal(cup.items.length, 0); assert.equal(cup.skipped.negative, 1);
});
