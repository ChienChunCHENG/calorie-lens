// Parity check: JS re-implementation of the snap preprocessing + locator decode, run through onnxruntime-node, compared with
// web/test/fixtures/parity.json (written by ml/snap/make_parity.py from the Python pipeline).
//   cd web && node ../ml/snap/parity_node.mjs
// Image decoding is NOT tested here: the fixture carries raw RGB (decoded by PIL). Only resize/normalise/letterbox, model I/O,
// NMS/containment decode and the embedding/similarity maths are checked.
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(path.join(process.cwd(), 'package.json'));
const ort = require('onnxruntime-node');

const models = path.resolve('public/models');
const fx = JSON.parse(fs.readFileSync(path.resolve('test/fixtures/parity.json'), 'utf8'));
const clipCfg = JSON.parse(fs.readFileSync(path.join(models, 'clip.json'), 'utf8'));
const loc = JSON.parse(fs.readFileSync(path.join(models, 'locator.json'), 'utf8'));
const vocab = JSON.parse(fs.readFileSync(path.join(models, 'vocab.json'), 'utf8'));
const embBuf = fs.readFileSync(path.join(models, 'vocab_emb.bin'));
const dim = clipCfg.dim;

function f16(u) { // IEEE half -> float
  const s = (u & 0x8000) ? -1 : 1, e = (u >> 10) & 31, m = u & 1023;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}
const emb = new Float32Array(vocab.length * dim);
for (let i = 0; i < emb.length; i++) emb[i] = f16(embBuf.readUInt16LE(i * 2));

const b64 = (s) => new Uint8Array(Buffer.from(s, 'base64'));

// bilinear resize of interleaved RGB uint8, half-pixel centres, edge clamp, uint8 rounding (matches PIL for upscaling)
function resizeRGB(src, w, h, nw, nh) {
  const out = new Uint8Array(nw * nh * 3), sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), h - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, h - 1), wy = fy - y0;
    for (let x = 0; x < nw; x++) {
      const fx_ = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), w - 1), x0 = Math.floor(fx_), x1 = Math.min(x0 + 1, w - 1), wx = fx_ - x0;
      for (let c = 0; c < 3; c++) {
        const v = (src[(y0 * w + x0) * 3 + c] * (1 - wx) + src[(y0 * w + x1) * 3 + c] * wx) * (1 - wy) +
                  (src[(y1 * w + x0) * 3 + c] * (1 - wx) + src[(y1 * w + x1) * 3 + c] * wx) * wy;
        out[(y * nw + x) * 3 + c] = Math.round(v);
      }
    }
  }
  return out;
}

function clipInput(rgb, w, h) {
  const S = clipCfg.size; let img = rgb, iw = w, ih = h;
  if (clipCfg.resize === 'squash') { img = resizeRGB(rgb, w, h, S, S); iw = ih = S; }
  else {
    const r = S / Math.min(w, h); iw = Math.max(S, Math.round(w * r)); ih = Math.max(S, Math.round(h * r)); img = resizeRGB(rgb, w, h, iw, ih);
    if (clipCfg.crop === 'center') {
      const l = Math.floor((iw - S) / 2), t = Math.floor((ih - S) / 2), c = new Uint8Array(S * S * 3);
      for (let y = 0; y < S; y++) for (let x = 0; x < S * 3; x++) c[y * S * 3 + x] = img[((y + t) * iw + l) * 3 + x];
      img = c; iw = ih = S;
    }
  }
  const t = new Float32Array(3 * iw * ih);
  for (let i = 0; i < iw * ih; i++) for (let c = 0; c < 3; c++) t[c * iw * ih + i] = (img[i * 3 + c] / 255 - clipCfg.mean[c]) / clipCfg.std[c];
  return { t, iw, ih };
}

function letterbox(rgb, w, h, size) {
  const r = Math.min(size / w, size / h), nw = Math.round(w * r), nh = Math.round(h * r), px = Math.floor((size - nw) / 2), py = Math.floor((size - nh) / 2);
  const res = resizeRGB(rgb, w, h, nw, nh), t = new Float32Array(3 * size * size).fill(114 / 255);
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) for (let c = 0; c < 3; c++) t[c * size * size + (y + py) * size + (x + px)] = res[(y * nw + x) * 3 + c] / 255;
  return { t, r, px, py };
}

const iouF = (a, b) => { const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1), u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i; return u > 0 ? i / u : 0; };
const insideF = (a, b) => { const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1), ar = (a[2] - a[0]) * (a[3] - a[1]); return ar > 0 ? i / ar : 0; };

function decode(out, dims, lb, W, H) { // out: Float32Array [1, C, N]
  const C = dims[1], N = dims[2], nc = loc.classes.length, cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, bi = 0;
    for (let k = 0; k < nc; k++) { const s = out[(4 + k) * N + i]; if (s > best) { best = s; bi = k; } }
    if (best >= loc.conf) cand.push({ i, score: best, cls: bi });
  }
  cand.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cand) {
    const cx = out[c.i], cy = out[N + c.i], w = out[2 * N + c.i], h = out[3 * N + c.i];
    let b = [(cx - w / 2 - lb.px) / lb.r, (cy - h / 2 - lb.py) / lb.r, (cx + w / 2 - lb.px) / lb.r, (cy + h / 2 - lb.py) / lb.r];
    b = [Math.max(0, b[0]), Math.max(0, b[1]), Math.min(W, b[2]), Math.min(H, b[3])];
    if (b[2] - b[0] < 4 || b[3] - b[1] < 4) continue;
    if (kept.some((k) => iouF(b, k.box) > loc.iou)) continue;
    kept.push({ box: b, score: c.score, cls: c.cls });
    if (kept.length >= 100) break;
  }
  const out2 = [];
  for (const c of kept) { if (out2.some((o) => insideF(c.box, o.box) >= loc.contain)) continue; out2.push(c); }
  return out2.slice(0, loc.max_det).map((c) => ({ ...c, food: loc.classes[c.cls].food }));
}

let fails = 0;
const check = (name, ok, info) => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (info ? '  ' + info : '')); if (!ok) fails++; };

const clipSess = await ort.InferenceSession.create(path.join(models, clipCfg.input ? 'clip_image.onnx' : 'clip_image.onnx'));
for (const im of fx.images) {
  const rgb = b64(im.rgb_b64), { t, iw } = clipInput(rgb, im.w, im.h);
  const dt = Math.max(...im.input_first8.map((v, k) => Math.abs(v - t[k])));
  check(`${im.file} preprocess first8`, dt < 0.02, `maxdiff ${dt.toFixed(4)}`);
  const res = await clipSess.run({ [clipCfg.input]: new ort.Tensor('float32', t, [1, 3, iw, iw]) });
  const e = Float32Array.from(res[clipCfg.output].data); let n = 0; for (const v of e) n += v * v; n = Math.sqrt(n); for (let i = 0; i < e.length; i++) e[i] /= n;
  const dot = (a, b) => { let s = 0; for (let i = 0; i < dim; i++) s += a[i] * b[i]; return s; };
  const ref = Float32Array.from(im.embedding_full);
  check(`${im.file} embedding cosine vs python`, dot(e, ref) > 0.995, `cos ${dot(e, ref).toFixed(5)}`);
  check(`${im.file} embedding first8`, Math.max(...im.embedding_first8.map((v, k) => Math.abs(v - e[k]))) < 0.03);
  for (const r of im.vocab_cos) {
    const row = vocab.findIndex((v) => v.id === r.id), c = dot(e, emb.subarray(row * dim, (row + 1) * dim));
    check(`${im.file} cos vs ${r.id}`, Math.abs(c - r.cos) < 0.01, `${c.toFixed(4)} vs ${r.cos.toFixed(4)}`);
  }
  const sims = []; for (let i = 0; i < vocab.length; i++) sims.push([dot(e, emb.subarray(i * dim, (i + 1) * dim)), vocab[i].id]);
  sims.sort((a, b) => b[0] - a[0]); const top3 = sims.slice(0, 3).map((s) => s[1]);
  const tie = im.top3_min_gap < 0.004; // near-ties may swap order between fp16/JS rounding: then compare as sets
  const same = tie ? JSON.stringify([...top3].sort()) === JSON.stringify([...im.top3].sort()) : JSON.stringify(top3) === JSON.stringify(im.top3);
  check(`${im.file} top-3 ids${tie ? ' (near-tie: set compare)' : ''}`, same, `${top3} vs ${im.top3}`);
}

const L = fx.locator, lrgb = b64(L.rgb_b64);
const lb = letterbox(lrgb, L.w, L.h, loc.imgsz);
const lsess = await ort.InferenceSession.create(path.join(models, 'locator.onnx'));
const lres = await lsess.run({ [loc.input]: new ort.Tensor('float32', lb.t, [1, 3, loc.imgsz, loc.imgsz]) });
const lo = lres[loc.output];
const boxes = decode(lo.data, lo.dims, lb, L.w, L.h);
check('locator box count', boxes.length === L.boxes.length, `${boxes.length} vs ${L.boxes.length}`);
for (const eb of L.boxes) {
  const m = Math.max(0, ...boxes.map((b) => iouF(b.box, eb.box)));
  check(`locator box ${JSON.stringify(eb.box.map(Math.round))} ${eb.name}`, m > 0.9, `best IoU ${m.toFixed(3)}`);
}
console.log(fails ? `${fails} FAILED` : 'ALL PARITY CHECKS PASSED');
process.exit(fails ? 1 : 0);
