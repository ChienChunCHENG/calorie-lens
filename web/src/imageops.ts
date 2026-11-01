// Pure pixel ops on raw RGBA arrays (so they run in node tests, in workers and on the main thread alike).
import { letterboxParams } from './yolo.ts';
import type { Letterbox } from './yolo.ts';

export interface Rgba { data: Uint8ClampedArray | Uint8Array; w: number; h: number }
export interface Rect { x: number; y: number; w: number; h: number }
export type Interp = 'bilinear' | 'bicubic';

const tri = (x: number) => { x = Math.abs(x); return x < 1 ? 1 - x : 0; };
const cub = (x: number) => { // PIL's bicubic (a = -0.5)
  x = Math.abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return (((-0.5 * x + 2.5) * x) - 4) * x + 2;
  return 0;
};

interface Coeffs { start: Int32Array; len: Int32Array; w: Float32Array; stride: number }

function coeffs(inSize: number, outSize: number, interp: Interp): Coeffs {
  const scale = inSize / outSize, fs = Math.max(scale, 1);
  const support = (interp === 'bicubic' ? 2 : 1) * fs, k = interp === 'bicubic' ? cub : tri;
  const stride = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(outSize), len = new Int32Array(outSize), w = new Float32Array(outSize * stride);
  for (let i = 0; i < outSize; i++) {
    const c = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(c - support + 0.5)), hi = Math.min(inSize, Math.floor(c + support + 0.5));
    let sum = 0;
    for (let j = lo; j < hi; j++) { const v = k((j - c + 0.5) / fs); w[i * stride + (j - lo)] = v; sum += v; }
    if (sum !== 0) for (let j = 0; j < hi - lo; j++) w[i * stride + j] /= sum;
    start[i] = lo; len[i] = hi - lo;
  }
  return { start, len, w, stride };
}

/** Antialiased separable resize (PIL-style kernels). Returns RGBA with alpha 255. */
export function resizeRGBA(src: Rgba, dw: number, dh: number, interp: Interp = 'bilinear'): Rgba {
  const { data, w: sw, h: sh } = src;
  const cx = coeffs(sw, dw, interp), cy = coeffs(sh, dh, interp);
  const tmp = new Float32Array(dw * sh * 3);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      let r = 0, g = 0, b = 0;
      const s = cx.start[x], n = cx.len[x], wo = x * cx.stride;
      for (let j = 0; j < n; j++) {
        const wt = cx.w[wo + j], p = (y * sw + s + j) * 4;
        r += data[p] * wt; g += data[p + 1] * wt; b += data[p + 2] * wt;
      }
      const o = (y * dw + x) * 3; tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
  }
  const out = new Uint8ClampedArray(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    const s = cy.start[y], n = cy.len[y], wo = y * cy.stride;
    for (let x = 0; x < dw; x++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < n; j++) {
        const wt = cy.w[wo + j], p = ((s + j) * dw + x) * 3;
        r += tmp[p] * wt; g += tmp[p + 1] * wt; b += tmp[p + 2] * wt;
      }
      const o = (y * dw + x) * 4; out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255; // clamped array rounds
    }
  }
  return { data: out, w: dw, h: dh };
}

export function cropRGBA(src: Rgba, r: Rect): Rgba {
  const out = new Uint8ClampedArray(r.w * r.h * 4);
  for (let y = 0; y < r.h; y++) {
    const so = ((r.y + y) * src.w + r.x) * 4;
    out.set(src.data.subarray(so, so + r.w * 4), y * r.w * 4);
  }
  return { data: out, w: r.w, h: r.h };
}

/** Expand a box by `pad` of its own size on every side, clamp to the image, return an integer rect (>= 1 px). */
export function cropRect(b: { x1: number; y1: number; x2: number; y2: number }, imgW: number, imgH: number, pad: number): Rect {
  const bw = b.x2 - b.x1, bh = b.y2 - b.y1;
  const x1 = Math.max(0, Math.floor(b.x1 - pad * bw)), y1 = Math.max(0, Math.floor(b.y1 - pad * bh));
  const x2 = Math.min(imgW, Math.ceil(b.x2 + pad * bw)), y2 = Math.min(imgH, Math.ceil(b.y2 + pad * bh));
  return { x: x1, y: y1, w: Math.max(1, x2 - x1), h: Math.max(1, y2 - y1) };
}

/** Fit so the long edge is <= maxEdge (never upscales). */
export function fitSize(w: number, h: number, maxEdge: number): { w: number; h: number } {
  const s = Math.min(1, maxEdge / Math.max(w, h));
  return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)) };
}

/** Locator input: letterbox with gray 114, RGB 0..1, CHW float32. */
export function letterboxTensor(img: Rgba, size: number): { data: Float32Array; lb: Letterbox } {
  const lb = letterboxParams(img.w, img.h, size);
  const r = lb.newW === img.w && lb.newH === img.h ? img : resizeRGBA(img, lb.newW, lb.newH, 'bilinear');
  const n = size * size, d = new Float32Array(3 * n).fill(114 / 255);
  for (let y = 0; y < lb.newH; y++) {
    for (let x = 0; x < lb.newW; x++) {
      const p = (y * lb.newW + x) * 4, o = (y + lb.padY) * size + x + lb.padX;
      d[o] = r.data[p] / 255; d[n + o] = r.data[p + 1] / 255; d[2 * n + o] = r.data[p + 2] / 255;
    }
  }
  return { data: d, lb };
}

export interface ClipPre {
  size: number; resize: 'shortest' | 'squash'; crop: 'center' | null; interpolation?: string;
  mean: number[]; std: number[];
}

/** CLIP image preprocessing -> CHW float32 (size x size), as described by clip.json. */
export function clipTensor(img: Rgba, c: ClipPre): Float32Array {
  const interp: Interp = /cubic/i.test(c.interpolation ?? '') ? 'bicubic' : 'bilinear';
  let nw: number, nh: number;
  if (c.resize === 'squash') { nw = c.size; nh = c.size; }
  else if (img.w <= img.h) { nw = c.size; nh = Math.max(c.size, Math.floor(c.size * img.h / img.w)); }
  else { nh = c.size; nw = Math.max(c.size, Math.floor(c.size * img.w / img.h)); }
  let r = nw === img.w && nh === img.h ? img : resizeRGBA(img, nw, nh, interp);
  if (r.w !== c.size || r.h !== c.size) { // center crop (torchvision rounding)
    const x = Math.max(0, Math.round((r.w - c.size) / 2)), y = Math.max(0, Math.round((r.h - c.size) / 2));
    r = cropRGBA(r, { x, y, w: Math.min(c.size, r.w), h: Math.min(c.size, r.h) });
    if (r.w !== c.size || r.h !== c.size) r = resizeRGBA(r, c.size, c.size, interp);
  }
  const n = c.size * c.size, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) out[k * n + i] = (r.data[i * 4 + k] / 255 - c.mean[k]) / c.std[k];
  }
  return out;
}
