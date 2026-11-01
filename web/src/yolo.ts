// Pure YOLO(E) helpers: letterbox math, output decoding, class-agnostic NMS, containment suppression.
export interface Box { x1: number; y1: number; x2: number; y2: number }
export interface Detection extends Box { cls: number; score: number }
export interface Letterbox { scale: number; padX: number; padY: number; size: number; newW: number; newH: number }

export function area(b: Box): number { return Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1); }

export function interArea(a: Box, b: Box): number {
  const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  return iw > 0 && ih > 0 ? iw * ih : 0;
}

export function iou(a: Box, b: Box): number {
  const inter = interArea(a, b);
  const ua = area(a) + area(b) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** Ultralytics-style letterbox: longest side -> size, centred, remaining area padded. */
export function letterboxParams(w: number, h: number, size: number): Letterbox {
  const scale = size / Math.max(w, h);
  const newW = Math.max(1, Math.round(w * scale)), newH = Math.max(1, Math.round(h * scale));
  return { scale, padX: Math.floor((size - newW) / 2), padY: Math.floor((size - newH) / 2), size, newW, newH };
}

export interface DecodeOpts { nc: number; conf: number }

/**
 * Decode output0 laid out [1, 4+nc(+extra), N] (cx,cy,w,h in letterbox px) into image-space candidates (sorted by
 * score, best class per anchor, only the first `nc` class rows are read). A transposed [1, N, C] layout is detected
 * when dims[1] > dims[2].
 */
export function decodeYolo(out: ArrayLike<number>, dims: readonly number[], lb: Letterbox, imgW: number, imgH: number,
  { nc, conf }: DecodeOpts): Detection[] {
  const transposed = dims[1] > dims[2];
  const N = transposed ? dims[1] : dims[2], C = transposed ? dims[2] : dims[1];
  if (C < 4 + nc) throw new Error(`locator output has ${C} channels, need >= ${4 + nc}`);
  const at = transposed ? (c: number, i: number) => out[i * C + c] : (c: number, i: number) => out[c * N + i];
  const cands: Detection[] = [];
  for (let i = 0; i < N; i++) {
    let best = -1, bs = conf;
    for (let c = 0; c < nc; c++) {
      const s = at(4 + c, i);
      if (s >= bs) { bs = s; best = c; }
    }
    if (best < 0) continue;
    const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    const cl = (v: number, m: number) => Math.max(0, Math.min(m, v));
    const x1 = cl((cx - w / 2 - lb.padX) / lb.scale, imgW), y1 = cl((cy - h / 2 - lb.padY) / lb.scale, imgH);
    const x2 = cl((cx + w / 2 - lb.padX) / lb.scale, imgW), y2 = cl((cy + h / 2 - lb.padY) / lb.scale, imgH);
    if (x2 - x1 < 2 || y2 - y1 < 2) continue;
    cands.push({ x1, y1, x2, y2, cls: best, score: bs });
  }
  cands.sort((a, b) => b.score - a.score);
  return cands;
}

/** Class-agnostic greedy NMS over score-sorted detections. */
export function nms<T extends Detection>(sorted: T[], iouThr: number): T[] {
  const keep: T[] = [];
  for (const d of sorted) if (keep.every(k => iou(k, d) <= iouThr)) keep.push(d);
  return keep;
}

/** Drop a box that lies >= `contain` (fraction of its own area) inside a higher-scoring box. Input score-sorted. */
export function suppressContained<T extends Detection>(sorted: T[], contain: number): T[] {
  const keep: T[] = [];
  for (const d of sorted) {
    const a = area(d);
    if (a > 0 && keep.some(k => k.score >= d.score && interArea(d, k) / a >= contain)) continue;
    keep.push(d);
  }
  return keep;
}

export interface SelectOpts extends DecodeOpts { iou: number; contain: number; maxDet: number }

export function selectBoxes(out: ArrayLike<number>, dims: readonly number[], lb: Letterbox, imgW: number, imgH: number,
  o: SelectOpts): Detection[] {
  const dets = suppressContained(nms(decodeYolo(out, dims, lb, imgW, imgH, o), o.iou), o.contain);
  return dets.slice(0, o.maxDet);
}
