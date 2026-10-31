export interface Box { x1: number; y1: number; x2: number; y2: number }
export interface Detection extends Box { cls: number; score: number }

export interface Letterbox { scale: number; padX: number; padY: number; size: number }

export function iou(a: Box, b: Box): number {
  const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** Draw a video frame into a size x size canvas, preserving aspect, gray padding; return CHW float32 [0,1]. */
export class Letterboxer {
  private ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  private buf: Float32Array;
  size: number;
  constructor(size: number) {
    this.size = size;
    const c: HTMLCanvasElement | OffscreenCanvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(size, size)
      : Object.assign(document.createElement('canvas'), { width: size, height: size });
    this.ctx = c.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D;
    this.buf = new Float32Array(3 * size * size);
  }
  run(video: HTMLVideoElement): { data: Float32Array; lb: Letterbox } {
    const s = this.size, vw = video.videoWidth, vh = video.videoHeight;
    const scale = s / Math.max(vw, vh);
    const w = Math.round(vw * scale), h = Math.round(vh * scale);
    const padX = Math.floor((s - w) / 2), padY = Math.floor((s - h) / 2);
    this.ctx.fillStyle = 'rgb(114,114,114)';
    this.ctx.fillRect(0, 0, s, s);
    this.ctx.drawImage(video, padX, padY, w, h);
    const px = this.ctx.getImageData(0, 0, s, s).data;
    const n = s * s, d = this.buf;
    for (let i = 0; i < n; i++) {
      d[i] = px[i * 4] / 255;
      d[n + i] = px[i * 4 + 1] / 255;
      d[2 * n + i] = px[i * 4 + 2] / 255;
    }
    return { data: d, lb: { scale, padX, padY, size: s } };
  }
}

/**
 * Decode Ultralytics YOLO output [1, 4+nc(+32 mask coefs), N] (cx,cy,w,h in letterboxed pixels) to video-space
 * detections. classMap folds several model classes (prompts) into one output class; NMS is class-agnostic when
 * `agnostic`, so two prompts firing on the same dish keep only the stronger box.
 */
export function decodeYolo(out: Float32Array, dims: readonly number[], lb: Letterbox, vw: number, vh: number,
  confThr = 0.25, iouThr = 0.45, maxDet = 50, classMap?: readonly number[], agnostic = false): Detection[] {
  const N = dims[2], nc = classMap ? classMap.length : dims[1] - 4;
  const cands: Detection[] = [];
  for (let i = 0; i < N; i++) {
    let best = -1, bs = confThr;
    for (let c = 0; c < nc; c++) {
      const s = out[(4 + c) * N + i];
      if (s > bs) { bs = s; best = c; }
    }
    if (best < 0) continue;
    const cx = out[i], cy = out[N + i], w = out[2 * N + i], h = out[3 * N + i];
    const x1 = Math.max(0, Math.min(vw, (cx - w / 2 - lb.padX) / lb.scale));
    const y1 = Math.max(0, Math.min(vh, (cy - h / 2 - lb.padY) / lb.scale));
    const x2 = Math.max(0, Math.min(vw, (cx + w / 2 - lb.padX) / lb.scale));
    const y2 = Math.max(0, Math.min(vh, (cy + h / 2 - lb.padY) / lb.scale));
    if (x2 - x1 < 2 || y2 - y1 < 2) continue;
    cands.push({ x1, y1, x2, y2, cls: classMap ? classMap[best] : best, score: bs });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  for (const d of cands) {
    let ok = true;
    for (const k of keep) if ((agnostic || k.cls === d.cls) && iou(k, d) > iouThr) { ok = false; break; }
    if (ok) { keep.push(d); if (keep.length >= maxDet) break; }
  }
  return keep;
}

/**
 * Drops a box that lies mostly (>= frac of its area) inside another box when both name the same class (keeps the
 * higher score: a dish boxed twice) or the outer one is a whole-bowl dish (the beef inside 牛肉麵 is not a second
 * dish). Plain NMS misses both because a small box inside a big one has low IoU.
 */
export function suppressContained<T extends Detection>(dets: T[], wholeCls: ReadonlySet<number>, frac = 0.7): T[] {
  const area = (b: Box) => (b.x2 - b.x1) * (b.y2 - b.y1);
  const inside = (a: Box, b: Box) => {
    const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1), ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
    return iw > 0 && ih > 0 ? (iw * ih) / area(a) : 0;
  };
  return dets.filter(d => !dets.some(k => {
    if (k === d) return false;
    if (k.cls === d.cls) return k.score > d.score && (inside(d, k) >= frac || inside(k, d) >= frac);
    return wholeCls.has(k.cls) && area(k) > area(d) && inside(d, k) >= frac;
  }));
}
