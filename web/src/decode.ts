// Photo decode (EXIF orientation applied, long edge <= 1280) + 320 px JPEG thumbnail.
// Works in a worker (OffscreenCanvas) or on the main thread (fallback for older iOS). Throws 'NO_CANVAS' when neither
// 2D canvas is available in the current context, so the worker can ask the main thread to decode instead.
import { fitSize } from './imageops.ts';

export const MAX_EDGE = 1280;
export const THUMB_EDGE = 320;

type AnyCanvas = { canvas: OffscreenCanvas | HTMLCanvasElement; ctx: any };

function makeCanvas(w: number, h: number): AnyCanvas {
  if (typeof OffscreenCanvas !== 'undefined') {
    try {
      const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx && typeof (canvas as any).convertToBlob === 'function') return { canvas, ctx };
    } catch { /* fall through */ }
  }
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (ctx) return { canvas, ctx };
  }
  throw new Error('NO_CANVAS');
}

function toBlob(c: AnyCanvas, type: string, q: number): Promise<Blob> {
  if ('convertToBlob' in c.canvas) return (c.canvas as OffscreenCanvas).convertToBlob({ type, quality: q });
  return new Promise((res, rej) => (c.canvas as HTMLCanvasElement).toBlob(b => (b ? res(b) : rej(new Error('toBlob failed'))), type, q));
}

async function toBitmap(file: Blob): Promise<{ src: CanvasImageSource; w: number; h: number; close(): void }> {
  if (typeof createImageBitmap === 'function') {
    try {
      const b = await createImageBitmap(file, { imageOrientation: 'from-image' } as ImageBitmapOptions);
      return { src: b, w: b.width, h: b.height, close: () => b.close() };
    } catch { /* options unsupported or decode failure: try plain */ }
    try {
      const b = await createImageBitmap(file);
      return { src: b, w: b.width, h: b.height, close: () => b.close() };
    } catch { /* fall through to <img> */ }
  }
  if (typeof document === 'undefined') throw new Error('NO_CANVAS');
  const url = URL.createObjectURL(file);
  try {
    const img = new Image(); img.src = url; await img.decode(); // <img> applies EXIF orientation by default
    return { src: img, w: img.naturalWidth, h: img.naturalHeight, close: () => {} };
  } finally { URL.revokeObjectURL(url); }
}

export interface Decoded { rgba: Uint8ClampedArray; w: number; h: number; thumb: Blob }

export async function decodeFile(file: Blob): Promise<Decoded> {
  makeCanvas(1, 1); // fail fast with NO_CANVAS
  const bm = await toBitmap(file);
  try {
    const { w, h } = fitSize(bm.w, bm.h, MAX_EDGE);
    const c = makeCanvas(w, h);
    c.ctx.imageSmoothingQuality = 'high';
    c.ctx.drawImage(bm.src, 0, 0, w, h);
    const rgba = c.ctx.getImageData(0, 0, w, h).data as Uint8ClampedArray;
    const t = fitSize(w, h, THUMB_EDGE), tc = makeCanvas(t.w, t.h);
    tc.ctx.imageSmoothingQuality = 'high';
    tc.ctx.drawImage(c.canvas as CanvasImageSource, 0, 0, t.w, t.h);
    return { rgba, w, h, thumb: await toBlob(tc, 'image/jpeg', 0.8) };
  } finally { bm.close(); }
}
