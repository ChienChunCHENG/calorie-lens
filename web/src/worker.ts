// Module worker: decode -> locate -> classify -> kcal mixture. Models are fetched once and cached (assets.ts).
import * as ort from 'onnxruntime-web';
import { fetchAsset, fetchJson } from './assets.ts';
import { LensEngine } from './pipeline.ts';
import type { Assets, ClipCfg, LensTrack, LocatorCfg, Runners } from './pipeline.ts';
import type { VocabRow } from './classify.ts';
import { f16ArrayToF32, f32ArrayToF16 } from './f16.ts';

export interface InMsg { type: 'warm' | 'frame' | 'reset'; base?: string; ortBase?: string; threads?: number; bitmap?: ImageBitmap; t?: number }
export interface WorkerResult { tracks: LensTrack[]; ms: number; w: number; h: number; nBoxes: number; classified: number; t: number; prepMs: number; locMs: number; clsMs: number }
export type OutMsg =
  | { type: 'progress'; phase: 'download' | 'init'; loaded?: number; total?: number }
  | { type: 'ready'; backend: string }
  | { type: 'result'; r: WorkerResult }
  | { type: 'error'; message: string };

const post = (m: OutMsg) => (self as unknown as Worker).postMessage(m);

let ortReady = false;
function setupOrt(base: string, ortBase?: string, threads?: number) {
  if (ortReady) return;
  const cdn = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  // Prod: pinned ORT wasm from jsDelivr (the webgpu wasm is ~29 MB; keeps the Pages artifact small). Dev: public/ort.
  ort.env.wasm.wasmPaths = ortBase ?? (import.meta.env.PROD ? cdn : `${base}ort/`);
  ort.env.wasm.numThreads = threads ?? (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);
  ortReady = true;
}

interface Loaded { assets: Assets; runners: Runners; backend: string }
let loaded: Promise<Loaded> | null = null;

type Sess = { s: ort.InferenceSession; backend: string; buf: Uint8Array };
interface Probe { input: string; output: string; size: number }

/** Smooth deterministic test image (ramps + a blob), so activations look like a real photo rather than noise. */
function probeTensor(size: number): ort.Tensor {
  const n = size * size, d = new Float32Array(3 * n);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = y * size + x, r = Math.hypot(x - size * 0.55, y - size * 0.45) / size;
    d[i] = 0.2 + 0.6 * x / size; d[n + i] = r < 0.25 ? 0.8 : 0.3; d[2 * n + i] = 0.2 + 0.6 * y / size;
  }
  return new ort.Tensor('float32', d, [1, 3, size, size]);
}

async function outData(s: ort.InferenceSession, p: Probe): Promise<Float32Array> {
  const t = (await s.run({ [p.input]: probeTensor(p.size) }))[p.output];
  return (t.data ?? await t.getData()) as Float32Array;
}

/** WebGPU results must match wasm on the probe (cosine > 0.999, max error < 2% of the largest value). */
export function sameOutputs(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length || !a.length) return false;
  let dot = 0, na = 0, nb = 0, maxErr = 0, maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(b[i])) return false;
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    maxErr = Math.max(maxErr, Math.abs(a[i] - b[i])); maxAbs = Math.max(maxAbs, Math.abs(a[i]));
  }
  return dot / (Math.sqrt(na * nb) || 1) > 0.999 && maxErr <= 0.02 * (maxAbs || 1);
}

/** wasm always works; WebGPU (much faster on phones) is used only if it reproduces the wasm output on a probe. */
async function createSession(buf: Uint8Array, probe: Probe): Promise<Sess> {
  const wasm = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  if (!(self.navigator as any)?.gpu) return { s: wasm, backend: 'wasm', buf };
  try {
    const gpu = await ort.InferenceSession.create(buf, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
    if (sameOutputs(await outData(wasm, probe), await outData(gpu, probe))) {
      await wasm.release();
      return { s: gpu, backend: 'webgpu', buf };
    }
    console.warn('webgpu self-test mismatch; using wasm');
    await gpu.release();
  } catch (e) { console.warn('webgpu unavailable; using wasm', e); }
  return { s: wasm, backend: 'wasm', buf };
}

async function load(base: string, ortBase: string | undefined, threads?: number): Promise<Loaded> {
  setupOrt(base, ortBase, threads);
  const m = `${base}models/`;
  const files = ['vocab_emb.bin', 'locator.onnx', 'clip_image.onnx'];
  // One combined total: sizes from HEAD requests up front so the percentage never jumps backwards.
  const heads = await Promise.all(files.map(f => fetch(m + f, { method: 'HEAD' })
    .then(r => (r.ok && !(r.headers.get('content-type') || '').includes('text/html') ? Number(r.headers.get('content-length')) || 0 : 0)).catch(() => 0)));
  const sizes = new Map<string, [number, number]>(files.map((f, i) => [f, [0, heads[i]]]));
  const prog = (name: string) => (l: number, t: number) => {
    sizes.set(name, [l, Math.max(t || 0, sizes.get(name)![1], l)]);
    let L = 0, T = 0; for (const [a, b] of sizes.values()) { L += a; T += b; }
    post({ type: 'progress', phase: 'download', loaded: Math.min(L, T), total: T });
  };
  post({ type: 'progress', phase: 'download', loaded: 0, total: heads.reduce((x, y) => x + y, 0) });
  const [locator, clip, vocab, embBytes, locBuf, clipBuf] = await Promise.all([
    fetchJson<LocatorCfg>(m + 'locator.json'), fetchJson<ClipCfg>(m + 'clip.json'), fetchJson<VocabRow[]>(m + 'vocab.json'),
    fetchAsset(m + 'vocab_emb.bin', { prefer: 'cache', onProgress: prog('vocab_emb.bin') }),
    fetchAsset(m + 'locator.onnx', { prefer: 'cache', onProgress: prog('locator.onnx') }),
    fetchAsset(m + 'clip_image.onnx', { prefer: 'cache', onProgress: prog('clip_image.onnx') }),
  ]);
  post({ type: 'progress', phase: 'init' });
  // fp16 little-endian -> Float32Array once.
  const u16 = new Uint16Array(embBytes.buffer.slice(embBytes.byteOffset, embBytes.byteOffset + embBytes.byteLength));
  const emb = f16ArrayToF32(u16);
  let loc = await createSession(locBuf, { input: locator.input, output: locator.output, size: locator.imgsz });
  let cl = await createSession(clipBuf, { input: clip.input, output: clip.output, size: clip.size });

  const runOn = async (which: 'loc' | 'cl', feeds: (s: ort.InferenceSession) => Record<string, ort.Tensor>, outName: string) => {
    const cur = which === 'loc' ? loc : cl;
    try {
      return (await cur.s.run(feeds(cur.s)))[outName];
    } catch (e) {
      if (cur.backend === 'wasm') throw e;
      console.warn('webgpu run failed, retrying on wasm', e);
      const next = await (async () => {
        const s = await ort.InferenceSession.create(cur.buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
        return { s, backend: 'wasm', buf: cur.buf };
      })();
      if (which === 'loc') loc = next; else cl = next;
      return (await next.s.run(feeds(next.s)))[outName];
    }
  };
  const runners: Runners = {
    async locate(data, size) {
      const t = await runOn('loc', () => ({ [locator.input]: new ort.Tensor('float32', data, [1, 3, size, size]) }), locator.output);
      return { data: (t.data ?? await t.getData()) as Float32Array, dims: t.dims };
    },
    async embed(data, size) {
      const x = clip.dtype === 'float16' ? new ort.Tensor('float16', f32ArrayToF16(data) as any, [1, 3, size, size])
        : new ort.Tensor('float32', data, [1, 3, size, size]);
      const t = await runOn('cl', () => ({ [clip.input]: x }), clip.output);
      const d = (t.data ?? await t.getData()) as ArrayLike<number>;
      return d instanceof Float32Array ? d : t.type === 'float16' && d instanceof Uint16Array ? f16ArrayToF32(d) : Float32Array.from(d);
    },
  };
  return { assets: { locator, clip, vocab, emb }, runners, backend: `${loc.backend}/${cl.backend}` };
}

let engine: LensEngine | null = null;
let canvas: OffscreenCanvas | null = null;

async function ensure(base: string, ortBase?: string, threads?: number): Promise<Loaded> {
  if (!loaded) loaded = load(base, ortBase, threads);
  try { return await loaded; } catch (e) { loaded = null; throw e; }
}

let base = '/', ortBase: string | undefined, threads: number | undefined;
self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;
  if (msg.type === 'reset') { engine?.reset(); return; }
  if (msg.type === 'warm') {
    base = msg.base ?? base; ortBase = msg.ortBase; threads = msg.threads;
    try { const L = await ensure(base, ortBase, threads); engine = new LensEngine(L.assets, L.runners); post({ type: 'ready', backend: L.backend }); }
    catch (e: any) { post({ type: 'error', message: String(e?.message ?? e) }); }
    return;
  }
  if (msg.type !== 'frame' || !msg.bitmap) return;
  const bm = msg.bitmap;
  try {
    if (!engine) { const L = await ensure(base, ortBase, threads); engine = new LensEngine(L.assets, L.runners); }
    const w = bm.width, h = bm.height, t0 = performance.now();
    if (!canvas || canvas.width !== w || canvas.height !== h) canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bm, 0, 0);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const tracks = await engine.step({ data: rgba, w, h }, t0);
    post({ type: 'result', r: { tracks, ms: performance.now() - t0, w, h, nBoxes: engine.stats.nBoxes, classified: engine.stats.classified, t: msg.t ?? 0,
      prepMs: engine.stats.prepMs, locMs: engine.stats.locMs, clsMs: engine.stats.clsMs } });
  } catch (e: any) {
    post({ type: 'error', message: String(e?.message ?? e) });
  } finally { bm.close(); }
};
