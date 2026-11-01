// Module worker: decode -> locate -> classify -> kcal mixture. Models are fetched once and cached (assets.ts).
import * as ort from 'onnxruntime-web';
import { decodeFile } from './decode.ts';
import type { Decoded } from './decode.ts';
import { fetchAsset, fetchJson } from './assets.ts';
import { analyze } from './pipeline.ts';
import type { Assets, ClipCfg, LocatorCfg, Runners } from './pipeline.ts';
import type { VocabRow } from './classify.ts';
import { f16ArrayToF32, f32ArrayToF16 } from './f16.ts';

export interface InMsg { type: 'analyze'; id: number; base: string; ortBase?: string; file?: Blob; decoded?: Decoded }
export type OutMsg =
  | { type: 'progress'; id: number; phase: 'decode' | 'download' | 'init' | 'locate' | 'classify'; loaded?: number; total?: number }
  | { type: 'result'; id: number; result: Awaited<ReturnType<typeof analyze>>; thumb: Blob; w: number; h: number; backend: string }
  | { type: 'error'; id: number; message: string; code?: string };

const post = (m: OutMsg) => (self as unknown as Worker).postMessage(m);

let ortReady = false;
function setupOrt(base: string, ortBase?: string) {
  if (ortReady) return;
  const cdn = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  // Prod: pinned ORT wasm from jsDelivr (the webgpu wasm is ~29 MB; keeps the Pages artifact small). Dev: public/ort.
  ort.env.wasm.wasmPaths = ortBase ?? (import.meta.env.PROD ? cdn : `${base}ort/`);
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ortReady = true;
}

interface Loaded { assets: Assets; runners: Runners; backend: string }
let loaded: Promise<Loaded> | null = null;
let curId = 0;

async function createSession(buf: Uint8Array): Promise<{ s: ort.InferenceSession; backend: string; buf: Uint8Array }> {
  // wasm only for now: Safari WebGPU can run but return wrong numbers, and there is no on-device check yet.
  const order = ['wasm'];
  let err: unknown;
  for (const ep of order) {
    try {
      const s = await ort.InferenceSession.create(buf, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      return { s, backend: ep, buf };
    } catch (e) { err = e; console.warn('ORT backend failed', ep, e); }
  }
  throw err;
}

async function load(base: string, ortBase: string | undefined, id: number): Promise<Loaded> {
  setupOrt(base, ortBase);
  const m = `${base}models/`;
  const sizes = new Map<string, [number, number]>();
  const prog = (name: string) => (l: number, t: number) => {
    sizes.set(name, [l, t || l]);
    let L = 0, T = 0; for (const [a, b] of sizes.values()) { L += a; T += b; }
    post({ type: "progress", id: curId, phase: "download", loaded: L, total: T });
  };
  post({ type: 'progress', id, phase: 'download', loaded: 0, total: 0 });
  const [locator, clip, vocab, embBytes, locBuf, clipBuf] = await Promise.all([
    fetchJson<LocatorCfg>(m + 'locator.json'), fetchJson<ClipCfg>(m + 'clip.json'), fetchJson<VocabRow[]>(m + 'vocab.json'),
    fetchAsset(m + 'vocab_emb.bin', { prefer: 'cache', onProgress: prog('emb') }),
    fetchAsset(m + 'locator.onnx', { prefer: 'cache', onProgress: prog('loc') }),
    fetchAsset(m + 'clip_image.onnx', { prefer: 'cache', onProgress: prog('clip') }),
  ]);
  post({ type: 'progress', id, phase: 'init' });
  // fp16 little-endian -> Float32Array once.
  const u16 = new Uint16Array(embBytes.buffer.slice(embBytes.byteOffset, embBytes.byteOffset + embBytes.byteLength));
  const emb = f16ArrayToF32(u16);
  let loc = await createSession(locBuf), cl = await createSession(clipBuf);

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

self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;
  if (msg.type !== 'analyze') return;
  const { id } = msg; curId = id;
  try {
    post({ type: 'progress', id, phase: 'decode' });
    const dec = msg.decoded ?? await decodeFile(msg.file!);
    if (!loaded) loaded = load(msg.base, msg.ortBase, id);
    let L: Loaded;
    try { L = await loaded; } catch (e) { loaded = null; throw e; }
    const result = await analyze({ data: dec.rgba, w: dec.w, h: dec.h }, L.assets, L.runners,
      s => post({ type: 'progress', id, phase: s }));
    post({ type: 'result', id, result, thumb: dec.thumb, w: dec.w, h: dec.h, backend: L.backend });
  } catch (e: any) {
    post({ type: 'error', id, message: String(e?.message ?? e), code: e?.message === 'NO_CANVAS' ? 'NO_CANVAS' : undefined });
  }
};
