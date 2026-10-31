import * as ort from 'onnxruntime-web';
import { COCO80 } from './coco';
import { Detection, Letterboxer, decodeYolo } from './yolo';

export type Backend = 'webgpu' | 'wasm';

const base = import.meta.env.BASE_URL;
// Dev: files copied to public/ort by scripts/copy-ort.mjs. Prod: Cloudflare Pages caps files at 25 MiB and the
// webgpu wasm is ~29 MB, so prod loads the pinned ORT files from jsDelivr (sends CORP cross-origin, so COEP is fine).
// Override with ?ortbase=<url ending in />.
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
ort.env.wasm.wasmPaths = new URLSearchParams(location.search).get('ortbase') ?? (import.meta.env.PROD ? ORT_CDN : `${base}ort/`);
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const isHtml = (r: Response) => (r.headers.get('content-type') || '').includes('text/html');

export class Detector {
  session!: ort.InferenceSession;
  backend: Backend = 'wasm';
  /** Output class names (deduplicated); classMap[modelClass] indexes into it. */
  labels: string[] = COCO80;
  classMap?: number[];
  confThr = 0.35; // tuned for the YOLOE prompt model (ml/export_openvocab.py)
  private lbx: Letterboxer;
  private inName = 'images';
  private outName = 'output0';

  constructor(public modelUrl: string, public imgsz: number) { this.lbx = new Letterboxer(imgsz); }

  static async modelExists(url: string): Promise<boolean> {
    try {
      const r = await fetch(url, { method: 'HEAD' });
      return r.ok && !isHtml(r); // SPA fallbacks return index.html
    } catch { return false; }
  }

  async init(preferred: Backend): Promise<void> {
    try {
      const r = await fetch(`${base}models/detector.labels.json`);
      if (r.ok && !isHtml(r)) {
        const j = await r.json();
        if (Array.isArray(j) && j.length) {
          // One entry per model class; an open-vocab export repeats a dish id for each of its prompts.
          const names = j.map(String), uniq = [...new Set(names)];
          this.labels = uniq;
          this.classMap = names.map(n => uniq.indexOf(n));
        }
      }
    } catch { /* keep COCO names */ }
    const buf = new Uint8Array(await (await fetch(this.modelUrl)).arrayBuffer());
    const order: Backend[] = preferred === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
    let lastErr: unknown;
    for (const b of order) {
      try {
        this.session = await ort.InferenceSession.create(buf, { executionProviders: [b], graphOptimizationLevel: 'all' });
        this.backend = b;
        this.inName = this.session.inputNames[0];
        this.outName = this.session.outputNames[0];
        return;
      } catch (e) { lastErr = e; console.warn(`backend ${b} failed`, e); }
    }
    throw lastErr;
  }

  /** Grabs the video's current frame at call time (latest frame wins). */
  async detect(video: HTMLVideoElement): Promise<Detection[]> {
    const { data, lb } = this.lbx.run(video);
    const s = this.imgsz;
    const out = await this.session.run({ [this.inName]: new ort.Tensor('float32', data, [1, 3, s, s]) });
    const t = out[this.outName];
    const arr = (t.data ?? await t.getData()) as Float32Array;
    return decodeYolo(arr, t.dims, lb, video.videoWidth, video.videoHeight, this.confThr, 0.45, 50,
      this.classMap ?? this.labels.map((_, i) => i), true);
  }
}
