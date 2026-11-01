// Model asset loading with Cache Storage: first visit downloads (with progress), later visits are instant/offline.
export const CACHE_NAME = 'calorie-lens-models-v1';
export type Progress = (loaded: number, total: number) => void;

const LEN_HDR = 'x-cl-len';
const isHtml = (r: Response) => (r.headers.get('content-type') || '').includes('text/html');

async function openCache(): Promise<Cache | null> {
  try { return typeof caches !== 'undefined' ? await caches.open(CACHE_NAME) : null; } catch { return null; }
}

async function readWithProgress(r: Response, onChunk: (n: number, total: number) => void): Promise<Uint8Array> {
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body) { const b = new Uint8Array(await r.arrayBuffer()); onChunk(b.length, b.length); return b; }
  const reader = r.body.getReader(), parts: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onChunk(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export interface FetchOpts {
  /** 'cache': use the cached copy when present (big files); 'network': try network first, fall back to cache (small configs). */
  prefer: 'cache' | 'network';
  onProgress?: Progress;
}

/**
 * Fetch a model asset. Big files are cache-first but revalidated cheaply: when online a HEAD request compares
 * Content-Length with the cached copy's recorded length and re-downloads on mismatch.
 */
export async function fetchAsset(url: string, { prefer, onProgress }: FetchOpts): Promise<Uint8Array> {
  const cache = await openCache();
  const cached = cache ? await cache.match(url).catch(() => undefined) : undefined;
  const fromCache = async () => {
    const b = new Uint8Array(await cached!.arrayBuffer()); onProgress?.(b.length, b.length); return b;
  };
  if (cached && prefer === 'cache') {
    let fresh = true;
    try {
      if (typeof navigator === 'undefined' || navigator.onLine !== false) {
        const h = await fetch(url, { method: 'HEAD' });
        const len = h.headers.get('content-length'), have = cached.headers.get(LEN_HDR);
        if (h.ok && !isHtml(h) && len && have && len !== have) fresh = false;
      }
    } catch { /* offline: keep cache */ }
    if (fresh) return fromCache();
  }
  try {
    const r = await fetch(url, prefer === 'network' ? { cache: 'no-cache' } : undefined);
    if (!r.ok || isHtml(r)) throw new Error(`${url}: HTTP ${r.status}`);
    const len = r.headers.get('content-length');
    const bytes = await readWithProgress(r, (n, t) => onProgress?.(n, t));
    if (cache) {
      const headers = new Headers({ 'content-type': r.headers.get('content-type') || 'application/octet-stream' });
      if (len) headers.set(LEN_HDR, len);
      try { await cache.put(url, new Response(bytes as BodyInit, { headers })); } catch { /* quota: still usable this session */ }
    }
    return bytes;
  } catch (e) {
    if (cached) return fromCache();
    throw e;
  }
}

export async function fetchJson<T>(url: string): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await fetchAsset(url, { prefer: 'network' }))) as T;
}
