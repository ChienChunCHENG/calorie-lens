// Minimal service worker: network-first for the app shell (so updates are never stuck behind a stale cache),
// cache fallback when offline. Models are cached by the app itself (Cache Storage, src/assets.ts) and skipped here.
// The pinned onnxruntime-web files from jsDelivr are cached cache-first so inference works offline.
const CACHE = 'calorie-lens-shell-v2';
// GitHub Pages cannot send COOP/COEP headers. Adding them here makes the page cross-origin isolated (after one reload,
// see main.ts), which lets onnxruntime-web use several CPU threads instead of one.
const ISOLATE = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cross-Origin-Resource-Policy': 'same-origin' };
function isolated(r) {
  if (!r || r.status === 0 || r.type === 'opaque') return r;
  const h = new Headers(r.headers);
  for (const [k, v] of Object.entries(ISOLATE)) h.set(k, v);
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
}
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;
  const ortCdn = url.hostname === 'cdn.jsdelivr.net' && url.pathname.includes('/onnxruntime-web@');
  if (!sameOrigin && !ortCdn) return;
  if (sameOrigin && url.pathname.includes('/models/')) return;
  if (ortCdn) {
    e.respondWith(caches.open(CACHE).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const r = await fetch(req);
      if (r.ok) c.put(req, r.clone());
      return r;
    }));
    return;
  }
  e.respondWith((async () => {
    const c = await caches.open(CACHE);
    try {
      const r = await fetch(req);
      if (r.ok) c.put(req, r.clone());
      return isolated(r);
    } catch (err) {
      const hit = (await c.match(req)) || (req.mode === 'navigate' ? await c.match(new URL('./', self.registration.scope).href) : undefined);
      if (hit) return isolated(hit);
      throw err;
    }
  })());
});
