// Minimal service worker: network-first for the app shell (so updates are never stuck behind a stale cache),
// cache fallback when offline. Models are cached by the app itself (Cache Storage, src/assets.ts) and skipped here.
// The pinned onnxruntime-web files from jsDelivr are cached cache-first so inference works offline.
const CACHE = 'calorie-lens-shell-v1';
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
      return r;
    } catch (err) {
      const hit = (await c.match(req)) || (req.mode === 'navigate' ? await c.match(new URL('./', self.registration.scope).href) : undefined);
      if (hit) return hit;
      throw err;
    }
  })());
});
