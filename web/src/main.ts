import { Detector, Backend } from './detector';
import { IouTracker } from './tracker';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video'), canvas = $<HTMLCanvasElement>('overlay');
const hud = $('hud'), msg = $('msg'), startBtn = $<HTMLButtonElement>('start'), backendSel = $<HTMLSelectElement>('backend');
const ctx = canvas.getContext('2d')!;
const q = new URLSearchParams(location.search);
const imgsz = Math.max(32, Math.round((parseInt(q.get('imgsz') || '320', 10) || 320) / 32) * 32);
const modelUrl = q.get('model') || `${import.meta.env.BASE_URL}models/detector.onnx`;
const TARGET_MS = 1000 / Math.min(10, Math.max(1, parseFloat(q.get('hz') || '8') || 8));

const tracker = new IouTracker();
let detector: Detector | null = null;
let modelStatus = 'not loaded';
let running = false;
const infMs: number[] = [];
const doneTimes: number[] = [];
let startTime = 0;

function showMsg(t: string) { msg.textContent = t; }

// ---- canvas sizing / mapping (video uses object-fit: cover) ----
let cssW = 0, cssH = 0;
function resizeCanvas() {
  const r = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cssW = r.width; cssH = r.height;
  canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
new ResizeObserver(resizeCanvas).observe(video);
window.addEventListener('orientationchange', resizeCanvas);
video.addEventListener('loadedmetadata', resizeCanvas);

function mapping() {
  const vw = video.videoWidth || 1, vh = video.videoHeight || 1;
  const s = Math.max(cssW / vw, cssH / vh);
  return { s, ox: (cssW - vw * s) / 2, oy: (cssH - vh * s) / 2 };
}

// ---- camera ----
async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showMsg('Camera needs a secure context (HTTPS or localhost). Open this page via the https:// tunnel URL.');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    startBtn.style.display = 'none';
    showMsg('');
    resizeCanvas();
    startTime = performance.now();
    await loadModel(backendSel.value as Backend);
    running = true;
    void inferLoop();
  } catch (e) {
    const err = e as DOMException;
    if (err.name === 'NotAllowedError' || err.name === 'SecurityError')
      showMsg('Camera permission denied. On iPhone: Settings > Safari > Camera, allow, then reload.');
    else if (err.name === 'NotFoundError') showMsg('No camera found on this device.');
    else showMsg(`Camera error: ${err.name}: ${err.message}`);
  }
}
startBtn.addEventListener('click', () => { void startCamera(); });

// ---- model ----
async function loadModel(b: Backend) {
  running = false;
  loopGen++;
  infMs.length = 0; doneTimes.length = 0;
  tracker.tracks = [];
  if (!(await Detector.modelExists(modelUrl))) { detector = null; modelStatus = 'camera-only (no model)'; return; }
  modelStatus = 'loading model...';
  try {
    const d = new Detector(modelUrl, imgsz);
    await d.init(b);
    detector = d; modelStatus = `ok (imgsz ${imgsz})`;
  } catch (e) {
    detector = null; modelStatus = `model error: ${(e as Error).message}`;
  }
}
backendSel.addEventListener('change', async () => {
  if (!video.srcObject) return;
  await loadModel(backendSel.value as Backend);
  running = true; void inferLoop();
});

// ---- inference loop: latest-frame-wins, frames are never queued ----
let loopGen = 0;
async function inferLoop() {
  const gen = ++loopGen;
  while (running && gen === loopGen && detector) {
    const t0 = performance.now();
    if (video.readyState >= 2 && video.videoWidth) {
      try {
        const dets = await detector.detect(video);
        const t1 = performance.now();
        infMs.push(t1 - t0); if (infMs.length > 20) infMs.shift();
        doneTimes.push(t1);
        if (gen === loopGen) tracker.update(dets);
      } catch (e) {
        modelStatus = `inference error: ${(e as Error).message}`; running = false; break;
      }
    }
    const wait = TARGET_MS - (performance.now() - t0);
    await new Promise(r => setTimeout(r, Math.max(0, wait)));
  }
}

// ---- render loop at rAF rate, from smoothed tracks ----
let last = performance.now();
function render(now: number) {
  const dt = now - last; last = now;
  tracker.step(dt);
  ctx.clearRect(0, 0, cssW, cssH);
  const { s, ox, oy } = mapping();
  ctx.lineWidth = 2; ctx.font = '600 14px -apple-system, system-ui, sans-serif'; ctx.textBaseline = 'top';
  const labels = detector?.labels;
  for (const t of tracker.visible()) {
    const x = t.x1 * s + ox, y = t.y1 * s + oy, w = (t.x2 - t.x1) * s, h = (t.y2 - t.y1) * s;
    const hue = (t.cls * 47) % 360;
    ctx.strokeStyle = `hsl(${hue} 90% 55%)`;
    ctx.strokeRect(x, y, w, h);
    const text = `${labels?.[t.cls] ?? t.cls} ${(t.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width + 8;
    const ly = y - 20 < 0 ? y : y - 20;
    ctx.fillStyle = `hsl(${hue} 90% 35%)`;
    ctx.fillRect(x, ly, tw, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x + 4, ly + 3);
  }
  updateHud(now);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ---- HUD ----
function median(a: number[]) { if (!a.length) return NaN; const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; }
let hudAt = 0;
function updateHud(now: number) {
  if (now - hudAt < 250) return;
  hudAt = now;
  while (doneTimes.length && now - doneTimes[0] > 3000) doneTimes.shift();
  const hz = doneTimes.length > 1 ? (doneTimes.length - 1) / ((doneTimes[doneTimes.length - 1] - doneTimes[0]) / 1000) : 0;
  const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
  const heap = mem ? `${(mem.usedJSHeapSize / 1048576).toFixed(0)} MB` : 'n/a';
  const el = startTime ? Math.floor((now - startTime) / 1000) : 0;
  const mm = String(Math.floor(el / 60)).padStart(2, '0'), ss = String(el % 60).padStart(2, '0');
  const m = median(infMs);
  hud.textContent = `backend: ${detector ? detector.backend : '-'}  model: ${modelStatus}\n` +
    `inference: ${isNaN(m) ? '-' : m.toFixed(1)} ms (median/20)  rate: ${hz.toFixed(1)} Hz  tracks: ${tracker.visible().length}\n` +
    `heap: ${heap}  time: ${mm}:${ss}  isolated: ${self.crossOriginIsolated}`;
}
