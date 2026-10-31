import { Detector, Backend } from './detector';
import { IouTracker } from './tracker';
import type { Track } from './tracker';
import { Assignments } from './assign.ts';
import { data, dishIds } from './data';
import { autoDishId, defaultItem, safeEstimate, totalOf, WHOLE_DISHES } from './meal.ts';
import { suppressContained } from './yolo';
import type { Item, ItemEstimate, SugarLevel } from './meal.ts';
import { formatRange } from '../../core/ts/rangeEngine.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video'), canvas = $<HTMLCanvasElement>('overlay');
const hud = $('hud'), msg = $('msg'), startBtn = $<HTMLButtonElement>('start'), backendSel = $<HTMLSelectElement>('backend');
const ctx = canvas.getContext('2d')!;
const q = new URLSearchParams(location.search);
const imgsz = Math.max(32, Math.round((parseInt(q.get('imgsz') || '416', 10) || 416) / 32) * 32);
const modelUrl = q.get('model') || `${import.meta.env.BASE_URL}models/detector.onnx`;
const TARGET_MS = 1000 / Math.min(10, Math.max(1, parseFloat(q.get('hz') || '8') || 8));

const tracker = new IouTracker();
const assignments = new Assignments();
let noNumbers = false;
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
        const wholeCls = new Set(detector.labels.flatMap((l, i) => (WHOLE_DISHES.has(l) ? [i] : [])));
        const dets = suppressContained(await detector.detect(video), wholeCls);
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
const rawLabel = (t: Track) => detector?.labels[t.cls] ?? String(t.cls);
const autoFor = (l: string | undefined): Item | null => { const id = autoDishId(l, data); return id ? defaultItem(data, id) : null; };

function estimateOf(item: Item): ItemEstimate | null { return safeEstimate(data, item); }
const FLAG_TEXT: Record<string, string> = { portion_default: '≈ typical portion', low_confidence: '± wide' };
const SUGAR_ZH: Record<SugarLevel, string> = { full: '全糖', less: '少糖', half: '半糖', light: '微糖', none: '無糖' };

function roundRect(x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

/** Draws a label pill above (or inside, when no room above) the box; lines = [main, sub?]. */
function drawLabel(x: number, y: number, lines: string[], bg: string, alpha: number) {
  const mainFont = '700 15px -apple-system, system-ui, "PingFang TC", sans-serif';
  const subFont = '500 11px -apple-system, system-ui, sans-serif';
  ctx.font = mainFont; const w1 = ctx.measureText(lines[0]).width;
  let w2 = 0;
  if (lines[1]) { ctx.font = subFont; w2 = ctx.measureText(lines[1]).width; }
  const w = Math.min(cssW - 8, Math.max(w1, w2) + 14), h = lines[1] ? 38 : 24;
  const lx = Math.max(4, Math.min(x, cssW - w - 4));
  const top = 56; // keep clear of the HUD pill
  const ly = y - h - 2 < top ? Math.max(y + 2, top) : y - h - 2;
  ctx.globalAlpha = alpha; ctx.fillStyle = bg; roundRect(lx, ly, w, h, 8); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.textBaseline = 'top';
  ctx.font = mainFont; ctx.fillText(lines[0], lx + 7, ly + 4, w - 10);
  if (lines[1]) { ctx.font = subFont; ctx.globalAlpha = alpha * 0.8; ctx.fillText(lines[1], lx + 7, ly + 22, w - 10); }
  ctx.globalAlpha = 1;
}

function itemLines(item: Item): string[] {
  const dish = data.dishes[item.dishId];
  const est = noNumbers ? null : estimateOf(item);
  const main = est ? `${dish.zh} ${formatRange(est.lo, est.hi)} kcal` : dish.zh;
  if (!est) return [main];
  const sub: string[] = [];
  if (dish.kind === 'drink') sub.push(`${SUGAR_ZH[item.sugar]}?`);
  for (const f of est.flags) if (FLAG_TEXT[f]) sub.push(FLAG_TEXT[f]);
  return sub.length ? [main, sub.join(' · ')] : [main];
}

let last = performance.now();
function render(now: number) {
  const dt = now - last; last = now;
  tracker.step(dt);
  const vis = tracker.visible();
  assignments.sync(vis.map(t => ({ id: t.id, box: t.target, label: rawLabel(t) })), now, autoFor);
  ctx.clearRect(0, 0, cssW, cssH);
  const { s, ox, oy } = mapping();
  ctx.lineWidth = 2;
  for (const t of vis) {
    const x = t.x1 * s + ox, y = t.y1 * s + oy, w = (t.x2 - t.x1) * s, h = (t.y2 - t.y1) * s;
    const item = assignments.get(t.id);
    if (item) {
      const hue = (dishIds.indexOf(item.dishId) * 41 + 120) % 360;
      ctx.strokeStyle = `hsl(${hue} 85% 55%)`; ctx.strokeRect(x, y, w, h);
      drawLabel(x, y, itemLines(item), `hsl(${hue} 70% 28%)`, 0.92);
    }
  }
  updateHud(now);
  updateBar(now);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ---- bottom meal bar ----
const totalEl = $('total'), nonumBtn = $<HTMLButtonElement>('nonum'), clearBtn = $<HTMLButtonElement>('clear');
let barAt = 0, barKey = '';
function updateBar(now: number) {
  if (now - barAt < 200) return;
  barAt = now;
  const items = assignments.activeItems();
  const ests = items.map(estimateOf).filter((e): e is ItemEstimate => e !== null);
  let big: string, sub: string;
  if (!items.length) { big = 'Point at your meal'; sub = 'Dishes are recognised automatically'; }
  else if (noNumbers || !ests.length) { big = `${items.length} item${items.length > 1 ? 's' : ''}`; sub = noNumbers ? 'kcal hidden' : ''; }
  else {
    const m = totalOf(ests);
    big = `${formatRange(m.lo, m.hi)} kcal`;
    sub = `Meal total, ${items.length} item${items.length > 1 ? 's' : ''} (median ${Math.round(m.med / 5) * 5})`;
  }
  const key = big + '|' + sub;
  if (key === barKey) return;
  barKey = key;
  totalEl.children[0].textContent = big; totalEl.children[1].textContent = sub;
}
nonumBtn.addEventListener('click', () => {
  noNumbers = !noNumbers; nonumBtn.classList.toggle('on', noNumbers); nonumBtn.textContent = noNumbers ? 'Show kcal' : 'Hide kcal'; barAt = 0;
});
clearBtn.addEventListener('click', () => { assignments.clear(); barAt = 0; });
$('hudpill').addEventListener('click', () => { $('hudwrap').classList.toggle('open'); });

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
