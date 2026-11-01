// Calorie Lens: one button, live camera, food chips with calories. Everything runs on-device (worker.ts).
import './style.css';
import { fmtRange, maybeSecond, round10 } from './format.ts';
import { renderMarkdown } from './md.ts';
import { sumRange } from './mixture.ts';
import type { InMsg, OutMsg, WorkerResult } from './worker.ts';
import type { LensTrack } from './pipeline.ts';

const BASE = import.meta.env.BASE_URL;
const MAX_EDGE = 960, TAU_MS = 90, HINT_MS = 3000, SUMMARY_MS = 3000;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

document.getElementById('app')!.innerHTML = `
<main id="start" class="start">
  <button id="info" class="info" aria-label="About, attribution and disclaimer">i</button>
  <div class="hero">
    <svg class="mark" viewBox="0 0 64 64" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M8 22V12a4 4 0 0 1 4-4h10M42 8h10a4 4 0 0 1 4 4v10M56 42v10a4 4 0 0 1-4 4H42M22 56H12a4 4 0 0 1-4-4V42"/></g><circle cx="32" cy="32" r="8" fill="currentColor"/></svg>
    <h1>Calorie Lens</h1>
    <button id="go" class="go">Turn on camera</button>
    <p class="sub">Runs on your phone. Photos never leave it.</p>
  </div>
</main>
<section id="live" class="live" hidden aria-label="Live camera">
  <video id="vid" playsinline muted autoplay disablepictureinpicture></video>
  <div id="scan" class="scan" hidden aria-hidden="true"></div>
  <div id="ov" class="ov"></div>
  <div id="load" class="load" hidden role="status"><span id="loadtxt"></span><i><b id="loadbar"></b></i></div>
  <div id="hint" class="hint" hidden>Point at your food</div>
  <div class="bar">
    <div id="pill" class="pill" hidden><span id="pilltxt"></span><small id="pillrange"></small></div>
    <button id="close" class="close" aria-label="Turn off camera">
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M5 5l14 14M19 5L5 19" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/></svg>
    </button>
  </div>
  <div id="err" class="errbox" hidden role="alert"><p id="errtxt"></p><button id="retry" class="go small">Try again</button></div>
  <div id="sum" class="sr" aria-live="polite"></div>
</section>
<div id="sheet" class="sheet" hidden>
  <div class="sheet-card" role="dialog" aria-modal="true" aria-labelledby="sheet-h" tabindex="-1">
    <div class="sheet-head"><h2 id="sheet-h">About Calorie Lens</h2><button id="sheetx" class="icon" aria-label="Close">
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M5 5l14 14M19 5L5 19" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg></button></div>
    <p class="disc"><strong>Estimates only, not medical advice.</strong> Calories are rough ranges from a photo and can be far off.</p>
    <div id="attr" class="md"></div>
  </div>
</div>
<div id="probe" class="probe" aria-hidden="true"></div>`;

const vid = $<HTMLVideoElement>('vid'), ov = $('ov'), scanEl = $('scan'), loadEl = $('load'), hintEl = $('hint');

/* ---------- worker + model loading ---------- */
let worker: Worker | null = null;
let ready = false, busy = false, running = false, loadError = false;
let dl = { loaded: 0, total: 0, phase: 'download' as 'download' | 'init' };

function getWorker(): Worker {
  if (worker) return worker;
  const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  w.onmessage = (ev: MessageEvent<OutMsg>) => onWorker(ev.data);
  w.onerror = () => { loadError = true; busy = false; updateLoad(); };
  return (worker = w);
}
function warm(): void {
  loadError = false;
  const qs = new URLSearchParams(location.search);
  getWorker().postMessage({ type: 'warm', base: BASE, ortBase: qs.get('ortbase') ?? undefined, threads: Number(qs.get('threads')) || undefined } satisfies InMsg);
}
function onWorker(m: OutMsg): void {
  if (m.type === 'progress') { dl = { loaded: m.loaded ?? dl.loaded, total: m.total ?? dl.total, phase: m.phase }; updateLoad(); }
  else if (m.type === 'ready') { ready = true; noneSince = performance.now(); updateLoad(); }
  else if (m.type === 'error') { busy = false; if (!ready) { loadError = true; updateLoad(); } }
  else if (m.type === 'result') { busy = false; if (running) onResult(m.r); }
}
const mb = (n: number) => Math.round(n / 1048576);
function updateLoad(): void {
  const show = running && !ready;
  loadEl.hidden = !show;
  if (!show) return;
  const total = Math.max(dl.total, 1), loaded = Math.min(dl.loaded, total);
  $('loadtxt').textContent = loadError ? 'Could not load the food models. Check your connection and try again.'
    : dl.phase === 'init' || (dl.total > 0 && loaded >= total) ? 'Starting food models…'
    : dl.total > 0 ? `Loading food models · ${mb(loaded)} / ${mb(total)} MB` : 'Loading food models…';
  $('loadbar').style.width = loadError ? '0' : `${dl.total > 0 ? Math.min(100, (loaded / total) * 100) : 0}%`;
  if (loadError) showErr('The food models could not be loaded. Check your connection and try again.');
}

/* ---------- camera ---------- */
let stream: MediaStream | null = null;
function showErr(msg: string | null): void { $('err').hidden = !msg; if (msg) $('errtxt').textContent = msg; }

async function startCamera(): Promise<void> {
  showErr(null);
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showErr('The camera needs a secure (https) connection. Open this page over https and try again.'); return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e: any) {
    const n = e?.name;
    showErr(n === 'NotAllowedError' || n === 'SecurityError' ? 'Camera access is blocked. Allow the camera for this site in your browser settings, then try again.'
      : n === 'NotFoundError' || n === 'OverconstrainedError' ? 'No camera was found on this device.'
      : n === 'NotReadableError' ? 'The camera is in use by another app. Close it and try again.'
      : 'The camera could not be started. Try again.');
    return;
  }
  if (!running) { stopStream(); return; }
  vid.srcObject = stream;
  try { await vid.play(); } catch { /* muted + playsinline autoplay is allowed; ignore */ }
  noneSince = performance.now(); camAt = performance.now();
}
function stopStream(): void { stream?.getTracks().forEach(t => t.stop()); stream = null; vid.srcObject = null; }

function openLive(): void {
  running = true; busy = false;
  $('start').hidden = true; $('live').hidden = false;
  resetLens(); updateLoad();
  if (!worker || loadError) warm();
  void startCamera();
}
function closeLive(): void {
  running = false;
  stopStream(); clearViews(); worker?.postMessage({ type: 'reset' } satisfies InMsg);
  $('live').hidden = true; $('start').hidden = false; showErr(null);
  $<HTMLButtonElement>('go').focus();
}

/* ---------- frame grabbing: latest frame wins, never queued ---------- */
let grabCanvas: HTMLCanvasElement | null = null;
async function grab(): Promise<ImageBitmap | null> {
  const vw = vid.videoWidth, vh = vid.videoHeight;
  const s = Math.min(1, MAX_EDGE / Math.max(vw, vh)), w = Math.max(1, Math.round(vw * s)), h = Math.max(1, Math.round(vh * s));
  try { return await createImageBitmap(vid, { resizeWidth: w, resizeHeight: h, resizeQuality: 'medium' }); } catch { /* older Safari */ }
  grabCanvas ??= document.createElement('canvas');
  grabCanvas.width = w; grabCanvas.height = h;
  grabCanvas.getContext('2d')!.drawImage(vid, 0, 0, w, h);
  try { return await createImageBitmap(grabCanvas); } catch { return null; }
}
async function pump(): Promise<void> {
  if (!running || !ready || busy || document.hidden || vid.readyState < 2 || !vid.videoWidth) return;
  busy = true;
  const bm = await grab();
  if (!bm || !running) { busy = false; bm?.close(); return; }
  firstSent ??= performance.now();
  getWorker().postMessage({ type: 'frame', bitmap: bm, t: performance.now() } satisfies InMsg, [bm]);
}

/* ---------- tracks, drawing ---------- */
type B4 = [number, number, number, number];
interface View {
  id: number; t: LensTrack; cur: B4; tgt: B4; since: number; el: HTMLDivElement; chip: HTMLButtonElement;
  cx: number; cy: number; cw: number; ch: number; key: string; placed: boolean; gone: boolean; b: R;
}
interface R { x: number; y: number; w: number; h: number }
const views = new Map<number, View>();
let expanded: number | null = null;
let noneSince = performance.now(), camAt = performance.now(), firstSent: number | null = null;
const stamps: number[] = [];
const lens = { frames: 0, lastMs: 0, fps: 0, tracks: [] as unknown[], firstLabelMs: null as number | null, tapToLabelMs: null as number | null,
  ready: false, nBoxes: 0, prepMs: 0, locMs: 0, clsMs: 0, changes: [] as { id: number; from: string; to: string; t: number }[] };
(window as unknown as { __clLens: typeof lens }).__clLens = lens;

function resetLens(): void {
  lens.frames = 0; lens.lastMs = 0; lens.fps = 0; lens.tracks = []; lens.firstLabelMs = null; lens.tapToLabelMs = null; lens.changes = [];
  stamps.length = 0; firstSent = null; camAt = performance.now(); noneSince = performance.now();
}

function fillChip(v: View, open: boolean): void {
  const c = v.chip;
  c.replaceChildren();
  const l1 = document.createElement('span'); l1.className = 'l1';
  const nm = document.createElement('b'); nm.textContent = v.t.name;
  l1.append(nm, ` · ~${round10(v.t.p50)} kcal`);
  c.append(l1);
  if (open) {
    const l2 = document.createElement('span'); l2.className = 'l2'; l2.textContent = `Range ${fmtRange(v.t)} kcal`; c.append(l2);
    const s = maybeSecond(v.t.cands);
    if (s) { const l3 = document.createElement('span'); l3.className = 'l2'; l3.textContent = `maybe ${s.name}`; c.append(l3); }
  }
  c.setAttribute('aria-expanded', String(open));
  c.classList.toggle('open', open);
  v.cw = c.offsetWidth; v.ch = c.offsetHeight;
}
const chipKey = (v: View, open: boolean) =>
  `${v.t.name}|${round10(v.t.p50)}|${open ? `${fmtRange(v.t)}|${maybeSecond(v.t.cands)?.name ?? ''}` : ''}`;

function makeView(t: LensTrack, now: number): View {
  const el = document.createElement('div'); el.className = 'br';
  for (const k of ['tl', 'tr', 'bl', 'rb']) { const s = document.createElement('i'); s.className = `c ${k}`; el.append(s); }
  const chip = document.createElement('button'); chip.className = 'chip'; chip.type = 'button';
  const v: View = { id: t.id, t, cur: [...t.box], tgt: [...t.box], since: now, el, chip, cx: 0, cy: 0, cw: 0, ch: 0, key: '', placed: false, gone: false,
    b: { x: 0, y: 0, w: 0, h: 0 } };
  chip.addEventListener('click', e => { e.stopPropagation(); expanded = expanded === v.id ? null : v.id; });
  ov.append(el, chip);
  requestAnimationFrame(() => { el.classList.add('in'); chip.classList.add('in'); });
  return v;
}
function clearViews(): void {
  for (const v of views.values()) { v.el.remove(); v.chip.remove(); }
  views.clear(); expanded = null; lens.tracks = [];
}

function onResult(r: WorkerResult): void {
  const now = performance.now();
  lens.frames++; lens.lastMs = Math.round(r.ms); lens.nBoxes = r.nBoxes; lens.prepMs = Math.round(r.prepMs); lens.locMs = Math.round(r.locMs); lens.clsMs = Math.round(r.clsMs); lens.ready = true;
  stamps.push(now); while (stamps.length > 1 && now - stamps[0] > 6000) stamps.shift();
  lens.fps = stamps.length > 1 ? Math.round(((stamps.length - 1) / ((now - stamps[0]) / 1000)) * 100) / 100 : 0;
  const seen = new Set<number>();
  for (const t of r.tracks) {
    seen.add(t.id);
    let v = views.get(t.id);
    if (!v) { v = makeView(t, now); views.set(t.id, v); }
    else if (v.t.name !== t.name) lens.changes.push({ id: t.id, from: v.t.name, to: t.name, t: Math.round(now - camAt) });
    v.t = t; v.tgt = [...t.box];
  }
  for (const v of views.values()) if (!seen.has(v.id) && !v.gone) {
    v.gone = true; v.el.classList.remove('in'); v.chip.classList.remove('in');
    setTimeout(() => { v.el.remove(); v.chip.remove(); views.delete(v.id); }, 260);
    if (expanded === v.id) expanded = null;
  }
  if (r.tracks.length) {
    noneSince = now;
    if (lens.firstLabelMs === null && firstSent !== null) { lens.firstLabelMs = Math.round(now - firstSent); lens.tapToLabelMs = Math.round(now - camAt); }
  }
  lens.tracks = r.tracks.map(t => ({ id: t.id, name: t.name, p50: t.p50, p10: t.p10, p90: t.p90, box: t.box.map(x => Math.round(x * 1000) / 1000),
    since: Math.round((views.get(t.id)?.since ?? now) - camAt) }));
}

/* safe area + layout */
const probe = $('probe');
function insets() {
  const cs = getComputedStyle(probe);
  return { t: parseFloat(cs.paddingTop) || 0, r: parseFloat(cs.paddingRight) || 0, b: parseFloat(cs.paddingBottom) || 0, l: parseFloat(cs.paddingLeft) || 0 };
}
const hit = (a: R, b: R, pad = 4) => a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;

let lastFrame = performance.now(), lastPill = 0, lastSum = 0, lastSumText = '';
function frame(now: number): void {
  requestAnimationFrame(frame);
  if (!running) return;
  void pump();
  const dt = Math.min(100, now - lastFrame); lastFrame = now;
  const a = 1 - Math.exp(-dt / TAU_MS);
  const W = ov.clientWidth, H = ov.clientHeight, vw = vid.videoWidth || 16, vh = vid.videoHeight || 9;
  const sc = Math.max(W / vw, H / vh), ox = (W - vw * sc) / 2, oy = (H - vh * sc) / 2;
  const ins = insets();
  const safe = { l: 10 + ins.l, r: W - 10 - ins.r, t: 12 + ins.t + (loadEl.hidden ? 0 : 40), b: H - ins.b - 96 };

  const list = [...views.values()].filter(v => !v.gone);
  for (const v of list) {
    for (let i = 0; i < 4; i++) v.cur[i] += (v.tgt[i] - v.cur[i]) * a;
    const x1 = v.cur[0] * vw * sc + ox, y1 = v.cur[1] * vh * sc + oy, x2 = v.cur[2] * vw * sc + ox, y2 = v.cur[3] * vh * sc + oy;
    const bx = Math.max(2, x1), by = Math.max(2, y1), bw = Math.max(24, Math.min(W - 2, x2) - bx), bh = Math.max(24, Math.min(H - 2, y2) - by);
    v.el.style.transform = `translate(${bx}px,${by}px)`; v.el.style.width = `${bw}px`; v.el.style.height = `${bh}px`;
    const open = expanded === v.id, key = chipKey(v, open);
    if (key !== v.key) { v.key = key; fillChip(v, open); }
    v.b = { x: bx, y: by, w: bw, h: bh };
  }
  // chips: the expanded one first (keeps its place), then top to bottom; the first free candidate position wins
  const order = [...list].sort((p, q) => (expanded === q.id ? 1 : 0) - (expanded === p.id ? 1 : 0) || p.b.y - q.b.y);
  const placed: R[] = [];
  for (const v of order) {
    const b = v.b, cw = Math.min(v.cw || 160, safe.r - safe.l), chh = v.ch || 44, g = 6;
    const clampR = (x: number, y: number): R => ({ x: Math.min(Math.max(x, safe.l), safe.r - cw), y: Math.min(Math.max(y, safe.t), safe.b - chh), w: cw, h: chh });
    const cands = [clampR(b.x, b.y - chh - g), clampR(b.x + g, b.y + g), clampR(b.x, b.y + b.h + g), clampR(b.x + g, b.y + b.h - chh - g)];
    let pick = cands.find(c => placed.every(p => !hit(c, p)));
    if (!pick) { // stack: sweep down (then up) from the inside-top candidate until a free slot is found
      const base = cands[1], step = chh + 4;
      for (let k = 1; k < 24 && !pick; k++) for (const dy of [k * step, -k * step]) {
        const c = clampR(base.x, base.y + dy);
        if (c.y === base.y + dy && placed.every(p => !hit(c, p))) { pick = c; break; }
      }
      pick ??= base;
    }
    placed.push(pick);
    if (!v.placed) { v.cx = pick.x; v.cy = pick.y; v.placed = true; }
    else { v.cx += (pick.x - v.cx) * a; v.cy += (pick.y - v.cy) * a; }
    v.chip.style.transform = `translate(${v.cx}px,${v.cy}px)`;
    v.chip.style.maxWidth = `${safe.r - safe.l}px`;
  }

  // scan effect, hint, pill, aria summary
  const n = list.length, errHidden = $('err').hidden;
  scanEl.hidden = !(ready && n === 0 && errHidden);
  hintEl.hidden = !(ready && n === 0 && errHidden && now - noneSince > HINT_MS);
  if (now - lastPill > 400) { lastPill = now; updatePill(list, now); }
}

function updatePill(list: View[], now: number): void {
  const pill = $('pill');
  if (!list.length) { pill.hidden = true; setSummary('', now); return; }
  const s = sumRange(list.map(v => ({ p10: v.t.p10, p50: v.t.p50, p90: v.t.p90, q: v.t.q })), 'lens')!;
  pill.hidden = false;
  $('pilltxt').textContent = `On screen · ~${round10(s.p50)} kcal`;
  $('pillrange').textContent = list.length > 1 ? fmtRange(s) : '';
  setSummary(`${list.length} ${list.length === 1 ? 'food' : 'foods'}, about ${round10(s.p50)} kcal`, now);
}
function setSummary(text: string, now: number): void {
  if (text === lastSumText || now - lastSum < SUMMARY_MS) return;
  lastSum = now; lastSumText = text; $('sum').textContent = text;
}

/* ---------- events ---------- */
ov.addEventListener('click', () => { expanded = null; });
$('go').addEventListener('click', openLive);
$('close').addEventListener('click', closeLive);
$('retry').addEventListener('click', () => { if (loadError) warm(); void startCamera(); });
document.addEventListener('visibilitychange', () => { if (!document.hidden && running) void vid.play().catch(() => {}); });

let attrLoaded = false, sheetFrom: HTMLElement | null = null;
function openSheet(): void {
  sheetFrom = document.activeElement as HTMLElement;
  $('sheet').hidden = false; $('sheetx').focus();
  if (!attrLoaded) {
    attrLoaded = true;
    fetch(`${BASE}models/ATTRIBUTION.md`).then(async r => {
      const t = await r.text();
      if (!r.ok || /^\s*<!doctype|^\s*<html/i.test(t)) throw new Error('missing');
      $('attr').innerHTML = renderMarkdown(t);
    }).catch(() => { attrLoaded = false; $('attr').textContent = 'Attribution could not be loaded.'; });
  }
}
function closeSheet(): void { $('sheet').hidden = true; sheetFrom?.focus(); }
$('info').addEventListener('click', openSheet);
$('sheetx').addEventListener('click', closeSheet);
$('sheet').addEventListener('click', e => { if (e.target === $('sheet')) closeSheet(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('sheet').hidden) closeSheet(); });

requestAnimationFrame(frame);
// First render done: start fetching / initialising the models so they are ready (or nearly) when the user taps.
setTimeout(warm, 0);
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register(`${BASE}sw.js`).catch(() => { /* optional */ });
}
