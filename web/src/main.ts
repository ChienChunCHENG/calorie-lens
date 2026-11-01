import './style.css';
import { decodeFile } from './decode.ts';
import { deleteMeal, exportBackup, getAllMeals, importBackup, putMeal, storageOk } from './db.ts';
import { displayName, fmtKcal, fmtRange, fmtTime, round10 } from './format.ts';
import { dayKey, dayTotal, mealTotal, mealsOfDay, parseDayKey, removeItem, shiftDay, weekMedians } from './log.ts';
import type { LogItem, Meal } from './log.ts';
import type { Analysis } from './pipeline.ts';
import type { InMsg, OutMsg } from './worker.ts';

const BASE = import.meta.env.BASE_URL;
const DISCLAIMER = 'Estimates only, not medical advice. Names may be wrong about half the time; ranges account for that.';

// ---------- state ----------
type Status =
  | { kind: 'idle' }
  | { kind: 'progress'; phase: string; loaded?: number; total?: number }
  | { kind: 'error'; message: string }
  | { kind: 'none'; photo: string; info: string }
  | { kind: 'done'; photo: string; meal: Meal; backend: string };

let meals: Meal[] = [];
let today = dayKey(Date.now());
let day = today;
let status: Status = { kind: 'idle' };
let storageErr: string | null = null;
let busy = false;
let sheetOpen = false;
let sheetNote = '';

const thumbUrls = new Map<string, string>();
const thumbUrl = (m: Meal) => {
  if (!m.thumb) return '';
  let u = thumbUrls.get(m.id);
  if (!u) { u = URL.createObjectURL(m.thumb); thumbUrls.set(m.id, u); }
  return u;
};

// ---------- tiny DOM helper ----------
type Child = Node | string | null | false | undefined;
function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, any> = {}, ...kids: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'class') el.className = v;
    else if (k === 'style') el.setAttribute('style', v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids) if (c) el.append(c as Node | string);
  return el;
}

// ---------- worker ----------
let worker: Worker | null = null;
let seq = 0;
let evalSeq = 0;
let pending: { id: number; resolve(m: Extract<OutMsg, { type: 'result' }>): void; reject(e: Error & { code?: string }): void } | null = null;

function getWorker(): Worker {
  if (worker) return worker;
  const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  w.onmessage = (ev: MessageEvent<OutMsg>) => {
    const m = ev.data;
    if (!pending || m.id !== pending.id) return;
    if (m.type === 'progress') { status = { kind: 'progress', phase: m.phase, loaded: m.loaded, total: m.total }; renderStatus(); }
    else if (m.type === 'result') { const p = pending; pending = null; p.resolve(m); }
    else { const p = pending; pending = null; p.reject(Object.assign(new Error(m.message), { code: m.code })); }
  };
  w.onerror = (ev) => {
    const p = pending; pending = null; worker = null;
    p?.reject(new Error(ev.message || 'worker crashed'));
  };
  worker = w;
  return w;
}

function runWorker(payload: Omit<InMsg, 'type' | 'id' | 'base'>): Promise<Extract<OutMsg, { type: 'result' }>> {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending = { id, resolve, reject };
    const ortBase = new URLSearchParams(location.search).get('ortbase') ?? undefined;
    const msg: InMsg = { type: 'analyze', id, base: BASE, ortBase, ...payload };
    const transfer: Transferable[] = payload.decoded ? [payload.decoded.rgba.buffer] : [];
    getWorker().postMessage(msg, transfer);
  });
}

// ---------- actions ----------
async function snap(file: File | undefined) {
  if (!file || busy) return;
  busy = true;
  const photo = URL.createObjectURL(file);
  const t0 = performance.now();
  status = { kind: 'progress', phase: 'decode' };
  render();
  try {
    let res;
    try { res = await runWorker({ file }); }
    catch (e: any) {
      if (e?.code !== 'NO_CANVAS') throw e;
      status = { kind: 'progress', phase: 'decode' }; renderStatus();
      res = await runWorker({ decoded: await decodeFile(file) }); // older iOS: decode on the main thread
    }
    const a: Analysis = res.result;
    // Read by the offline evaluation harness (ml/snap/eval_web.py); harmless in normal use.
    (window as any).__clLast = { n: ++evalSeq, ms: Math.round(performance.now() - t0), backend: res.backend, result: a };
    if (a.items.length === 0) {
      status = { kind: 'none', photo, info: `boxes ${a.nBoxes}, empty/non-food ${a.skipped.negative}, unsure ${a.skipped.low_food}` };
    } else {
      const meal: Meal = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, ts: Date.now(), thumb: res.thumb, items: a.items };
      try { await putMeal(meal); storageErr = null; } catch (e: any) { storageErr = String(e?.message ?? e); }
      meals.push(meal);
      today = dayKey(Date.now()); day = today;
      status = { kind: 'done', photo, meal, backend: res.backend };
    }
  } catch (e: any) {
    (window as any).__clLast = { n: ++evalSeq, error: String(e?.message ?? e) };
    status = { kind: 'error', message: String(e?.message ?? e) };
  } finally { busy = false; render(); }
}

async function removeMeal(id: string) {
  if (!confirm('刪除這一餐？ Delete this meal?')) return;
  try { await deleteMeal(id); } catch (e: any) { storageErr = String(e?.message ?? e); }
  meals = meals.filter(m => m.id !== id);
  const u = thumbUrls.get(id); if (u) { URL.revokeObjectURL(u); thumbUrls.delete(id); }
  if (status.kind === 'done' && status.meal.id === id) status = { kind: 'idle' };
  render();
}

async function removeOneItem(m: Meal, itemId: string) {
  const next = removeItem(m, itemId);
  if (!next) return removeMeal(m.id);
  try { await putMeal(next); } catch (e: any) { storageErr = String(e?.message ?? e); }
  meals = meals.map(x => (x.id === m.id ? next : x));
  if (status.kind === 'done' && status.meal.id === m.id) status = { ...status, meal: next };
  render();
}

// ---------- views ----------
const phaseText: Record<string, string> = {
  decode: '讀取照片… Reading photo', init: '載入模型… Starting models', locate: '找出食物… Finding food', classify: '辨識中… Naming items',
};
const mb = (n: number) => (n / 1048576).toFixed(1);

function progressCard(s: Extract<Status, { kind: 'progress' }>): HTMLElement {
  const dl = s.phase === 'download';
  const pct = dl && s.total ? Math.min(100, (100 * (s.loaded ?? 0)) / s.total) : 0;
  const bar = h('div', { class: 'progress' + (dl && s.total ? '' : ' indet') }, h('i', { style: dl && s.total ? `width:${pct}%` : '' }));
  return h('div', { class: 'card', role: 'status' },
    h('div', {}, dl ? '下載模型（僅第一次）… Downloading models (first time only)' : phaseText[s.phase] ?? s.phase),
    bar,
    dl && s.total ? h('div', { class: 'muted' }, `${mb(s.loaded ?? 0)} / ${mb(s.total)} MB`) : null,
    dl ? h('div', { class: 'muted' }, '之後離線也能用 Cached for offline use afterwards') : null);
}

function itemRow(m: Meal, it: LogItem, deletable = true): HTMLElement {
  return h('div', { class: 'row' },
    h('div', { class: 'name' }, displayName(it.name, it.unsure),
      it.name !== it.nameEn ? h('small', {}, it.nameEn) : null),
    h('div', { class: 'kcal' }, fmtKcal(it)),
    deletable ? h('button', { class: 'x', 'aria-label': `刪除 ${it.name} Delete item`, onClick: () => removeOneItem(m, it.id) }, '✕') : null);
}

function resultCard(s: Extract<Status, { kind: 'done' }>): HTMLElement {
  const m = s.meal, tot = mealTotal(m)!;
  const photo = h('div', { class: 'photo' }, h('img', { src: s.photo, alt: '' }));
  m.items.forEach(it => {
    const [x1, y1, x2, y2] = it.box;
    const first = y1 < 0.06;
    photo.append(h('div', { class: 'box' + (first ? ' first' : ''), style: `left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%` },
      h('span', {}, `${displayName(it.name, it.unsure)} ${fmtRange(it)}`)));
  });
  return h('div', { class: 'card' },
    photo,
    h('div', { style: 'margin-top:8px' }, ...m.items.map(it => itemRow(m, it))),
    h('div', { class: 'total' }, h('span', {}, '這餐 This meal'), h('span', {}, fmtKcal(tot))),
    h('div', { class: 'muted', style: 'margin-top:6px' }, storageErr ? '' : `✓ 已記錄 Saved · ${fmtTime(m.ts)} · ${s.backend}`),
    h('div', { class: 'btns', style: 'margin-top:8px' }, h('button', { class: 'btn', onClick: () => { status = { kind: 'idle' }; render(); } }, '關閉 Dismiss')));
}

function statusCard(): HTMLElement | null {
  switch (status.kind) {
    case 'idle': return null;
    case 'progress': return progressCard(status);
    case 'error':
      return h('div', { class: 'card' }, h('div', { class: 'err' }, '發生錯誤 Something went wrong'),
        h('div', { class: 'muted', style: 'overflow-wrap:anywhere' }, status.message),
        h('div', { class: 'btns', style: 'margin-top:8px' }, h('button', { class: 'btn', onClick: () => { status = { kind: 'idle' }; render(); } }, 'OK')));
    case 'none':
      return h('div', { class: 'card' }, h('div', { class: 'photo' }, h('img', { src: status.photo, alt: '' })),
        h('div', { class: 'warn', style: 'margin-top:8px;font-weight:600' }, '沒有認出食物，未記錄。 No food recognised; nothing logged.'),
        h('div', { class: 'muted' }, status.info),
        h('div', { class: 'btns', style: 'margin-top:8px' }, h('button', { class: 'btn', onClick: () => { status = { kind: 'idle' }; render(); } }, 'OK')));
    case 'done': return resultCard(status);
  }
}

const fmtDay = (k: string) => parseDayKey(k).toLocaleDateString('zh-TW', { month: 'long', day: 'numeric', weekday: 'short' });

function mealCard(m: Meal): HTMLElement {
  const tot = mealTotal(m)!;
  return h('div', { class: 'card meal' },
    m.thumb ? h('img', { class: 'thumb', src: thumbUrl(m), alt: '' }) : h('div', { class: 'thumb' }),
    h('div', { class: 'body' },
      h('div', { class: 'head' }, h('span', {}, fmtTime(m.ts)), h('span', {}, fmtKcal(tot))),
      ...m.items.map(it => itemRow(m, it)),
      h('div', { class: 'tools' }, h('button', { class: 'link', onClick: () => removeMeal(m.id) }, '🗑 刪除這餐 Delete meal'))));
}

function dayView(): HTMLElement[] {
  const ms = mealsOfDay(meals, day), t = dayTotal(ms);
  const out: HTMLElement[] = [];
  out.push(h('div', { class: 'card daytotal' },
    h('div', { class: 'muted' }, day === today ? '今天合計 Today' : '當日合計 Day total'),
    t ? h('div', { class: 'big' }, fmtRange(t), h('small', {}, ' kcal')) : h('div', { class: 'big', style: 'color:var(--muted)' }, '—'),
    t ? h('div', { class: 'muted' }, `中位數 median ≈ ${round10(t.p50)} kcal · ${ms.length} 餐 meals`) : h('div', { class: 'muted' }, '還沒有紀錄 Nothing logged')));
  if (!ms.length) out.push(h('div', { class: 'empty' }, '拍一張餐點照片開始吧 Snap a meal to start'));
  for (const m of ms) out.push(mealCard(m));
  // 7-day bars of daily medians
  const wk = weekMedians(meals, day), max = Math.max(1, ...wk.map(w => w.median ?? 0));
  out.push(h('div', { class: 'card' }, h('div', { style: 'font-weight:700' }, '近 7 天（中位數）Last 7 days (median)'),
    h('div', { class: 'week' }, ...wk.map(w => h('button', { class: 'd' + (w.key === day ? ' sel' : ''), 'aria-label': w.key, onClick: () => { day = w.key; render(); } },
      h('em', {}, w.median == null ? '' : String(round10(w.median))),
      h('i', { style: `height:${w.median == null ? 3 : Math.max(4, (w.median / max) * 84)}px` }),
      h('span', {}, parseDayKey(w.key).toLocaleDateString('zh-TW', { weekday: 'narrow' })))))));
  return out;
}

// ---------- settings sheet ----------
function sheet(): HTMLElement {
  const note = h('div', { class: 'muted', role: 'status' }, sheetNote);
  const about = h('pre', { class: 'about' }, '…');
  fetch(`${BASE}models/ATTRIBUTION.md`).then(async r => {
    const t = await r.text();
    about.textContent = r.ok && !/^\s*<!doctype|^\s*<html/i.test(t) ? t : '(ATTRIBUTION.md not available)';
  }).catch(() => { about.textContent = '(ATTRIBUTION.md not available offline)'; });
  const importInput = h('input', { type: 'file', accept: 'application/json,.json', class: 'sr', onChange: async (ev: Event) => {
    const f = (ev.target as HTMLInputElement).files?.[0]; if (!f) return;
    try { const n = await importBackup(await f.text()); meals = await getAllMeals(); sheetNote = `已匯入 ${n} 餐 Imported ${n} meals`; render(true); }
    catch (e: any) { note.textContent = `匯入失敗 Import failed: ${e?.message ?? e}`; }
    (ev.target as HTMLInputElement).value = '';
  } });
  const close = () => { sheetOpen = false; render(); };
  const box = h('div', { class: 'sheet', role: 'dialog', 'aria-modal': 'true', 'aria-label': '設定 Settings' },
    h('div', { class: 'row', style: 'border:0' }, h('h2', { style: 'flex:1' }, '設定 Settings'), h('button', { class: 'x', 'aria-label': 'Close', onClick: close }, '✕')),
    h('h3', {}, '備份 Backup'),
    h('div', { class: 'btns' },
      h('button', { class: 'btn', onClick: async () => {
        try {
          const blob = new Blob([await exportBackup(meals)], { type: 'application/json' });
          const a = h('a', { href: URL.createObjectURL(blob), download: `calorie-lens-${dayKey(Date.now())}.json` });
          document.body.append(a); a.click(); a.remove();
          note.textContent = `已匯出 ${meals.length} 餐 Exported ${meals.length} meals`;
        } catch (e: any) { note.textContent = `匯出失敗 Export failed: ${e?.message ?? e}`; }
      } }, '⬇ 匯出 Export JSON'),
      h('button', { class: 'btn', onClick: () => importInput.click() }, '⬆ 匯入 Import JSON'), importInput),
    note,
    h('h3', {}, '關於與資料來源 About & data sources'),
    h('p', { class: 'muted' }, DISCLAIMER),
    h('p', { class: 'muted' }, '照片只在你的手機上處理，不會上傳。 Photos are processed on this phone and never uploaded.'),
    about);
  const bg = h('div', { class: 'sheet-bg', onClick: (e: Event) => { if (e.target === bg) close(); } }, box);
  return bg;
}

// ---------- render ----------
let statusEl: HTMLElement | null = null;
let fileCam: HTMLInputElement, filePick: HTMLInputElement;

function renderStatus() {
  if (!statusEl) return;
  const c = statusCard();
  statusEl.replaceChildren(...(c ? [c] : []));
}

function render(keepSheet = false) {
  const app = document.getElementById('app')!;
  const nextDisabled = day >= today;
  const st = h("div", { id: "status" }); statusEl = st;
  app.replaceChildren(
    h('header', { class: 'top' },
      h('button', { class: 'icon-btn', 'aria-label': '前一天 Previous day', onClick: () => { day = shiftDay(day, -1); render(); } }, '‹'),
      h('div', { class: 'date' }, fmtDay(day), h('small', {}, day === today ? '今天 Today' : day)),
      h('button', { class: 'icon-btn', 'aria-label': '後一天 Next day', disabled: nextDisabled, onClick: () => { day = shiftDay(day, 1); render(); } }, '›'),
      h('button', { class: 'icon-btn', 'aria-label': '設定 Settings', onClick: () => { sheetOpen = true; render(); } }, '⚙')),
    h('div', { class: 'capture' },
      h('button', { class: 'snap', disabled: busy, onClick: () => fileCam.click() }, '📷 拍照記錄', h('small', {}, 'Snap meal')),
      h('button', { class: 'pick', disabled: busy, onClick: () => filePick.click() }, '從相簿選 Choose photo')),
    storageErr ? h('div', { class: 'card err' }, `無法使用本機儲存，紀錄不會保存。 Storage unavailable, meals will not be saved. (${storageErr})`) : "",
    st,
    ...dayView(),
    h('div', { class: 'disclaimer' }, DISCLAIMER),
  );
  renderStatus();
  if (sheetOpen && !keepSheet) document.body.append(sheet());
  else if (sheetOpen && keepSheet) { document.querySelector('.sheet-bg')?.remove(); document.body.append(sheet()); }
  if (!sheetOpen) document.querySelectorAll('.sheet-bg').forEach(e => e.remove());
}

// Hidden inputs live outside #app so re-rendering never detaches an input mid-pick.
function mkInput(capture: boolean): HTMLInputElement {
  const i = h('input', { type: 'file', accept: 'image/*', class: 'sr', tabindex: '-1', 'aria-hidden': 'true', capture: capture ? 'environment' : false,
    onChange: () => { const f = i.files?.[0]; i.value = ''; void snap(f); } });
  document.body.append(i);
  return i;
}

async function boot() {
  fileCam = mkInput(true); filePick = mkInput(false);
  storageErr = await storageOk();
  if (!storageErr) { try { meals = await getAllMeals(); } catch (e: any) { storageErr = String(e?.message ?? e); } }
  render();
  document.addEventListener('visibilitychange', () => {
    const t = dayKey(Date.now());
    if (!document.hidden && t !== today) { if (day === today) day = t; today = t; render(); }
  });
  if ('serviceWorker' in navigator && import.meta.env.PROD) {
    navigator.serviceWorker.register(`${BASE}sw.js`).catch(() => { /* optional */ });
  }
}
void boot();
