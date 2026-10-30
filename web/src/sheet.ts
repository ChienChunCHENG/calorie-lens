// Bottom-sheet chip picker. Reads/writes an Item through callbacks; all state lives in the caller.
import { data, dishIds } from './data';
import { containerChoices, defaultItem, UNKNOWN_CONTAINER } from './meal.ts';
import type { Item, SugarLevel, Ice } from './meal.ts';

const SUGARS: [SugarLevel, string][] = [['full', '全糖'], ['less', '少糖'], ['half', '半糖'], ['light', '微糖'], ['none', '無糖']];
const ICES: [Ice, string][] = [['normal', '正常冰'], ['less', '少冰'], ['none', '去冰']];
export const sugarZh = (s: SugarLevel) => SUGARS.find(x => x[0] === s)![1];

export interface SheetCtx { title: string; drinkOnly: boolean; preferBowl: boolean; get(): Item | null; set(i: Item | null): void }

const root = document.getElementById('sheet')!;
const backdrop = document.getElementById('backdrop')!;
let ctx: SheetCtx | null = null;

function chip(parent: HTMLElement, main: string, sub: string | null, on: boolean, onTap: () => void) {
  const b = document.createElement('button');
  b.className = 'chip' + (on ? ' on' : '');
  const m = document.createElement('span'); m.className = 'zh'; m.textContent = main; b.appendChild(m);
  if (sub) { const s = document.createElement('span'); s.className = 'en'; s.textContent = sub; b.appendChild(s); }
  b.addEventListener('click', onTap);
  parent.appendChild(b);
}
function section(title: string): HTMLElement {
  const h = document.createElement('div'); h.className = 'sec'; h.textContent = title; root.appendChild(h);
  const row = document.createElement('div'); row.className = 'chips'; root.appendChild(row);
  return row;
}

function render() {
  if (!ctx) return;
  const c = ctx;
  const item = c.get();
  root.replaceChildren();
  const head = document.createElement('div'); head.className = 'head';
  const t = document.createElement('div'); t.className = 'ttl'; t.textContent = c.title;
  const done = document.createElement('button'); done.className = 'btn'; done.textContent = 'Done';
  done.addEventListener('click', closeSheet);
  head.append(t, done); root.appendChild(head);

  const dishRow = section(c.drinkOnly ? 'Drink' : 'Dish');
  for (const id of dishIds) {
    const d = data.dishes[id];
    if (c.drinkOnly && d.kind !== 'drink') continue;
    chip(dishRow, d.zh, d.en, item?.dishId === id, () => {
      if (item?.dishId === id) return;
      c.set(defaultItem(data, id, c.preferBowl ? 'bowl' : undefined)); render();
    });
  }
  if (!item) return;
  const dish = data.dishes[item.dishId];
  const isDrink = dish.kind === 'drink';
  const cRow = section(isDrink ? 'Cup' : 'Container');
  const cids = containerChoices(data, item.dishId, item.containerId);
  for (const id of cids) chip(cRow, data.containers[id].zh, id, item.containerId === id, () => { c.set({ ...item, containerId: id }); render(); });
  if (!isDrink) chip(cRow, '不確定', 'unknown (default portion)', item.containerId === UNKNOWN_CONTAINER,
    () => { c.set({ ...item, containerId: UNKNOWN_CONTAINER }); render(); });
  if (isDrink) {
    const sRow = section('Sugar');
    for (const [k, zh] of SUGARS) chip(sRow, zh, k, item.sugar === k, () => { c.set({ ...item, sugar: k }); render(); });
    const iRow = section('Ice (display only)');
    for (const [k, zh] of ICES) chip(iRow, zh, null, item.ice === k, () => { c.set({ ...item, ice: k }); render(); });
  }
  const rm = document.createElement('button'); rm.className = 'btn ghost'; rm.textContent = 'Remove label';
  rm.addEventListener('click', () => { c.set(null); closeSheet(); });
  root.appendChild(rm);
}

export function openSheet(c: SheetCtx) { ctx = c; render(); root.classList.add('open'); backdrop.classList.add('open'); root.scrollTop = 0; }
export function closeSheet() { ctx = null; root.classList.remove('open'); backdrop.classList.remove('open'); }
export const sheetOpen = () => ctx !== null;
backdrop.addEventListener('click', closeSheet);
