// Run: node --experimental-strip-types --test src/meal.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildMealData, defaultItem, estimateItem, volumeFor, totalOf, autoDishId } from './meal.ts';
import type { RawNutrition, Item } from './meal.ts';
import { Assignments } from './assign.ts';
import { IouTracker } from './tracker.ts';
import { suppressContained } from './yolo.ts';

const raw = JSON.parse(readFileSync(new URL('../../data/nutrition.json', import.meta.url), 'utf8')) as RawNutrition;
const data = buildMealData(raw);
const close = (a: number, b: number, tol: number) => assert.ok(Math.abs(a - b) <= tol, `${a} not within ${tol} of ${b}`);

test('bowl_large volume triple = capacity x fill (0.6/0.8/1.0)', () => {
  const v = volumeFor('bowl_large', data.containers)!;
  close(v.min, 650 * 0.6, 1e-9); close(v.typ, 900 * 0.8, 1e-9); close(v.max, 1200, 1e-9);
  const rice = estimateItem(data, { dishId: 'rice_white', containerId: 'bowl_large', sugar: 'half', ice: 'normal' });
  assert.ok(rice.lo < rice.med && rice.med < rice.hi);
  assert.deepEqual(rice.flags.includes('portion_default'), false);
});

test('plate / bento / unknown -> null volume -> portion_default', () => {
  for (const c of ['plate_26cm', 'bento_box', 'unknown']) assert.equal(volumeFor(c, data.containers), null);
  const e = estimateItem(data, { dishId: 'pork_chop', containerId: 'plate_26cm', sugar: 'half', ice: 'normal' });
  assert.ok(e.flags.includes('portion_default'));
});

test('bubble milk tea 700 half ~ 584 kcal median; default item is cup_700 + half', () => {
  const it = defaultItem(data, 'bubble_milk_tea');
  assert.equal(it.containerId, 'cup_700'); assert.equal(it.sugar, 'half');
  close(estimateItem(data, it).med, 584, 1.5);
  close(estimateItem(data, { ...it, sugar: 'full' }).med, 651, 1.5);
  assert.ok(estimateItem(data, { ...it, containerId: 'cup_500' }).med < 584);
});

test('combineMeal over 3 items', () => {
  const items: Item[] = [defaultItem(data, 'lu_rou_fan'), defaultItem(data, 'fried_egg'), defaultItem(data, 'bubble_milk_tea')];
  const es = items.map(i => estimateItem(data, i));
  const t = totalOf(es);
  close(t.med, es.reduce((s, e) => s + e.med, 0), 1e-9);
  assert.ok(t.lo < t.med && t.med < t.hi);
  // quadrature: narrower than a linear sum of the item bounds
  assert.ok(t.lo > es.reduce((s, e) => s + e.lo, 0));
  assert.ok(t.hi < es.reduce((s, e) => s + e.hi, 0));
});

test('preselected container is container_prior[0]; COCO bowl prefers a bowl', () => {
  assert.equal(defaultItem(data, 'chicken_leg').containerId, 'bento_box');
  assert.equal(defaultItem(data, 'chicken_leg', 'bowl').containerId, 'bowl_small');
  assert.equal(defaultItem(data, 'beef_noodle_soup', 'bowl').containerId, 'bowl_large');
});

test('auto-assign only for dish-id labels', () => {
  assert.equal(autoDishId('lu_rou_fan', data), 'lu_rou_fan');
  assert.equal(autoDishId('bowl', data), null);
});

test('auto assignment follows the label, re-attaches within 1.5 s at IoU > 0.3, not after', () => {
  const a = new Assignments();
  const auto = (l: string | undefined) => (autoDishId(l, data) ? defaultItem(data, l!) : null);
  const box = { x1: 0, y1: 0, x2: 100, y2: 100 };
  a.sync([{ id: 1, box, label: 'rice_white' }], 0, auto);
  assert.equal(a.get(1)?.dishId, 'rice_white');
  a.sync([{ id: 1, box, label: 'lu_rou_fan' }], 100, auto); // track's vote changed
  assert.equal(a.get(1)?.dishId, 'lu_rou_fan');
  a.sync([], 200, auto); // lost
  assert.equal(a.activeItems().length, 1); // still counted
  a.sync([{ id: 2, box: { x1: 10, y1: 10, x2: 110, y2: 110 }, label: 'lu_rou_fan' }], 900, auto);
  assert.equal(a.get(2)?.dishId, 'lu_rou_fan');
  assert.equal(a.activeItems().length, 1);
  // too late: a new track gets its own label's item
  a.sync([], 1000, auto);
  a.sync([{ id: 3, box, label: 'bowl' }], 3000, auto);
  assert.equal(a.get(3), null);
  a.sync([{ id: 4, box, label: 'fried_egg' }], 3100, auto);
  // far away: no re-attach
  a.sync([], 3200, auto);
  a.sync([{ id: 5, box: { x1: 500, y1: 500, x2: 600, y2: 600 } }], 3300, auto);
  assert.equal(a.get(5), null);
});

test('tracker keeps one track per dish and votes its label', () => {
  const t = new IouTracker();
  const b = { x1: 0, y1: 0, x2: 100, y2: 100 };
  t.update([{ ...b, cls: 2, score: 0.6 }]);
  t.update([{ ...b, cls: 1, score: 0.4 }]); // one noisy frame
  t.update([{ ...b, cls: 2, score: 0.5 }]);
  assert.equal(t.tracks.length, 1);
  assert.equal(t.tracks[0].cls, 2);
  t.update([{ ...b, cls: 1, score: 0.9 }]);
  t.update([{ ...b, cls: 1, score: 0.9 }]);
  assert.equal(t.tracks[0].cls, 1); // sustained evidence wins
});

test('boxes inside a whole-bowl dish or a same-dish box are dropped', () => {
  const bowl = { x1: 0, y1: 0, x2: 200, y2: 200, cls: 8, score: 0.8 };
  const beef = { x1: 20, y1: 20, x2: 80, y2: 80, cls: 2, score: 0.5 };
  const rice = { x1: 0, y1: 0, x2: 200, y2: 200, cls: 0, score: 0.7 };
  const leg = { x1: 20, y1: 20, x2: 80, y2: 80, cls: 2, score: 0.5 };
  assert.deepEqual(suppressContained([bowl, beef], new Set([8])), [bowl]);
  assert.deepEqual(suppressContained([rice, leg], new Set([8])), [rice, leg]); // bento: leg on rice stays
  const eggBig = { x1: 0, y1: 0, x2: 200, y2: 200, cls: 5, score: 0.4 };
  const egg = { x1: 10, y1: 10, x2: 90, y2: 90, cls: 5, score: 0.6 };
  assert.deepEqual(suppressContained([eggBig, egg], new Set()), [egg]);
});
