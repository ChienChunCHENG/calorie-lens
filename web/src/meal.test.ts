// Run: node --experimental-strip-types --test src/meal.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildMealData, defaultItem, estimateItem, volumeFor, totalOf, autoDishId } from './meal.ts';
import type { RawNutrition, Item } from './meal.ts';
import { Assignments } from './assign.ts';

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

test('assignment sticks, re-attaches within 1.5 s at IoU > 0.3, not after', () => {
  const a = new Assignments();
  const box = { x1: 0, y1: 0, x2: 100, y2: 100 };
  a.sync([{ id: 1, box }], 0);
  a.set(1, defaultItem(data, 'rice_white'));
  a.sync([{ id: 1, box }], 100);
  assert.equal(a.get(1)?.dishId, 'rice_white');
  a.sync([], 200); // lost
  assert.equal(a.activeItems().length, 1); // still counted
  a.sync([{ id: 2, box: { x1: 10, y1: 10, x2: 110, y2: 110 } }], 900);
  assert.equal(a.get(2)?.dishId, 'rice_white');
  assert.equal(a.activeItems().length, 1);
  // too late
  a.sync([], 1000);
  a.sync([{ id: 3, box }], 3000);
  assert.equal(a.get(3), null);
  // far away
  a.set(3, defaultItem(data, 'fried_egg'));
  a.sync([], 3100);
  a.sync([{ id: 4, box: { x1: 500, y1: 500, x2: 600, y2: 600 } }], 3200);
  assert.equal(a.get(4), null);
  // auto-assign for new track
  a.sync([{ id: 5, box, label: 'pork_chop' }], 9000, l => (autoDishId(l, data) ? defaultItem(data, l!) : null));
  assert.equal(a.get(5)?.dishId, 'pork_chop');
});
