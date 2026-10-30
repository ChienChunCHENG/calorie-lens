import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { estimateSolid, estimateDrink, combineMeal, formatRange } from './rangeEngine.ts';

const path = new URL('../../spec/vectors.json', import.meta.url);
const cases = JSON.parse(readFileSync(path, 'utf-8')).cases;

function run(c: any): any {
  const i = c.input;
  if (c.fn === 'estimateSolid') return estimateSolid(i.dish, i.volumeMl);
  if (c.fn === 'estimateDrink')
    return estimateDrink(i.drink, i.sizeMl, i.sugarLevel, i.fullSugarGPer700, i.sugarFraction);
  return combineMeal(i.items);
}

function close(a: number, b: number, what: string) {
  const tol = 1e-9 * Math.max(Math.abs(a), Math.abs(b)) + 1e-12;
  assert.ok(Math.abs(a - b) <= tol, what + ': ' + a + ' vs ' + b);
}

test('has at least 20 vectors', () => assert.ok(cases.length >= 20));

for (const c of cases) {
  test(c.name, () => {
    if (c.expected.error) {
      assert.throws(() => run(c));
      return;
    }
    const r = run(c);
    for (const k of ['lo', 'med', 'hi']) close(r[k], c.expected[k], k);
    assert.deepEqual(r.flags ?? [], c.expected.flags);
  });
}

test('formatRange', () => assert.equal(formatRange(447.4, 602.6), '445–605'));
