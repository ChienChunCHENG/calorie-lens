// Converts ../ml/labels.yaml (flat one-line flow maps) to src/generated/labels.json. Runs in predev/prebuild.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const text = readFileSync(join(root, '../ml/labels.yaml'), 'utf8');
const out = [];
for (const line of text.split(/\r?\n/)) {
  const m = line.match(/^\s*-\s*\{\s*id:\s*([\w]+)\s*,\s*zh:\s*([^,]+?)\s*,\s*en:\s*([^,]+?)\s*,\s*type:\s*(\w+)\s*\}/);
  if (m) out.push({ id: m[1], zh: m[2], en: m[3], type: m[4] });
}
if (out.length < 15) throw new Error('labels.yaml parse found only ' + out.length + ' classes');
mkdirSync(join(root, 'src/generated'), { recursive: true });
writeFileSync(join(root, 'src/generated/labels.json'), JSON.stringify(out, null, 1) + '\n');
console.log(`gen-labels: ${out.length} classes`);
