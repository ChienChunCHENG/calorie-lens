// Loads public/models/detector.onnx with onnxruntime-node (CPU) and runs one dummy inference.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
import * as ort from 'onnxruntime-node';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const imgsz = parseInt(process.argv[2] || '320', 10);
const model = readFileSync(join(root, 'public/models/detector.onnx'));
const session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ['cpu'] });
console.log('inputs', session.inputNames, 'outputs', session.outputNames);
const x = new ort.Tensor('float32', new Float32Array(3 * imgsz * imgsz).fill(0.5), [1, 3, imgsz, imgsz]);
const t0 = performance.now();
const out = await session.run({ [session.inputNames[0]]: x });
const o = out[session.outputNames[0]];
console.log('output dims', JSON.stringify(o.dims), 'ms', (performance.now() - t0).toFixed(1));
