/**
 * Copies MediaPipe's runtime WASM files and face_landmarker.task into web/public/
 * so the web app works completely offline without relying on external CDNs.
 */

import { cp, mkdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = join(here, '..');
const repoRoot = join(webRoot, '..');

const WASM_SOURCE = join(webRoot, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const WASM_TARGET = join(webRoot, 'public', 'mediapipe', 'wasm');

const MODEL_SOURCE = join(repoRoot, 'assets', 'face_landmarker.task');
const MODEL_TARGET = join(webRoot, 'public', 'assets', 'face_landmarker.task');

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (!(await exists(WASM_SOURCE))) {
    throw new Error(`Missing ${WASM_SOURCE}. Run npm install first.`);
  }
  await mkdir(WASM_TARGET, { recursive: true });
  await cp(WASM_SOURCE, WASM_TARGET, { recursive: true });
  console.log(`wasm  -> ${WASM_TARGET}`);

  if (!(await exists(MODEL_SOURCE))) {
    throw new Error(`Missing ${MODEL_SOURCE}. Run main.py once or download face_landmarker.task into assets/.`);
  }
  await mkdir(dirname(MODEL_TARGET), { recursive: true });
  await cp(MODEL_SOURCE, MODEL_TARGET);
  const { size } = await stat(MODEL_TARGET);
  console.log(`model -> ${MODEL_TARGET} (${(size / 1e6).toFixed(1)} MB)`);
}

await main();
