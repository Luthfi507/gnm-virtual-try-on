/**
 * Identity fit & similarity transform math: fits GNM identity coefficients from tracked landmarks.
 */

import type { GnmModel } from './model.ts';

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface SimilarityTransform {
  scale: number;
  rotation: Float64Array;
  translation: Float64Array;
}

export interface IdentityFitOptions {
  iterations?: number;
  ridge?: number;
  depthWeight?: number;
  limit?: number;
  rigidOnly?: boolean;
  components?: number;
}

export interface IdentityFit {
  identity: Float32Array;
  transform: SimilarityTransform;
  points: number;
  components: number;
  rmsBefore: number;
  rmsAfter: number;
  peak: number;
}

const DEFAULTS = {
  iterations: 4,
  components: 24,
  ridge: 0.06,
  depthWeight: 0.1,
  limit: 3,
  rigidOnly: true,
} as const;

export function landmarksToModelAxes(
  landmarks: readonly Landmark[],
  indices: ArrayLike<number>,
  aspect: number,
  out: Float64Array,
): Float64Array {
  for (let i = 0; i < indices.length; i++) {
    const landmark = landmarks[indices[i]];
    out[i * 3] = landmark.x * aspect;
    out[i * 3 + 1] = -landmark.y;
    out[i * 3 + 2] = -landmark.z * aspect;
  }
  return out;
}

function jacobiEigen(matrix: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const vectors = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vectors[i * n + i] = 1;

  for (let sweep = 0; sweep < 32; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += matrix[p * n + q] * matrix[p * n + q];
    }
    if (off < 1e-24) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = matrix[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;

        const theta = (matrix[q * n + q] - matrix[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        for (let k = 0; k < n; k++) {
          const akp = matrix[k * n + p];
          const akq = matrix[k * n + q];
          matrix[k * n + p] = c * akp - s * akq;
          matrix[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = matrix[p * n + k];
          const aqk = matrix[q * n + k];
          matrix[p * n + k] = c * apk - s * aqk;
          matrix[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = vectors[k * n + p];
          const vkq = vectors[k * n + q];
          vectors[k * n + p] = c * vkp - s * vkq;
          vectors[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = matrix[i * n + i];
  return { values, vectors };
}

export function fitSimilarity(source: Float64Array, target: Float64Array): SimilarityTransform {
  if (source.length !== target.length) {
    throw new Error(`source and target length mismatch: ${source.length} vs ${target.length}`);
  }
  const count = source.length / 3;
  if (count < 3) throw new Error(`Need at least 3 points, got ${count}`);

  let sx = 0, sy = 0, sz = 0, tx = 0, ty = 0, tz = 0;
  for (let i = 0; i < count; i++) {
    sx += source[i * 3]; sy += source[i * 3 + 1]; sz += source[i * 3 + 2];
    tx += target[i * 3]; ty += target[i * 3 + 1]; tz += target[i * 3 + 2];
  }
  sx /= count; sy /= count; sz /= count;
  tx /= count; ty /= count; tz /= count;

  const m = new Float64Array(9);
  let sourceVariance = 0;
  for (let i = 0; i < count; i++) {
    const ax = source[i * 3] - sx;
    const ay = source[i * 3 + 1] - sy;
    const az = source[i * 3 + 2] - sz;
    const bx = target[i * 3] - tx;
    const by = target[i * 3 + 1] - ty;
    const bz = target[i * 3 + 2] - tz;

    m[0] += ax * bx; m[1] += ax * by; m[2] += ax * bz;
    m[3] += ay * bx; m[4] += ay * by; m[5] += ay * bz;
    m[6] += az * bx; m[7] += az * by; m[8] += az * bz;
    sourceVariance += ax * ax + ay * ay + az * az;
  }

  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = m;
  const n = Float64Array.from([
    xx + yy + zz, yz - zy, zx - xz, xy - yx,
    yz - zy, xx - yy - zz, xy + yx, zx + xz,
    zx - xz, xy + yx, -xx + yy - zz, yz + zy,
    xy - yx, zx + xz, yz + zy, -xx - yy + zz,
  ]);

  const { values, vectors } = jacobiEigen(n, 4);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  const qw = vectors[best];
  const qx = vectors[4 + best];
  const qy = vectors[8 + best];
  const qz = vectors[12 + best];

  const rotation = Float64Array.from([
    1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
  ]);

  let aligned = 0;
  for (let i = 0; i < count; i++) {
    const ax = source[i * 3] - sx;
    const ay = source[i * 3 + 1] - sy;
    const az = source[i * 3 + 2] - sz;
    aligned +=
      (target[i * 3] - tx) * (rotation[0] * ax + rotation[1] * ay + rotation[2] * az) +
      (target[i * 3 + 1] - ty) * (rotation[3] * ax + rotation[4] * ay + rotation[5] * az) +
      (target[i * 3 + 2] - tz) * (rotation[6] * ax + rotation[7] * ay + rotation[8] * az);
  }
  const scale = aligned / Math.max(sourceVariance, 1e-12);

  const translation = Float64Array.from([
    tx - scale * (rotation[0] * sx + rotation[1] * sy + rotation[2] * sz),
    ty - scale * (rotation[3] * sx + rotation[4] * sy + rotation[5] * sz),
    tz - scale * (rotation[6] * sx + rotation[7] * sy + rotation[8] * sz),
  ]);

  return { scale, rotation, translation };
}

export function applySimilarity(
  points: Float64Array,
  transform: SimilarityTransform,
  out: Float64Array,
): Float64Array {
  const { scale, rotation: r, translation: t } = transform;
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i], y = points[i + 1], z = points[i + 2];
    out[i] = scale * (r[0] * x + r[1] * y + r[2] * z) + t[0];
    out[i + 1] = scale * (r[3] * x + r[4] * y + r[5] * z) + t[1];
    out[i + 2] = scale * (r[6] * x + r[7] * y + r[8] * z) + t[2];
  }
  return out;
}

export function choleskySolve(matrix: Float64Array, rhs: Float64Array, n: number): Float64Array {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = matrix[i * n + j];
      for (let k = 0; k < j; k++) sum -= l[i * n + k] * l[j * n + k];
      if (i === j) {
        if (sum <= 0) throw new Error('Normal matrix is not positive definite.');
        l[i * n + j] = Math.sqrt(sum);
      } else {
        l[i * n + j] = sum / l[j * n + j];
      }
    }
  }

  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = rhs[i];
    for (let k = 0; k < i; k++) sum -= l[i * n + k] * y[k];
    y[i] = sum / l[i * n + i];
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let k = i + 1; k < n; k++) sum -= l[k * n + i] * x[k];
    x[i] = sum / l[i * n + i];
  }
  return x;
}

export function fitIdentity(
  model: GnmModel,
  landmarks: readonly Landmark[],
  aspect: number,
  options: IdentityFitOptions = {},
): IdentityFit {
  const correspondence = model.correspondence;
  if (!correspondence) {
    throw new Error('Model exported without correspondence.');
  }
  const { iterations, ridge, depthWeight, limit, rigidOnly, components } = {
    ...DEFAULTS,
    ...options,
  };

  const active: number[] = [];
  for (let i = 0; i < correspondence.rigid.length; i++) {
    if (!rigidOnly || correspondence.rigid[i]) active.push(i);
  }
  const count = active.length;
  const dim = Math.max(1, Math.min(components, model.identityDim));
  const rows = count * 3;
  const stride = model.vertexCount * 3;

  const basePoints = new Float64Array(rows);
  const design = new Float64Array(rows * dim);
  for (let i = 0; i < count; i++) {
    const vertex = correspondence.vertices[active[i]];
    for (let axis = 0; axis < 3; axis++) {
      const row = i * 3 + axis;
      basePoints[row] = correspondence.reference[active[i] * 3 + axis];
      for (let k = 0; k < dim; k++) {
        design[row * dim + k] = model.identityBasis[k * stride + vertex * 3 + axis];
      }
    }
  }

  const weights = new Float64Array(rows);
  for (let i = 0; i < count; i++) {
    weights[i * 3] = 1;
    weights[i * 3 + 1] = 1;
    weights[i * 3 + 2] = depthWeight;
  }

  const normal = new Float64Array(dim * dim);
  for (let row = 0; row < rows; row++) {
    const w = weights[row];
    if (w === 0) continue;
    const base = row * dim;
    for (let a = 0; a < dim; a++) {
      const wa = w * design[base + a];
      if (wa === 0) continue;
      for (let b = a; b < dim; b++) normal[a * dim + b] += wa * design[base + b];
    }
  }
  for (let a = 0; a < dim; a++) {
    for (let b = 0; b < a; b++) normal[a * dim + b] = normal[b * dim + a];
  }

  let trace = 0;
  for (let a = 0; a < dim; a++) trace += normal[a * dim + a];
  const lambda = (ridge * trace) / dim;
  for (let a = 0; a < dim; a++) normal[a * dim + a] += lambda;

  const observed = new Float64Array(rows);
  const indices = new Uint16Array(count);
  for (let i = 0; i < count; i++) indices[i] = correspondence.landmarks[active[i]];
  landmarksToModelAxes(landmarks, indices, aspect, observed);

  const identity = new Float64Array(dim);
  const current = new Float64Array(rows);
  const aligned = new Float64Array(rows);
  const rhs = new Float64Array(dim);

  const residualRms = (a: Float64Array, b: Float64Array): number => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = (a[i] - b[i]) * 1000;
      sum += d * d;
    }
    return Math.sqrt((sum * 3) / a.length);
  };

  let transform: SimilarityTransform | null = null;
  let rmsBefore = 0;
  let rmsAfter = 0;

  for (let iteration = 0; iteration < iterations; iteration++) {
    current.set(basePoints);
    for (let k = 0; k < dim; k++) {
      const c = identity[k];
      if (c === 0) continue;
      for (let row = 0; row < rows; row++) current[row] += c * design[row * dim + k];
    }
    transform = fitSimilarity(observed, current);
    applySimilarity(observed, transform, aligned);
    if (iteration === 0) rmsBefore = residualRms(aligned, basePoints);

    rhs.fill(0);
    for (let row = 0; row < rows; row++) {
      const w = weights[row];
      if (w === 0) continue;
      const target = w * (aligned[row] - basePoints[row]);
      const base = row * dim;
      for (let a = 0; a < dim; a++) rhs[a] += design[base + a] * target;
    }

    const solved = choleskySolve(normal, rhs, dim);
    for (let k = 0; k < dim; k++) {
      identity[k] = Math.max(-limit, Math.min(limit, solved[k]));
    }
  }

  current.set(basePoints);
  for (let k = 0; k < dim; k++) {
    const c = identity[k];
    if (c === 0) continue;
    for (let row = 0; row < rows; row++) current[row] += c * design[row * dim + k];
  }
  transform = fitSimilarity(observed, current);
  applySimilarity(observed, transform, aligned);
  rmsAfter = residualRms(aligned, current);

  let peak = 0;
  const result = new Float32Array(model.identityDim);
  for (let k = 0; k < dim; k++) {
    result[k] = identity[k];
    peak = Math.max(peak, Math.abs(identity[k]));
  }

  return { identity: result, transform: transform!, points: count, components: dim, rmsBefore, rmsAfter, peak };
}
