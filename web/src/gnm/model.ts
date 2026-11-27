/**
 * GNM head parametric forward pass in TypeScript.
 * Decoupled from Three.js/DOM so it can be unit-tested in Node.
 */

export interface ArrayView {
  dtype: 'float32' | 'float16' | 'uint16' | 'uint8';
  shape: number[];
  byteOffset: number;
  byteLength: number;
}

export interface Manifest {
  model: {
    version: string;
    variant: string;
    vertexCount: number;
    jointCount: number;
    jointNames: string[];
    jointParentIndices: number[];
    componentNames: string[];
    maxNormalCorners: number;
  };
  correspondence: { count: number };
  correctives?: { names: string[]; blendshapes: string[] };
  identity: { count: number; sourceCount: number; indices: number[]; names: string[] };
  expression: {
    count: number;
    sourceCount: number;
    indices: number[];
    names: string[];
    regions: Record<string, { start: number; count: number }>;
  };
  arrays: Record<string, ArrayView>;
}

const HALF_TO_FLOAT = /* @__PURE__ */ (() => {
  const table = new Float32Array(65536);
  const view = new DataView(new ArrayBuffer(4));
  for (let bits = 0; bits < 65536; bits++) {
    const sign = (bits & 0x8000) << 16;
    const exponent = (bits & 0x7c00) >> 10;
    const mantissa = bits & 0x03ff;

    if (exponent === 0) {
      view.setFloat32(0, mantissa * 5.960464477539063e-8);
      table[bits] = view.getFloat32(0) * (bits & 0x8000 ? -1 : 1);
      continue;
    }
    if (exponent === 0x1f) {
      table[bits] = mantissa ? NaN : (bits & 0x8000 ? -Infinity : Infinity);
      continue;
    }
    view.setUint32(0, sign | ((exponent - 15 + 127) << 23) | (mantissa << 13));
    table[bits] = view.getFloat32(0);
  }
  return table;
})();

function decodeHalf(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) out[i] = HALF_TO_FLOAT[bits[i]];
  return out;
}

function arrayView(buffer: ArrayBuffer, spec: ArrayView): Float32Array | Uint16Array | Uint8Array {
  switch (spec.dtype) {
    case 'float32':
      return new Float32Array(buffer, spec.byteOffset, spec.byteLength / 4);
    case 'float16':
    case 'uint16':
      return new Uint16Array(buffer, spec.byteOffset, spec.byteLength / 2);
    case 'uint8':
      return new Uint8Array(buffer, spec.byteOffset, spec.byteLength);
  }
}

function floatArray(buffer: ArrayBuffer, spec: ArrayView): Float32Array {
  const raw = arrayView(buffer, spec);
  if (spec.dtype === 'float16') return decodeHalf(raw as Uint16Array);
  if (spec.dtype === 'float32') return raw as Float32Array;
  throw new Error(`Expected float array, got ${spec.dtype}`);
}

function axisAngleToMatrix(x: number, y: number, z: number, out: Float32Array, at: number): void {
  const angle = Math.hypot(x, y, z);
  if (angle < 1e-12) {
    out[at] = 1; out[at + 1] = 0; out[at + 2] = 0;
    out[at + 3] = 0; out[at + 4] = 1; out[at + 5] = 0;
    out[at + 6] = 0; out[at + 7] = 0; out[at + 8] = 1;
    return;
  }
  const ax = x / angle, ay = y / angle, az = z / angle;
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  const t = 1 - c;

  out[at] = t * ax * ax + c;
  out[at + 1] = t * ax * ay - s * az;
  out[at + 2] = t * ax * az + s * ay;
  out[at + 3] = t * ax * ay + s * az;
  out[at + 4] = t * ay * ay + c;
  out[at + 5] = t * ay * az - s * ax;
  out[at + 6] = t * ax * az - s * ay;
  out[at + 7] = t * ay * az + s * ax;
  out[at + 8] = t * az * az + c;
}

export class GnmModel {
  readonly manifest: Manifest;
  readonly vertexCount: number;
  readonly jointCount: number;
  readonly identityDim: number;
  readonly expressionDim: number;

  readonly template: Float32Array;
  readonly identityBasis: Float32Array;
  readonly expressionBasis: Float32Array;
  readonly correctiveBasis: Float32Array;
  readonly correctiveDim: number;
  readonly correctiveBlendshapes: string[];
  readonly templateJoints: Float32Array;
  readonly jointIdentityBasis: Float32Array;
  readonly skinningWeights: Float32Array;
  readonly triangles: Uint16Array;
  readonly quadEdges: Uint16Array;
  readonly componentIds: Uint8Array;
  readonly normalAdjacency: Uint16Array;
  readonly normalAdjacencyCount: Uint8Array;
  readonly maxNormalCorners: number;
  readonly jointParents: number[];
  readonly buffer: ArrayBuffer;

  readonly correspondence: {
    landmarks: Uint16Array;
    vertices: Uint16Array;
    rigid: Uint8Array;
    reference: Float32Array;
    colors: Uint8Array;
  } | null;

  private readonly bindVertices: Float32Array;
  private readonly bindJoints: Float32Array;
  private readonly transforms: Float32Array;

  constructor(manifest: Manifest, buffer: ArrayBuffer) {
    this.manifest = manifest;
    const a = manifest.arrays;

    this.template = floatArray(buffer, a.template_vertex_positions);
    this.identityBasis = floatArray(buffer, a.identity_basis);
    this.expressionBasis = floatArray(buffer, a.expression_basis);
    this.correctiveBasis = a.corrective_basis ? floatArray(buffer, a.corrective_basis) : new Float32Array(0);
    this.correctiveBlendshapes = manifest.correctives?.blendshapes ?? [];
    this.correctiveDim = this.correctiveBlendshapes.length;
    this.templateJoints = floatArray(buffer, a.template_joint_positions);
    this.jointIdentityBasis = floatArray(buffer, a.joint_identity_basis);
    this.skinningWeights = floatArray(buffer, a.skinning_weights);
    this.triangles = arrayView(buffer, a.triangles) as Uint16Array;
    this.quadEdges = arrayView(buffer, a.quad_edges) as Uint16Array;
    this.componentIds = arrayView(buffer, a.component_ids) as Uint8Array;
    this.normalAdjacency = arrayView(buffer, a.normal_adjacency) as Uint16Array;
    this.normalAdjacencyCount = arrayView(buffer, a.normal_adjacency_count) as Uint8Array;
    this.maxNormalCorners = manifest.model.maxNormalCorners;
    this.buffer = buffer;

    this.correspondence = a.correspondence_landmarks
      ? {
          landmarks: arrayView(buffer, a.correspondence_landmarks) as Uint16Array,
          vertices: arrayView(buffer, a.correspondence_vertices) as Uint16Array,
          rigid: arrayView(buffer, a.correspondence_rigid) as Uint8Array,
          reference: floatArray(buffer, a.correspondence_reference),
          colors: arrayView(buffer, a.correspondence_colors) as Uint8Array,
        }
      : null;

    this.vertexCount = manifest.model.vertexCount;
    this.jointCount = manifest.model.jointCount;
    this.identityDim = manifest.identity.count;
    this.expressionDim = manifest.expression.count;
    this.jointParents = manifest.model.jointParentIndices;

    this.bindVertices = new Float32Array(this.vertexCount * 3);
    this.bindJoints = new Float32Array(this.jointCount * 3);
    this.transforms = new Float32Array(this.jointCount * 12);
  }

  private accumulate(coefficients: ArrayLike<number>, basis: Float32Array, out: Float32Array): void {
    const stride = out.length;
    for (let i = 0; i < coefficients.length; i++) {
      const c = coefficients[i];
      if (c > -1e-8 && c < 1e-8) continue;
      const base = i * stride;
      for (let k = 0; k < stride; k++) out[k] += c * basis[base + k];
    }
  }

  bindPose(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    correctives?: ArrayLike<number>,
  ): Float32Array {
    this.bindVertices.set(this.template);
    this.accumulate(identity, this.identityBasis, this.bindVertices);
    this.accumulate(expression, this.expressionBasis, this.bindVertices);
    if (correctives) {
      this.accumulate(correctives, this.correctiveBasis, this.bindVertices);
    }
    return this.bindVertices;
  }

  private poseJoints(rotations: ArrayLike<number>, translation: ArrayLike<number>): void {
    const joints = this.bindJoints;
    const rotation = new Float32Array(9);
    const world = new Float32Array(this.jointCount * 12);

    for (let j = 0; j < this.jointCount; j++) {
      axisAngleToMatrix(rotations[j * 3], rotations[j * 3 + 1], rotations[j * 3 + 2], rotation, 0);

      const parent = this.jointParents[j];
      const tx = parent < 0 ? joints[0] + translation[0] : joints[j * 3] - joints[parent * 3];
      const ty = parent < 0 ? joints[1] + translation[1] : joints[j * 3 + 1] - joints[parent * 3 + 1];
      const tz = parent < 0 ? joints[2] + translation[2] : joints[j * 3 + 2] - joints[parent * 3 + 2];

      const o = j * 12;
      if (parent < 0) {
        for (let r = 0; r < 3; r++) {
          world[o + r * 4] = rotation[r * 3];
          world[o + r * 4 + 1] = rotation[r * 3 + 1];
          world[o + r * 4 + 2] = rotation[r * 3 + 2];
        }
        world[o + 3] = tx; world[o + 7] = ty; world[o + 11] = tz;
      } else {
        const p = parent * 12;
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            world[o + r * 4 + c] =
              world[p + r * 4] * rotation[c] +
              world[p + r * 4 + 1] * rotation[3 + c] +
              world[p + r * 4 + 2] * rotation[6 + c];
          }
          world[o + r * 4 + 3] =
            world[p + r * 4] * tx +
            world[p + r * 4 + 1] * ty +
            world[p + r * 4 + 2] * tz +
            world[p + r * 4 + 3];
        }
      }
    }

    this.transforms.set(world);
    for (let j = 0; j < this.jointCount; j++) {
      const o = j * 12;
      const jx = joints[j * 3], jy = joints[j * 3 + 1], jz = joints[j * 3 + 2];
      for (let r = 0; r < 3; r++) {
        this.transforms[o + r * 4 + 3] -=
          world[o + r * 4] * jx + world[o + r * 4 + 1] * jy + world[o + r * 4 + 2] * jz;
      }
    }
  }

  jointTransforms(
    identity: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
  ): Float32Array {
    this.bindJoints.set(this.templateJoints);
    this.accumulate(identity, this.jointIdentityBasis, this.bindJoints);
    this.poseJoints(rotations, translation);
    return this.transforms;
  }

  evaluate(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
    out?: Float32Array,
    correctives?: ArrayLike<number>,
  ): Float32Array {
    const result = out ?? new Float32Array(this.vertexCount * 3);
    const vertices = this.bindPose(identity, expression, correctives);

    this.bindJoints.set(this.templateJoints);
    this.accumulate(identity, this.jointIdentityBasis, this.bindJoints);
    this.poseJoints(rotations, translation);

    const weights = this.skinningWeights;
    const transforms = this.transforms;
    const vertexCount = this.vertexCount;

    result.fill(0);
    for (let j = 0; j < this.jointCount; j++) {
      const o = j * 12;
      const m0 = transforms[o], m1 = transforms[o + 1], m2 = transforms[o + 2], m3 = transforms[o + 3];
      const m4 = transforms[o + 4], m5 = transforms[o + 5], m6 = transforms[o + 6], m7 = transforms[o + 7];
      const m8 = transforms[o + 8], m9 = transforms[o + 9], m10 = transforms[o + 10], m11 = transforms[o + 11];
      const base = j * vertexCount;

      for (let v = 0; v < vertexCount; v++) {
        const w = weights[base + v];
        if (w === 0) continue;
        const i = v * 3;
        const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2];
        result[i] += w * (m0 * x + m1 * y + m2 * z + m3);
        result[i + 1] += w * (m4 * x + m5 * y + m6 * z + m7);
        result[i + 2] += w * (m8 * x + m9 * y + m10 * z + m11);
      }
    }
    return result;
  }

  groupTrianglesByComponent(): {
    indices: Uint16Array;
    groups: { component: number; start: number; count: number }[];
  } {
    const componentCount = this.manifest.model.componentNames.length;
    const buckets: number[][] = Array.from({ length: componentCount }, () => []);

    for (let t = 0; t < this.triangles.length; t += 3) {
      const a = this.componentIds[this.triangles[t]];
      const b = this.componentIds[this.triangles[t + 1]];
      const c = this.componentIds[this.triangles[t + 2]];
      if (a !== b || b !== c) {
        throw new Error(`Triangle ${t / 3} spans components ${a}/${b}/${c}.`);
      }
      buckets[a].push(this.triangles[t], this.triangles[t + 1], this.triangles[t + 2]);
    }

    const indices = new Uint16Array(this.triangles.length);
    const groups: { component: number; start: number; count: number }[] = [];
    let offset = 0;
    for (let component = 0; component < componentCount; component++) {
      const bucket = buckets[component];
      if (bucket.length === 0) continue;
      indices.set(bucket, offset);
      groups.push({ component, start: offset, count: bucket.length });
      offset += bucket.length;
    }
    return { indices, groups };
  }
}

export async function loadGnmModel(baseUrl: string): Promise<GnmModel> {
  const [manifest, buffer] = await Promise.all([
    fetch(`${baseUrl}/gnm_head.json`).then((r) => r.json() as Promise<Manifest>),
    fetch(`${baseUrl}/gnm_head.bin`).then((r) => r.arrayBuffer()),
  ]);
  return new GnmModel(manifest, buffer);
}
