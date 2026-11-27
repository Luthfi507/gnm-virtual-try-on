/**
 * Two-pass WebGL GNM forward pass (positions -> smooth area-weighted normals).
 */

import * as THREE from 'three';
import type { GnmModel } from './model.ts';

const VERTEX_TEXTURE_WIDTH = 512;
const BASIS_TEXTURE_WIDTH = 2048;

const TEXEL_LOOKUP = /* glsl */ `
  vec2 texelUv(float index, vec2 size) {
    float column = mod(index, size.x);
    float row = floor(index / size.x);
    return (vec2(column, row) + 0.5) / size;
  }
`;

function textureSize(count: number, width: number): THREE.Vector2 {
  return new THREE.Vector2(width, Math.ceil(count / width));
}

function packRgba<T extends Float32Array | Uint16Array>(
  source: T,
  texelCount: number,
  make: (length: number) => T,
  alpha?: Float32Array,
): T {
  const out = make(texelCount * 4);
  const triples = Math.min(texelCount, Math.floor(source.length / 3));
  for (let i = 0; i < triples; i++) {
    out[i * 4] = source[i * 3];
    out[i * 4 + 1] = source[i * 3 + 1];
    out[i * 4 + 2] = source[i * 3 + 2];
  }
  if (alpha) {
    for (let i = 0; i < triples; i++) out[i * 4 + 3] = alpha[i] as never;
  }
  return out;
}

function dataTexture(
  data: Float32Array | Uint16Array,
  size: THREE.Vector2,
  type: THREE.TextureDataType,
): THREE.DataTexture {
  const texture = new THREE.DataTexture(data, size.x, size.y, THREE.RGBAFormat, type);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

function renderTarget(size: THREE.Vector2): THREE.WebGLRenderTarget {
  return new THREE.WebGLRenderTarget(size.x, size.y, {
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
  });
}

export class GpuEvaluator {
  readonly vertexSize: THREE.Vector2;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly model: GnmModel;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly quad: THREE.Mesh;

  private readonly positionMaterial: THREE.ShaderMaterial;
  private readonly normalMaterial: THREE.ShaderMaterial;
  private readonly positionTarget: THREE.WebGLRenderTarget;
  private readonly normalTarget: THREE.WebGLRenderTarget;
  private readonly jointMatrices: THREE.Matrix4[];
  private lastJointTransforms: Float32Array | null = null;

  constructor(renderer: THREE.WebGLRenderer, model: GnmModel) {
    this.renderer = renderer;
    this.model = model;

    const vertexCount = model.vertexCount;
    this.vertexSize = textureSize(vertexCount, VERTEX_TEXTURE_WIDTH);
    const texels = this.vertexSize.x * this.vertexSize.y;

    const counts = new Float32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) counts[i] = model.normalAdjacencyCount[i];
    const templateTexture = dataTexture(
      packRgba(model.template, texels, (n) => new Float32Array(n), counts),
      this.vertexSize,
      THREE.FloatType,
    );

    const skinning = new Float32Array(texels * 4);
    for (let j = 0; j < model.jointCount; j++) {
      for (let v = 0; v < vertexCount; v++) {
        skinning[v * 4 + j] = model.skinningWeights[j * vertexCount + v];
      }
    }
    const skinningTexture = dataTexture(skinning, this.vertexSize, THREE.FloatType);

    const identityBits = new Uint16Array(
      model.buffer,
      model.manifest.arrays.identity_basis.byteOffset,
      model.manifest.arrays.identity_basis.byteLength / 2,
    );
    const expressionBits = new Uint16Array(
      model.buffer,
      model.manifest.arrays.expression_basis.byteOffset,
      model.manifest.arrays.expression_basis.byteLength / 2,
    );
    const identitySize = textureSize(model.identityDim * vertexCount, BASIS_TEXTURE_WIDTH);
    const expressionSize = textureSize(model.expressionDim * vertexCount, BASIS_TEXTURE_WIDTH);
    const identityTexture = dataTexture(
      packRgba(identityBits, identitySize.x * identitySize.y, (n) => new Uint16Array(n)),
      identitySize,
      THREE.HalfFloatType,
    );
    const expressionTexture = dataTexture(
      packRgba(expressionBits, expressionSize.x * expressionSize.y, (n) => new Uint16Array(n)),
      expressionSize,
      THREE.HalfFloatType,
    );

    const correctiveSlots = Math.max(model.correctiveDim, 1);
    const correctiveBits = model.correctiveDim
      ? new Uint16Array(
          model.buffer,
          model.manifest.arrays.corrective_basis.byteOffset,
          model.manifest.arrays.corrective_basis.byteLength / 2,
        )
      : new Uint16Array(vertexCount * 3);
    const correctiveSize = textureSize(correctiveSlots * vertexCount, BASIS_TEXTURE_WIDTH);
    const correctiveTexture = dataTexture(
      packRgba(correctiveBits, correctiveSize.x * correctiveSize.y, (n) => new Uint16Array(n)),
      correctiveSize,
      THREE.HalfFloatType,
    );

    const corners = model.maxNormalCorners;
    const adjacencySize = textureSize(vertexCount * corners, BASIS_TEXTURE_WIDTH);
    const adjacency = new Float32Array(adjacencySize.x * adjacencySize.y * 4);
    for (let i = 0; i < vertexCount * corners; i++) {
      adjacency[i * 4] = model.normalAdjacency[i * 2];
      adjacency[i * 4 + 1] = model.normalAdjacency[i * 2 + 1];
    }
    const adjacencyTexture = dataTexture(adjacency, adjacencySize, THREE.FloatType);

    this.positionTarget = renderTarget(this.vertexSize);
    this.normalTarget = renderTarget(this.vertexSize);
    this.jointMatrices = Array.from({ length: model.jointCount }, () => new THREE.Matrix4());

    this.positionMaterial = new THREE.ShaderMaterial({
      defines: {
        IDENTITY_COUNT: model.identityDim,
        EXPRESSION_COUNT: model.expressionDim,
        CORRECTIVE_COUNT: correctiveSlots,
        JOINT_COUNT: model.jointCount,
      },
      uniforms: {
        uTemplate: { value: templateTexture },
        uIdentityBasis: { value: identityTexture },
        uExpressionBasis: { value: expressionTexture },
        uCorrectiveBasis: { value: correctiveTexture },
        uSkinning: { value: skinningTexture },
        uIdentity: { value: new Float32Array(model.identityDim) },
        uExpression: { value: new Float32Array(model.expressionDim) },
        uCorrective: { value: new Float32Array(correctiveSlots) },
        uJoints: { value: this.jointMatrices },
        uVertexSize: { value: this.vertexSize },
        uIdentitySize: { value: identitySize },
        uExpressionSize: { value: expressionSize },
        uCorrectiveSize: { value: correctiveSize },
        uVertexCount: { value: vertexCount },
      },
      vertexShader: /* glsl */ `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D uTemplate;
        uniform sampler2D uIdentityBasis;
        uniform sampler2D uExpressionBasis;
        uniform sampler2D uCorrectiveBasis;
        uniform sampler2D uSkinning;
        uniform float uIdentity[IDENTITY_COUNT];
        uniform float uExpression[EXPRESSION_COUNT];
        uniform float uCorrective[CORRECTIVE_COUNT];
        uniform mat4 uJoints[JOINT_COUNT];
        uniform vec2 uVertexSize;
        uniform vec2 uIdentitySize;
        uniform vec2 uExpressionSize;
        uniform vec2 uCorrectiveSize;
        uniform float uVertexCount;
        ${TEXEL_LOOKUP}

        void main() {
          vec2 texel = floor(gl_FragCoord.xy);
          float index = texel.y * uVertexSize.x + texel.x;
          vec2 uv = (texel + 0.5) / uVertexSize;
          vec3 vertex = texture2D(uTemplate, uv).xyz;

          for (int i = 0; i < IDENTITY_COUNT; i++) {
            vertex += uIdentity[i] * texture2D(
              uIdentityBasis,
              texelUv(float(i) * uVertexCount + index, uIdentitySize)
            ).xyz;
          }
          for (int i = 0; i < EXPRESSION_COUNT; i++) {
            vertex += uExpression[i] * texture2D(
              uExpressionBasis,
              texelUv(float(i) * uVertexCount + index, uExpressionSize)
            ).xyz;
          }
          for (int i = 0; i < CORRECTIVE_COUNT; i++) {
            vertex += uCorrective[i] * texture2D(
              uCorrectiveBasis,
              texelUv(float(i) * uVertexCount + index, uCorrectiveSize)
            ).xyz;
          }

          vec4 weights = texture2D(uSkinning, uv);
          vec4 homogeneous = vec4(vertex, 1.0);
          vec3 posed = vec3(0.0);
          for (int j = 0; j < JOINT_COUNT; j++) {
            posed += weights[j] * (uJoints[j] * homogeneous).xyz;
          }
          gl_FragColor = vec4(posed, 1.0);
        }
      `,
    });

    this.normalMaterial = new THREE.ShaderMaterial({
      defines: { MAX_CORNERS: corners },
      uniforms: {
        uPositions: { value: this.positionTarget.texture },
        uAdjacency: { value: adjacencyTexture },
        uTemplate: { value: templateTexture },
        uVertexSize: { value: this.vertexSize },
        uAdjacencySize: { value: adjacencySize },
      },
      vertexShader: /* glsl */ `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D uPositions;
        uniform sampler2D uAdjacency;
        uniform sampler2D uTemplate;
        uniform vec2 uVertexSize;
        uniform vec2 uAdjacencySize;
        ${TEXEL_LOOKUP}

        void main() {
          vec2 texel = floor(gl_FragCoord.xy);
          float index = texel.y * uVertexSize.x + texel.x;
          vec2 uv = (texel + 0.5) / uVertexSize;

          vec3 origin = texture2D(uPositions, uv).xyz;
          float corners = texture2D(uTemplate, uv).w;

          vec3 accumulated = vec3(0.0);
          for (int k = 0; k < MAX_CORNERS; k++) {
            if (float(k) >= corners) break;
            vec2 pair = texture2D(
              uAdjacency,
              texelUv(index * float(MAX_CORNERS) + float(k), uAdjacencySize)
            ).xy;
            vec3 first = texture2D(uPositions, texelUv(pair.x, uVertexSize)).xyz;
            vec3 second = texture2D(uPositions, texelUv(pair.y, uVertexSize)).xyz;
            accumulated += cross(first - origin, second - origin);
          }

          float len = max(length(accumulated), 1e-12);
          gl_FragColor = vec4(accumulated / len, 1.0);
        }
      `,
    });

    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.positionMaterial);
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
  }

  get positionTexture(): THREE.Texture {
    return this.positionTarget.texture;
  }

  get normalTexture(): THREE.Texture {
    return this.normalTarget.texture;
  }

  getJointMatrix(jointIndex: number, target = new THREE.Matrix4()): THREE.Matrix4 {
    if (!this.lastJointTransforms || jointIndex < 0 || jointIndex >= this.model.jointCount) {
      return target.identity();
    }
    return target.copy(this.jointMatrices[jointIndex]);
  }

  update(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
    correctives?: ArrayLike<number>,
  ): void {
    const transforms = this.model.jointTransforms(identity, rotations, translation);
    this.lastJointTransforms = transforms;
    for (let j = 0; j < this.model.jointCount; j++) {
      const o = j * 12;
      this.jointMatrices[j].set(
        transforms[o], transforms[o + 1], transforms[o + 2], transforms[o + 3],
        transforms[o + 4], transforms[o + 5], transforms[o + 6], transforms[o + 7],
        transforms[o + 8], transforms[o + 9], transforms[o + 10], transforms[o + 11],
        0, 0, 0, 1,
      );
    }

    (this.positionMaterial.uniforms.uIdentity.value as Float32Array).set(identity as ArrayLike<number> as never);
    (this.positionMaterial.uniforms.uExpression.value as Float32Array).set(expression as ArrayLike<number> as never);
    const correctiveWeights = this.positionMaterial.uniforms.uCorrective.value as Float32Array;
    correctiveWeights.fill(0);
    if (correctives) correctiveWeights.set(correctives as ArrayLike<number> as never, 0);

    const previousTarget = this.renderer.getRenderTarget();
    const previousAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = false;

    this.quad.material = this.positionMaterial;
    this.renderer.setRenderTarget(this.positionTarget);
    this.renderer.render(this.scene, this.camera);

    this.quad.material = this.normalMaterial;
    this.renderer.setRenderTarget(this.normalTarget);
    this.renderer.render(this.scene, this.camera);

    this.renderer.setRenderTarget(previousTarget);
    this.renderer.autoClear = previousAutoClear;
  }
}
