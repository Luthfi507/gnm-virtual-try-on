/**
 * Dual-mode Three.js viewer:
 *   1. 'studio': 3D Puppet Studio (stylized GNM mesh + bloom + wireframe + attached .obj)
 *   2. 'ar': AR Webcam Overlay (video background + GNM depth occluder + attached .obj)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

import { GpuEvaluator } from '../gnm/gpu.ts';
import type { GnmModel } from '../gnm/model.ts';
import {
  buildComponentPalettes,
  hexToNumber,
  STYLE_PRESETS,
  type CharacterStyle,
  type ComponentPalette,
} from './character.ts';
import type { ObjAccessoryManager } from './obj_manager.ts';

export type RenderMode = 'studio' | 'ar';

/** Face center in GNM bind-pose meters (correspondence centroid). */
const FACE_ANCHOR = new THREE.Vector3(0.0, 0.268, 0.09);
const AR_BASE_FOV_DEG = 46;

const SURFACE_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uPositions;
  uniform sampler2D uNormals;
  uniform vec2 uVertexSize;
  uniform vec3 uVertexOffset;

  attribute float aIndex;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;

  void main() {
    vec2 uv = (vec2(
      mod(aIndex, uVertexSize.x),
      floor(aIndex / uVertexSize.x)
    ) + 0.5) / uVertexSize;

    vec3 posed = texture2D(uPositions, uv).xyz + uVertexOffset;
    vec3 posedNormal = texture2D(uNormals, uv).xyz;

    vViewNormal = normalize(normalMatrix * posedNormal);
    vec4 viewPosition = modelViewMatrix * vec4(posed, 1.0);
    vViewPosition = -viewPosition.xyz;
    gl_Position = projectionMatrix * viewPosition;
  }
`;

const SURFACE_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uDeep;
  uniform vec3 uLit;
  uniform vec3 uFill;
  uniform float uFillStrength;
  uniform vec3 uRim;
  uniform float uRimPower;
  uniform float uRimStrength;
  uniform float uSpecular;
  uniform vec3 uEmissive;
  uniform float uEmissiveStrength;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;

  const vec3 KEY_DIRECTION = vec3(-0.55, 0.42, 0.72);
  const vec3 FILL_DIRECTION = vec3(0.86, -0.16, -0.30);

  void main() {
    vec3 normal = normalize(vViewNormal);
    vec3 toEye = normalize(vViewPosition);

    float sky = normal.y * 0.5 + 0.5;
    vec3 color = mix(uDeep * 0.4, uDeep, sky);

    vec3 key = normalize(KEY_DIRECTION);
    float diffuse = clamp(dot(normal, key), 0.0, 1.0);
    color += uLit * pow(diffuse, 1.9);

    float fill = clamp(dot(normal, normalize(FILL_DIRECTION)), 0.0, 1.0);
    color += uFill * pow(fill, 2.3) * uFillStrength;

    vec3 halfway = normalize(key + toEye);
    color += uLit * uSpecular * pow(clamp(dot(normal, halfway), 0.0, 1.0), 48.0);

    float fresnel = pow(1.0 - clamp(dot(normal, toEye), 0.0, 1.0), uRimPower);
    color += uRim * fresnel * uRimStrength;

    color += uEmissive * uEmissiveStrength;
    gl_FragColor = vec4(color, 1.0);
  }
`;

const WIREFRAME_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;

  void main() {
    vec3 normal = normalize(vViewNormal);
    vec3 toEye = normalize(vViewPosition);
    float facing = clamp(dot(normal, toEye), 0.0, 1.0);
    float weight = mix(0.35, 1.0, pow(1.0 - facing, 1.6));
    gl_FragColor = vec4(uColor * weight, uOpacity * weight);
  }
`;

const OCCLUDER_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uPositions;
  uniform vec2 uVertexSize;
  uniform vec3 uVertexOffset;

  attribute float aIndex;

  void main() {
    vec2 uv = (vec2(
      mod(aIndex, uVertexSize.x),
      floor(aIndex / uVertexSize.x)
    ) + 0.5) / uVertexSize;

    vec3 posed = texture2D(uPositions, uv).xyz + uVertexOffset;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(posed, 1.0);
  }
`;

const CORRESPONDENCE_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uPositions;
  uniform vec2 uVertexSize;
  uniform float uPointSize;

  attribute float aIndex;
  attribute vec3 aColor;

  varying vec3 vColor;

  void main() {
    vec2 uv = (vec2(
      mod(aIndex, uVertexSize.x),
      floor(aIndex / uVertexSize.x)
    ) + 0.5) / uVertexSize;

    vColor = aColor;
    vec4 viewPosition = modelViewMatrix * vec4(texture2D(uPositions, uv).xyz, 1.0);
    viewPosition.z += 0.0015;
    gl_PointSize = uPointSize / max(-viewPosition.z, 1e-4);
    gl_Position = projectionMatrix * viewPosition;
  }
`;

const CORRESPONDENCE_FRAGMENT_SHADER = /* glsl */ `
  varying vec3 vColor;

  void main() {
    vec2 offset = gl_PointCoord - 0.5;
    float radius = length(offset);
    if (radius > 0.5) discard;
    float alpha = smoothstep(0.5, 0.32, radius);
    gl_FragColor = vec4(vColor, alpha);
  }
`;

const FINISH_SHADER = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uTime: { value: 0 },
    uVignette: { value: 1.05 },
    uGrain: { value: 0.03 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uVignette;
    uniform float uGrain;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      vec2 offset = vUv - 0.5;
      float vignette = 1.0 - uVignette * dot(offset, offset);
      color.rgb *= clamp(vignette, 0.0, 1.0);
      float noise = fract(sin(dot(vUv * 1024.0 + uTime, vec2(12.9898, 78.233))) * 43758.5453);
      color.rgb += (noise - 0.5) * uGrain;
      gl_FragColor = color;
    }
  `,
};

interface SharedUniforms {
  uPositions: { value: THREE.Texture };
  uNormals: { value: THREE.Texture };
  uVertexSize: { value: THREE.Vector2 };
  uVertexOffset: { value: THREE.Vector3 };
}

function createSurfaceMaterial(palette: ComponentPalette, shared: SharedUniforms): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: SURFACE_VERTEX_SHADER,
    fragmentShader: SURFACE_FRAGMENT_SHADER,
    uniforms: {
      ...shared,
      uDeep: { value: new THREE.Color(palette.deep) },
      uLit: { value: new THREE.Color(palette.lit) },
      uFill: { value: new THREE.Color(palette.fill) },
      uFillStrength: { value: palette.fillStrength },
      uRim: { value: new THREE.Color(palette.rim) },
      uRimPower: { value: palette.rimPower },
      uRimStrength: { value: palette.rimStrength },
      uSpecular: { value: palette.specular },
      uEmissive: { value: new THREE.Color(palette.emissive) },
      uEmissiveStrength: { value: palette.emissiveStrength },
    },
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

function addLightsToScene(scene: THREE.Scene): void {
  const ambient = new THREE.AmbientLight(0xffffff, 0.75);
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(-1.2, 1.6, 2.2);
  const fill = new THREE.DirectionalLight(0xd8e8ff, 0.9);
  fill.position.set(1.6, -0.4, 1.0);
  const rim = new THREE.DirectionalLight(0xaef2e0, 0.8);
  rim.position.set(0.0, 1.2, -2.0);
  scene.add(ambient, key, fill, rim);
}

export class DualModeViewer {
  readonly gpu: GpuEvaluator;
  mode: RenderMode = 'studio';
  showOccluderDebug = false;
  arHeadScale = 1.0;
  arHeadOffset = new THREE.Vector3(0, 0, 0);

  private readonly canvas: HTMLCanvasElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly accessories: ObjAccessoryManager;
  private readonly headJointIndex: number;

  // --- Studio Mode ---
  private readonly studioScene = new THREE.Scene();
  private readonly studioCamera = new THREE.PerspectiveCamera(28, 1, 0.01, 100);
  private readonly controls: OrbitControls;
  private readonly composer: EffectComposer;
  private readonly renderPass: RenderPass;
  private readonly bloom: UnrealBloomPass;
  private readonly finish: ShaderPass;
  private readonly studioPivot = new THREE.Group();
  private readonly studioAccessoryAnchor = new THREE.Group();
  private readonly surfaceMaterials: THREE.ShaderMaterial[] = [];
  private readonly groupComponentIds: number[] = [];
  private readonly wireMaterial: THREE.ShaderMaterial;
  private readonly wire: THREE.LineSegments;
  private readonly correspondence: THREE.Points | null;

  // --- AR Mode ---
  private readonly arScene = new THREE.Scene();
  private readonly arCamera = new THREE.PerspectiveCamera(AR_BASE_FOV_DEG, 1, 0.01, 20);
  private readonly arBgScene = new THREE.Scene();
  private readonly arBgCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly arBgMaterial: THREE.ShaderMaterial;
  private readonly arHeadPivot = new THREE.Group();
  private readonly arOccluderMaterial: THREE.ShaderMaterial;
  private readonly arWireOverlay: THREE.LineSegments;
  private readonly arWireMaterial: THREE.ShaderMaterial;
  private videoTexture: THREE.VideoTexture | null = null;
  private videoElement: HTMLVideoElement | null = null;

  private readonly studioShared: SharedUniforms;
  private readonly arShared: SharedUniforms;
  private readonly faceAnchorMatrix = new THREE.Matrix4().makeTranslation(FACE_ANCHOR.x, FACE_ANCHOR.y, FACE_ANCHOR.z);
  private readonly tmpMatrix = new THREE.Matrix4();
  private viewportWidth = 1;
  private viewportHeight = 1;

  constructor(canvas: HTMLCanvasElement, model: GnmModel, accessories: ObjAccessoryManager) {
    this.canvas = canvas;
    this.accessories = accessories;
    this.headJointIndex = model.manifest.model.jointNames.indexOf('head');

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.gpu = new GpuEvaluator(this.renderer, model);

    const vertexCount = model.vertexCount;
    const templateAttr = new THREE.BufferAttribute(model.template, 3);
    const indexAttr = new THREE.BufferAttribute(Float32Array.from({ length: vertexCount }, (_, i) => i), 1);
    const { indices, groups } = model.groupTrianglesByComponent();

    const surfaceGeo = new THREE.BufferGeometry();
    surfaceGeo.setAttribute('position', templateAttr);
    surfaceGeo.setAttribute('aIndex', indexAttr);
    surfaceGeo.setIndex(new THREE.BufferAttribute(indices, 1));

    const wireGeo = new THREE.BufferGeometry();
    wireGeo.setAttribute('position', templateAttr);
    wireGeo.setAttribute('aIndex', indexAttr);
    wireGeo.setIndex(new THREE.BufferAttribute(model.quadEdges, 1));

    // --- Studio Scene Setup ---
    this.studioShared = {
      uPositions: { value: this.gpu.positionTexture },
      uNormals: { value: this.gpu.normalTexture },
      uVertexSize: { value: this.gpu.vertexSize },
      uVertexOffset: { value: new THREE.Vector3(0, 0, 0) },
    };

    const defaultStyle = STYLE_PRESETS.cyber_teal.style;
    const palettes = buildComponentPalettes(defaultStyle);
    this.surfaceMaterials = groups.map((group, slot) => {
      surfaceGeo.addGroup(group.start, group.count, slot);
      this.groupComponentIds.push(group.component);
      return createSurfaceMaterial(palettes[group.component], this.studioShared);
    });

    const studioMesh = new THREE.Mesh(surfaceGeo, this.surfaceMaterials);
    studioMesh.frustumCulled = false;
    this.studioPivot.add(studioMesh);

    this.wireMaterial = new THREE.ShaderMaterial({
      vertexShader: SURFACE_VERTEX_SHADER,
      fragmentShader: WIREFRAME_FRAGMENT_SHADER,
      uniforms: {
        ...this.studioShared,
        uColor: { value: new THREE.Color(defaultStyle.wireColor) },
        uOpacity: { value: defaultStyle.wireOpacity },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.wire = new THREE.LineSegments(wireGeo, this.wireMaterial);
    this.wire.frustumCulled = false;
    this.studioPivot.add(this.wire);

    this.correspondence = this.buildCorrespondencePoints(model, this.studioShared);
    if (this.correspondence) this.studioPivot.add(this.correspondence);

    this.studioAccessoryAnchor.matrixAutoUpdate = false;
    this.studioPivot.add(this.studioAccessoryAnchor);
    this.studioAccessoryAnchor.add(this.accessories.root);

    addLightsToScene(this.studioScene);
    this.studioScene.add(this.studioPivot);

    const bg = hexToNumber(defaultStyle.background);
    this.renderer.setClearColor(bg, 1);
    this.composer = new EffectComposer(this.renderer);
    this.renderPass = new RenderPass(this.studioScene, this.studioCamera, null, new THREE.Color(bg), 1);
    this.composer.addPass(this.renderPass);
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(1, 1),
      defaultStyle.bloomStrength,
      0.45,
      0.82,
    );
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.finish = new ShaderPass(FINISH_SHADER);
    this.composer.addPass(this.finish);

    this.controls = new OrbitControls(this.studioCamera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;

    // --- AR Scene Setup ---
    this.arShared = {
      uPositions: { value: this.gpu.positionTexture },
      uNormals: { value: this.gpu.normalTexture },
      uVertexSize: { value: this.gpu.vertexSize },
      uVertexOffset: { value: FACE_ANCHOR.clone().negate() },
    };

    this.arBgMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tVideo: { value: null },
        uUvScale: { value: new THREE.Vector2(1, 1) },
        uUvOffset: { value: new THREE.Vector2(0, 0) },
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D tVideo;
        uniform vec2 uUvScale;
        uniform vec2 uUvOffset;
        varying vec2 vUv;
        void main() {
          vec2 uv = vUv * uUvScale + uUvOffset;
          gl_FragColor = texture2D(tVideo, uv);
        }
      `,
      depthTest: false,
      depthWrite: false,
    });
    const bgQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.arBgMaterial);
    bgQuad.frustumCulled = false;
    this.arBgScene.add(bgQuad);

    this.arOccluderMaterial = new THREE.ShaderMaterial({
      vertexShader: OCCLUDER_VERTEX_SHADER,
      fragmentShader: /* glsl */ `void main() { gl_FragColor = vec4(0.0); }`,
      uniforms: {
        uPositions: this.arShared.uPositions,
        uVertexSize: this.arShared.uVertexSize,
        uVertexOffset: this.arShared.uVertexOffset,
      },
      colorWrite: false,
      depthWrite: true,
      depthTest: true,
    });
    const arOccluderMesh = new THREE.Mesh(surfaceGeo, this.arOccluderMaterial);
    arOccluderMesh.frustumCulled = false;
    arOccluderMesh.renderOrder = 1;
    this.arHeadPivot.add(arOccluderMesh);

    this.arWireMaterial = new THREE.ShaderMaterial({
      vertexShader: SURFACE_VERTEX_SHADER,
      fragmentShader: WIREFRAME_FRAGMENT_SHADER,
      uniforms: {
        ...this.arShared,
        uColor: { value: new THREE.Color(0x54c4a7) },
        uOpacity: { value: 0.45 },
      },
      transparent: true,
      depthWrite: false,
    });
    this.arWireOverlay = new THREE.LineSegments(wireGeo, this.arWireMaterial);
    this.arWireOverlay.frustumCulled = false;
    this.arWireOverlay.visible = false;
    this.arWireOverlay.renderOrder = 2;
    this.arHeadPivot.add(this.arWireOverlay);

    this.arHeadPivot.matrixAutoUpdate = false;
    this.arHeadPivot.visible = false;
    addLightsToScene(this.arScene);
    this.arScene.add(this.arHeadPivot);
  }

  private buildCorrespondencePoints(model: GnmModel, shared: SharedUniforms): THREE.Points | null {
    const mapping = model.correspondence;
    if (!mapping) return null;

    const count = mapping.vertices.length;
    const geometry = new THREE.BufferGeometry();
    const indices = new Float32Array(count);
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const v = mapping.vertices[i];
      indices[i] = v;
      positions[i * 3] = model.template[v * 3];
      positions[i * 3 + 1] = model.template[v * 3 + 1];
      positions[i * 3 + 2] = model.template[v * 3 + 2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aIndex', new THREE.BufferAttribute(indices, 1));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(mapping.colors, 3, true));

    const points = new THREE.Points(
      geometry,
      new THREE.ShaderMaterial({
        vertexShader: CORRESPONDENCE_VERTEX_SHADER,
        fragmentShader: CORRESPONDENCE_FRAGMENT_SHADER,
        uniforms: { ...shared, uPointSize: { value: 3.4 } },
        transparent: true,
        depthWrite: false,
      }),
    );
    points.frustumCulled = false;
    points.visible = false;
    return points;
  }

  attachVideo(video: HTMLVideoElement): void {
    this.videoElement = video;
    this.videoTexture = new THREE.VideoTexture(video);
    this.videoTexture.colorSpace = THREE.SRGBColorSpace;
    this.arBgMaterial.uniforms.tVideo.value = this.videoTexture;
  }

  setMode(mode: RenderMode, mirror = true): void {
    this.mode = mode;
    this.controls.enabled = mode === 'studio';
    if (mode === 'studio') {
      this.studioAccessoryAnchor.add(this.accessories.root);
      this.canvas.style.transform = '';
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    } else {
      this.arHeadPivot.add(this.accessories.root);
      this.canvas.style.transform = mirror ? 'scaleX(-1)' : '';
      this.renderer.toneMapping = THREE.NoToneMapping;
    }
  }

  setArMirror(mirror: boolean): void {
    if (this.mode === 'ar') {
      this.canvas.style.transform = mirror ? 'scaleX(-1)' : '';
    }
  }

  setCorrespondenceVisible(visible: boolean): void {
    if (this.correspondence) this.correspondence.visible = visible;
  }

  get hasCorrespondence(): boolean {
    return this.correspondence !== null;
  }

  applyStyle(style: CharacterStyle): void {
    const palettes = buildComponentPalettes(style);
    for (let i = 0; i < this.surfaceMaterials.length; i++) {
      const mat = this.surfaceMaterials[i];
      const pal = palettes[this.groupComponentIds[i]];
      (mat.uniforms.uDeep.value as THREE.Color).setHex(pal.deep);
      (mat.uniforms.uLit.value as THREE.Color).setHex(pal.lit);
      (mat.uniforms.uFill.value as THREE.Color).setHex(pal.fill);
      mat.uniforms.uFillStrength.value = pal.fillStrength;
      (mat.uniforms.uRim.value as THREE.Color).setHex(pal.rim);
      mat.uniforms.uRimPower.value = pal.rimPower;
      mat.uniforms.uRimStrength.value = pal.rimStrength;
      mat.uniforms.uSpecular.value = pal.specular;
      (mat.uniforms.uEmissive.value as THREE.Color).setHex(pal.emissive);
      mat.uniforms.uEmissiveStrength.value = pal.emissiveStrength;
    }

    (this.wireMaterial.uniforms.uColor.value as THREE.Color).set(style.wireColor);
    this.wireMaterial.uniforms.uOpacity.value = style.wireOpacity;
    this.wire.visible = style.wireVisible;
    this.bloom.strength = style.bloomStrength;

    const bg = hexToNumber(style.background);
    this.renderer.setClearColor(bg, 1);
    if (this.renderPass.clearColor) {
      this.renderPass.clearColor.setHex(bg);
    } else {
      this.renderPass.clearColor = new THREE.Color(bg);
    }
  }

  frame(vertices: Float32Array): void {
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    for (let i = 0; i < vertices.length; i += 3) {
      box.expandByPoint(point.set(vertices[i], vertices[i + 1], vertices[i + 2]));
    }
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    this.studioPivot.position.set(-centre.x, -centre.y, -centre.z);
    const extent = Math.max(size.x, size.y);
    const distance = (extent / (2 * Math.tan((this.studioCamera.fov * Math.PI) / 360))) * 1.65;
    this.studioCamera.position.set(0, 0, distance);
    this.studioCamera.near = distance / 100;
    this.studioCamera.far = distance * 10;
    this.studioCamera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  updateGnm(
    identity: ArrayLike<number>,
    expression: ArrayLike<number>,
    rotations: ArrayLike<number>,
    translation: ArrayLike<number>,
    correctives?: ArrayLike<number>,
  ): void {
    this.gpu.update(identity, expression, rotations, translation, correctives);
    const headMatrix = this.gpu.getJointMatrix(this.headJointIndex, this.tmpMatrix);
    this.studioAccessoryAnchor.matrix.copy(headMatrix).multiply(this.faceAnchorMatrix);
  }

  updateArPose(poseMatrixColMajor: Float32Array | null): void {
    if (!poseMatrixColMajor) {
      this.arHeadPivot.visible = false;
      return;
    }
    this.arHeadPivot.visible = true;
    const m = this.tmpMatrix.fromArray(poseMatrixColMajor);
    // MediaPipe translation is in cm; convert to meters
    m.elements[12] = m.elements[12] / 100.0 + this.arHeadOffset.x;
    m.elements[13] = m.elements[13] / 100.0 + this.arHeadOffset.y;
    m.elements[14] = m.elements[14] / 100.0 + this.arHeadOffset.z;
    if (this.arHeadScale !== 1.0) {
      m.scale(new THREE.Vector3(this.arHeadScale, this.arHeadScale, this.arHeadScale));
    }
    this.arHeadPivot.matrix.copy(m);
  }

  setSize(width: number, height: number): void {
    this.viewportWidth = Math.max(width, 1);
    this.viewportHeight = Math.max(height, 1);
    const aspect = this.viewportWidth / this.viewportHeight;

    this.studioCamera.aspect = aspect;
    this.studioCamera.updateProjectionMatrix();

    this.renderer.setSize(this.viewportWidth, this.viewportHeight, false);
    this.composer.setSize(this.viewportWidth, this.viewportHeight);
    this.bloom.resolution.set(this.viewportWidth, this.viewportHeight);
  }

  private syncArProjection(): void {
    const canvasAspect = this.viewportWidth / this.viewportHeight;
    const video = this.videoElement;
    const vw = video?.videoWidth || 1280;
    const vh = video?.videoHeight || 720;
    const videoAspect = vw / vh;

    const uvScale = this.arBgMaterial.uniforms.uUvScale.value as THREE.Vector2;
    const uvOffset = this.arBgMaterial.uniforms.uUvOffset.value as THREE.Vector2;

    const baseHalfTan = Math.tan(THREE.MathUtils.degToRad(AR_BASE_FOV_DEG / 2));
    if (canvasAspect < videoAspect) {
      const sx = canvasAspect / videoAspect;
      uvScale.set(sx, 1);
      uvOffset.set((1 - sx) / 2, 0);
      this.arCamera.fov = AR_BASE_FOV_DEG;
    } else {
      const sy = videoAspect / canvasAspect;
      uvScale.set(1, sy);
      uvOffset.set(0, (1 - sy) / 2);
      this.arCamera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(baseHalfTan * sy));
    }
    this.arCamera.aspect = canvasAspect;
    this.arCamera.updateProjectionMatrix();
  }

  render(elapsedSeconds: number): void {
    if (this.mode === 'studio') {
      this.finish.uniforms.uTime.value = elapsedSeconds;
      this.controls.update();
      this.composer.render();
      return;
    }

    this.syncArProjection();
    this.arWireOverlay.visible = this.showOccluderDebug;

    const prevAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = false;
    this.renderer.clear();

    if (this.videoElement && this.videoElement.readyState >= 2) {
      this.renderer.render(this.arBgScene, this.arBgCamera);
    }
    this.renderer.clearDepth();
    this.renderer.render(this.arScene, this.arCamera);
    this.renderer.autoClear = prevAutoClear;
  }
}
