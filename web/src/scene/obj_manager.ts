/**
 * Loads, normalizes, and manages multiple .obj (+ optional .mtl / texture) accessories
 * attached to the face-centered head coordinate frame.
 */

import * as THREE from 'three';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

export interface ObjTransform {
  position: [number, number, number];
  rotationDeg: [number, number, number];
  scale: number;
}

export interface AttachedObjItem {
  id: string;
  name: string;
  source: string;
  visible: boolean;
  tintColor: string;
  useTint: boolean;
  transform: ObjTransform;
  defaultTransform: ObjTransform;
  group: THREE.Group;
}

export interface ObjPresetSpec {
  id: string;
  name: string;
  objUrl: string;
  mtlUrl: string;
  defaultTransform: ObjTransform;
}

export const OBJ_PRESETS: ObjPresetSpec[] = [
  {
    id: 'round_glasses',
    name: 'Round Glasses',
    objUrl: 'assets/objects/round_glasses.obj',
    mtlUrl: 'assets/objects/round_glasses.mtl',
    defaultTransform: { position: [0, 0.032, 0.036], rotationDeg: [0, 0, 0], scale: 1.0 },
  },
  {
    id: 'cyber_visor',
    name: 'Cyber Visor',
    objUrl: 'assets/objects/cyber_visor.obj',
    mtlUrl: 'assets/objects/cyber_visor.mtl',
    defaultTransform: { position: [0, 0.034, 0.036], rotationDeg: [0, 0, 0], scale: 1.0 },
  },
  {
    id: 'royal_crown',
    name: 'Royal Crown',
    objUrl: 'assets/objects/royal_crown.obj',
    mtlUrl: 'assets/objects/royal_crown.mtl',
    defaultTransform: { position: [0, 0.132, -0.024], rotationDeg: [-5, 0, 0], scale: 1.0 },
  },
  {
    id: 'top_hat',
    name: 'Top Hat',
    objUrl: 'assets/objects/top_hat.obj',
    mtlUrl: 'assets/objects/top_hat.mtl',
    defaultTransform: { position: [0, 0.135, -0.030], rotationDeg: [-6, 0, 0], scale: 1.0 },
  },
  {
    id: 'cat_ears',
    name: 'Cat Ears',
    objUrl: 'assets/objects/cat_ears.obj',
    mtlUrl: 'assets/objects/cat_ears.mtl',
    defaultTransform: { position: [0, 0.072, -0.022], rotationDeg: [0, 0, 0], scale: 1.0 },
  },
];

function cloneTransform(t: ObjTransform): ObjTransform {
  return {
    position: [...t.position],
    rotationDeg: [...t.rotationDeg],
    scale: t.scale,
  };
}

function normalizeObjectToHeadScale(object: THREE.Object3D, targetExtent = 0.16): void {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-5);
  const scale = targetExtent / maxDim;

  object.position.sub(center).multiplyScalar(scale);
  object.scale.multiplyScalar(scale);
}

function ensureCleanMaterials(root: THREE.Object3D): void {
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    if (!child.geometry.attributes.normal) child.geometry.computeVertexNormals();
    child.renderOrder = 2;

    const mats = Array.isArray(child.material) ? child.material : [child.material];
    const upgraded = mats.map((mat) => {
      if (!mat) {
        return new THREE.MeshStandardMaterial({ color: 0xcfd8dc, roughness: 0.4, metalness: 0.2 });
      }
      if (mat instanceof THREE.MeshPhongMaterial) {
        const std = new THREE.MeshStandardMaterial({
          color: mat.color.clone(),
          map: mat.map,
          roughness: 0.35,
          metalness: Math.min(0.85, (mat.specular.r + mat.specular.g + mat.specular.b) / 3),
          transparent: mat.transparent,
          opacity: mat.opacity,
          side: THREE.DoubleSide,
        });
        mat.dispose();
        return std;
      }
      mat.side = THREE.DoubleSide;
      return mat;
    });
    child.material = Array.isArray(child.material) ? upgraded : upgraded[0];
  });
}

export class ObjAccessoryManager {
  readonly root = new THREE.Group();
  readonly items: AttachedObjItem[] = [];
  private nextId = 1;

  async addPreset(preset: ObjPresetSpec): Promise<AttachedObjItem> {
    const mtlLoader = new MTLLoader();
    const materials = await mtlLoader.loadAsync(preset.mtlUrl);
    materials.preload();

    const objLoader = new OBJLoader();
    objLoader.setMaterials(materials);
    const mesh = await objLoader.loadAsync(preset.objUrl);
    ensureCleanMaterials(mesh);

    return this.registerItem(preset.name, `preset:${preset.id}`, mesh, preset.defaultTransform);
  }

  async addFromFiles(files: FileList | File[]): Promise<AttachedObjItem[]> {
    const fileArray = Array.from(files);
    const objFiles = fileArray.filter((f) => f.name.toLowerCase().endsWith('.obj'));
    if (objFiles.length === 0) {
      throw new Error('Pilih minimal satu file .obj');
    }

    const blobUrls = new Map<string, string>();
    for (const file of fileArray) {
      blobUrls.set(file.name.toLowerCase(), URL.createObjectURL(file));
    }

    const manager = new THREE.LoadingManager();
    manager.setURLModifier((url) => {
      const base = url.split('/').pop()?.split('\\').pop()?.toLowerCase() ?? '';
      return blobUrls.get(base) ?? url;
    });

    const mtlFile = fileArray.find((f) => f.name.toLowerCase().endsWith('.mtl'));
    let preloadedMaterials: MTLLoader.MaterialCreator | null = null;
    if (mtlFile) {
      const mtlLoader = new MTLLoader(manager);
      const mtlText = await mtlFile.text();
      preloadedMaterials = mtlLoader.parse(mtlText, '');
      preloadedMaterials.preload();
    }

    const added: AttachedObjItem[] = [];
    for (const objFile of objFiles) {
      const objLoader = new OBJLoader(manager);
      if (preloadedMaterials) objLoader.setMaterials(preloadedMaterials);
      const text = await objFile.text();
      const mesh = objLoader.parse(text);
      normalizeObjectToHeadScale(mesh, 0.16);
      ensureCleanMaterials(mesh);

      const cleanName = objFile.name.replace(/\.obj$/i, '');
      const item = this.registerItem(
        cleanName,
        `upload:${objFile.name}`,
        mesh,
        { position: [0, 0.03, 0.04], rotationDeg: [0, 0, 0], scale: 1.0 },
      );
      added.push(item);
    }

    for (const url of blobUrls.values()) URL.revokeObjectURL(url);
    return added;
  }

  private registerItem(
    name: string,
    source: string,
    object: THREE.Object3D,
    defaultTransform: ObjTransform,
  ): AttachedObjItem {
    const group = new THREE.Group();
    group.add(object);
    this.root.add(group);

    const item: AttachedObjItem = {
      id: `obj-${this.nextId++}`,
      name,
      source,
      visible: true,
      tintColor: '#54c4a7',
      useTint: false,
      transform: cloneTransform(defaultTransform),
      defaultTransform: cloneTransform(defaultTransform),
      group,
    };
    this.applyTransform(item);
    this.items.push(item);
    return item;
  }

  applyTransform(item: AttachedObjItem): void {
    const { position, rotationDeg, scale } = item.transform;
    item.group.position.set(position[0], position[1], position[2]);
    item.group.rotation.set(
      THREE.MathUtils.degToRad(rotationDeg[0]),
      THREE.MathUtils.degToRad(rotationDeg[1]),
      THREE.MathUtils.degToRad(rotationDeg[2]),
    );
    item.group.scale.setScalar(scale);
    item.group.visible = item.visible;
  }

  applyTint(item: AttachedObjItem): void {
    const color = new THREE.Color(item.tintColor);
    item.group.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const mat of mats) {
        if (mat instanceof THREE.MeshStandardMaterial) {
          if (!mat.userData.originalColor) {
            mat.userData.originalColor = mat.color.clone();
          }
          mat.color.copy(item.useTint ? color : (mat.userData.originalColor as THREE.Color));
        }
      }
    });
  }

  resetTransform(item: AttachedObjItem): void {
    item.transform = cloneTransform(item.defaultTransform);
    this.applyTransform(item);
  }

  removeItem(id: string): void {
    const index = this.items.findIndex((x) => x.id === id);
    if (index < 0) return;
    const [removed] = this.items.splice(index, 1);
    this.root.remove(removed.group);
    removed.group.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of mats) m.dispose();
      }
    });
  }

  clearAll(): void {
    for (const item of [...this.items]) this.removeItem(item.id);
  }
}
