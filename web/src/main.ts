/**
 * Main application orchestrating:
 * - Dual-mode rendering: 3D Puppet Studio & AR Webcam Try-On
 * - Multi-.obj accessory loading, upload, and live 6-DoF adjustment
 * - Option A Character Customizer (semantic presets, identity sliders, webcam fit, shader/material studio)
 * - Real-time MediaPipe face tracking + GNM GPU forward pass
 */

import { fitIdentity } from './gnm/fit.ts';
import { loadGnmModel } from './gnm/model.ts';
import { FaceRetargeter } from './gnm/retarget.ts';
import {
  STYLE_PRESETS,
  type CharacterIdentityPreset,
  type CharacterStyle,
} from './scene/character.ts';
import {
  OBJ_PRESETS,
  ObjAccessoryManager,
  type AttachedObjItem,
} from './scene/obj_manager.ts';
import { DualModeViewer, type RenderMode } from './scene/viewer.ts';
import { CameraController } from './tracking/camera.ts';

interface AppState {
  identity: Float32Array;
  expression: Float32Array;
  rotations: Float32Array;
  translation: Float32Array;
  correctives: Float32Array;
  style: CharacterStyle;
}

const SLIDER_BUDGET: Record<string, number> = {
  lower_face_region: 6,
  left_eye_region: 3,
  right_eye_region: 3,
};
const IDENTITY_SLIDER_COUNT = 12;
const CORRECTIVE_ALPHA = 0.35;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addSlider(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  decimals: number,
  onInput: (val: number) => void,
): { input: HTMLInputElement; setDisplay: (val: number) => void } {
  const row = el('div', 'row');
  const name = el('label', 'name', label);
  const readout = el('span', 'value', value.toFixed(decimals));
  const input = el('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);

  const setDisplay = (v: number) => {
    input.value = String(v);
    readout.textContent = v.toFixed(decimals);
  };

  input.addEventListener('input', () => {
    const next = Number(input.value);
    readout.textContent = next.toFixed(decimals);
    onInput(next);
  });

  row.append(name, input, readout);
  parent.append(row);
  return { input, setDisplay };
}

async function loadIdentityPresets(): Promise<CharacterIdentityPreset[]> {
  try {
    const res = await fetch('assets/character_presets.json');
    if (!res.ok) return [];
    const data = (await res.json()) as { presets?: CharacterIdentityPreset[] };
    return data.presets ?? [];
  } catch {
    return [];
  }
}

function setupTabs(): void {
  const buttons = document.querySelectorAll<HTMLButtonElement>('.tab-btn');
  const panes = document.querySelectorAll<HTMLElement>('.tab-pane');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      buttons.forEach((b) => {
        const active = b === btn;
        b.classList.toggle('active', active);
        b.setAttribute('aria-selected', String(active));
      });
      panes.forEach((p) => p.classList.toggle('active', p.id === `tab-${target}`));
    });
  });
}

async function main(): Promise<void> {
  setupTabs();

  const statusEl = document.querySelector<HTMLElement>('#status')!;
  const costEl = document.querySelector<HTMLElement>('#cost')!;
  const verifyEl = document.querySelector<HTMLElement>('#verify')!;
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
  const pipCanvas = document.querySelector<HTMLCanvasElement>('#pip')!;
  const pipStatus = document.querySelector<HTMLElement>('#pipStatus')!;

  statusEl.textContent = 'Loading GNM assets (18 MB)...';
  const gnmAssetsBase = import.meta.env.VITE_GNM_ASSETS_URL?.trim() || 'assets';
  const [model, identityPresets] = await Promise.all([
    loadGnmModel(gnmAssetsBase),
    loadIdentityPresets(),
  ]);

  const accessories = new ObjAccessoryManager();
  const viewer = new DualModeViewer(canvas, model, accessories);
  const camera = new CameraController(pipCanvas, 175);
  viewer.attachVideo(camera.video);

  const state: AppState = {
    identity: new Float32Array(model.identityDim),
    expression: new Float32Array(model.expressionDim),
    rotations: new Float32Array(model.jointCount * 3),
    translation: new Float32Array(3),
    correctives: new Float32Array(model.correctiveDim),
    style: { ...STYLE_PRESETS.cyber_teal.style },
  };

  let dirty = true;
  const invalidate = () => {
    dirty = true;
  };

  const initialVerts = new Float32Array(model.vertexCount * 3);
  model.evaluate(state.identity, state.expression, state.rotations, state.translation, initialVerts, state.correctives);
  viewer.updateGnm(state.identity, state.expression, state.rotations, state.translation, state.correctives);
  viewer.frame(initialVerts);
  viewer.applyStyle(state.style);

  const retargeter = new FaceRetargeter(model);
  const headJoint = model.manifest.model.jointNames.indexOf('head');
  let driving = false;
  let mirror = true;

  // --- Shared callbacks for UI synchronization ---
  const identitySliderUpdaters: ((v: number) => void)[] = [];
  const poseResetters: (() => void)[] = [];
  let selectedItemId: string | null = null;
  let refreshObjectsUi = () => {};

  const syncIdentitySliders = () => {
    identitySliderUpdaters.forEach((fn, i) => fn(state.identity[i] ?? 0));
  };

  const applyIdentityVector = (vec: ArrayLike<number>) => {
    state.identity.fill(0);
    const n = Math.min(state.identity.length, vec.length);
    for (let i = 0; i < n; i++) state.identity[i] = vec[i];
    syncIdentitySliders();
    retargeter.setIdentity(state.identity);
    invalidate();
  };

  // --- Top Toolbar Controls ---
  const modeStudioBtn = document.querySelector<HTMLButtonElement>('#modeStudio')!;
  const modeArBtn = document.querySelector<HTMLButtonElement>('#modeAr')!;
  const btnCamera = document.querySelector<HTMLButtonElement>('#btnCamera')!;
  const btnFit = document.querySelector<HTMLButtonElement>('#btnFit')!;
  const btnDrive = document.querySelector<HTMLButtonElement>('#btnDrive')!;
  const btnMirror = document.querySelector<HTMLButtonElement>('#btnMirror')!;

  const ensureCameraRunning = async (): Promise<boolean> => {
    if (camera.isRunning) return true;
    btnCamera.disabled = true;
    btnCamera.textContent = 'Starting...';
    try {
      await camera.start();
      pipCanvas.classList.add('live');
      btnCamera.classList.add('active');
      btnCamera.textContent = 'Stop Camera';
      return true;
    } catch (err) {
      verifyEl.textContent = `Camera error: ${String(err)}`;
      btnCamera.textContent = 'Start Camera';
      return false;
    } finally {
      btnCamera.disabled = false;
    }
  };

  const stopDriving = () => {
    driving = false;
    btnDrive.classList.remove('active');
    btnDrive.textContent = 'Drive Face';
    state.expression.fill(0);
    state.rotations.fill(0);
    state.correctives.fill(0);
    poseResetters.forEach((r) => r());
    invalidate();
  };

  const switchMode = async (mode: RenderMode) => {
    modeStudioBtn.classList.toggle('active', mode === 'studio');
    modeArBtn.classList.toggle('active', mode === 'ar');
    viewer.setMode(mode, mirror);
    camera.drawPip = mode === 'studio';
    pipCanvas.style.display = mode === 'studio' ? 'block' : 'none';
    pipStatus.style.display = mode === 'studio' ? 'block' : 'none';

    if (mode === 'ar') {
      const ok = await ensureCameraRunning();
      if (ok && !driving) {
        driving = true;
        retargeter.reset();
        btnDrive.classList.add('active');
        btnDrive.textContent = 'Stop Driving';
      }
    }
    invalidate();
  };

  modeStudioBtn.addEventListener('click', () => void switchMode('studio'));
  modeArBtn.addEventListener('click', () => void switchMode('ar'));

  btnCamera.addEventListener('click', async () => {
    if (camera.isRunning) {
      if (driving) stopDriving();
      camera.stop();
      pipCanvas.classList.remove('live');
      pipStatus.textContent = '';
      btnCamera.classList.remove('active');
      btnCamera.textContent = 'Start Camera';
      if (viewer.mode === 'ar') void switchMode('studio');
      return;
    }
    await ensureCameraRunning();
  });

  btnMirror.addEventListener('click', () => {
    mirror = !mirror;
    camera.mirror = mirror;
    viewer.setArMirror(mirror);
    btnMirror.classList.toggle('active', mirror);
  });

  btnFit.addEventListener('click', () => {
    const frame = camera.latest;
    if (!frame) {
      verifyEl.textContent = camera.isRunning
        ? 'Fit Identity: wajah belum terdeteksi'
        : 'Fit Identity: nyalakan kamera dulu';
      return;
    }
    const t0 = performance.now();
    const fit = fitIdentity(model, frame.landmarks, camera.aspect);
    applyIdentityVector(fit.identity);
    verifyEl.textContent =
      `Identity fit: ${fit.rmsBefore.toFixed(1)} -> ${fit.rmsAfter.toFixed(1)} mm (${(performance.now() - t0).toFixed(0)} ms)`;
  });

  btnDrive.addEventListener('click', async () => {
    if (driving) {
      stopDriving();
      return;
    }
    const ok = await ensureCameraRunning();
    if (!ok) return;
    driving = true;
    retargeter.reset();
    btnDrive.classList.add('active');
    btnDrive.textContent = 'Stop Driving';
  });

  // =========================================================================
  // TAB 1: .OBJ TRY-ON
  // =========================================================================
  const tabObjects = document.querySelector<HTMLElement>('#tab-objects')!;

  tabObjects.append(el('h2', undefined, 'Preset .OBJ Objects'));
  const presetGroup = el('div', 'group');
  const presetBtns = el('div', 'buttons');
  for (const preset of OBJ_PRESETS) {
    const btn = el('button', 'btn', `+ ${preset.name}`);
    btn.addEventListener('click', async () => {
      try {
        const item = await accessories.addPreset(preset);
        selectedItemId = item.id;
        refreshObjectsUi();
      } catch (err) {
        verifyEl.textContent = `Load OBJ failed: ${String(err)}`;
      }
    });
    presetBtns.append(btn);
  }
  presetGroup.append(presetBtns);
  tabObjects.append(presetGroup);

  tabObjects.append(el('h2', undefined, 'Upload Custom .OBJ (+ .MTL / Texture)'));
  const uploadGroup = el('div', 'group');
  const fileInput = el('input');
  fileInput.type = 'file';
  fileInput.multiple = true;
  fileInput.accept = '.obj,.mtl,.png,.jpg,.jpeg,.webp';
  fileInput.style.display = 'none';

  const dropzone = el(
    'div',
    'dropzone',
    'Klik atau drag-and-drop file .obj di sini (boleh pilih bersamaan dengan .mtl & gambar tekstur)',
  );
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
    if (!e.dataTransfer?.files.length) return;
    try {
      const items = await accessories.addFromFiles(e.dataTransfer.files);
      if (items.length) selectedItemId = items[items.length - 1].id;
      refreshObjectsUi();
    } catch (err) {
      verifyEl.textContent = String(err);
    }
  });
  fileInput.addEventListener('change', async () => {
    if (!fileInput.files?.length) return;
    try {
      const items = await accessories.addFromFiles(fileInput.files);
      if (items.length) selectedItemId = items[items.length - 1].id;
      refreshObjectsUi();
    } catch (err) {
      verifyEl.textContent = String(err);
    } finally {
      fileInput.value = '';
    }
  });
  uploadGroup.append(fileInput, dropzone);
  tabObjects.append(uploadGroup);

  tabObjects.append(el('h2', undefined, 'Attached Objects'));
  const listGroup = el('div', 'group');
  const objListEl = el('div', 'obj-list');
  const clearBtn = el('button', 'btn', 'Clear All Objects');
  clearBtn.addEventListener('click', () => {
    accessories.clearAll();
    selectedItemId = null;
    refreshObjectsUi();
  });
  listGroup.append(objListEl, clearBtn);
  tabObjects.append(listGroup);

  tabObjects.append(el('h2', undefined, 'Selected Object Transform'));
  const transformGroup = el('div', 'group');
  tabObjects.append(transformGroup);

  const renderSelectedTransform = (item: AttachedObjItem | undefined) => {
    transformGroup.replaceChildren();
    if (!item) {
      transformGroup.append(el('div', 'name', 'Pilih atau tambahkan objek .obj untuk diatur.'));
      return;
    }

    const axes: [string, 0 | 1 | 2][] = [
      ['Pos X (m)', 0],
      ['Pos Y (m)', 1],
      ['Pos Z (m)', 2],
    ];
    for (const [label, idx] of axes) {
      addSlider(transformGroup, label, -0.25, 0.25, 0.002, item.transform.position[idx], 3, (v) => {
        item.transform.position[idx] = v;
        accessories.applyTransform(item);
      });
    }

    const rotAxes: [string, 0 | 1 | 2][] = [
      ['Rot X (deg)', 0],
      ['Rot Y (deg)', 1],
      ['Rot Z (deg)', 2],
    ];
    for (const [label, idx] of rotAxes) {
      addSlider(transformGroup, label, -180, 180, 1, item.transform.rotationDeg[idx], 0, (v) => {
        item.transform.rotationDeg[idx] = v;
        accessories.applyTransform(item);
      });
    }

    addSlider(transformGroup, 'Scale', 0.15, 2.8, 0.01, item.transform.scale, 2, (v) => {
      item.transform.scale = v;
      accessories.applyTransform(item);
    });

    const tintRow = el('div', 'color-item');
    const tintLabel = el('label', 'check-row');
    const tintCheck = el('input');
    tintCheck.type = 'checkbox';
    tintCheck.checked = item.useTint;
    tintCheck.addEventListener('change', () => {
      item.useTint = tintCheck.checked;
      accessories.applyTint(item);
    });
    tintLabel.append(tintCheck, el('span', undefined, 'Override Color Tint'));

    const tintColor = el('input');
    tintColor.type = 'color';
    tintColor.value = item.tintColor;
    tintColor.addEventListener('input', () => {
      item.tintColor = tintColor.value;
      item.useTint = true;
      tintCheck.checked = true;
      accessories.applyTint(item);
    });
    tintRow.append(tintLabel, tintColor);
    transformGroup.append(tintRow);

    const resetObjBtn = el('button', 'btn', 'Reset Transform');
    resetObjBtn.addEventListener('click', () => {
      accessories.resetTransform(item);
      renderSelectedTransform(item);
    });
    transformGroup.append(resetObjBtn);
  };

  refreshObjectsUi = () => {
    objListEl.replaceChildren();
    if (accessories.items.length === 0) {
      objListEl.append(el('div', 'name', 'Belum ada objek .obj terpasang.'));
      renderSelectedTransform(undefined);
      return;
    }
    if (!selectedItemId || !accessories.items.some((x) => x.id === selectedItemId)) {
      selectedItemId = accessories.items[accessories.items.length - 1].id;
    }

    for (const item of accessories.items) {
      const row = el('div', `obj-item${item.id === selectedItemId ? ' selected' : ''}`);
      row.addEventListener('click', () => {
        selectedItemId = item.id;
        refreshObjectsUi();
      });

      const title = el('span', 'obj-item-name', item.name);
      const visBtn = el('button', 'icon-btn', item.visible ? 'Hide' : 'Show');
      visBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        item.visible = !item.visible;
        accessories.applyTransform(item);
        refreshObjectsUi();
      });

      const delBtn = el('button', 'icon-btn danger', 'Del');
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        accessories.removeItem(item.id);
        refreshObjectsUi();
      });

      row.append(title, visBtn, delBtn);
      objListEl.append(row);
    }

    renderSelectedTransform(accessories.items.find((x) => x.id === selectedItemId));
  };

  tabObjects.append(el('h2', undefined, 'AR Occlusion & Alignment'));
  const arTuneGroup = el('div', 'group');
  const debugLabel = el('label', 'check-row');
  const debugCheck = el('input');
  debugCheck.type = 'checkbox';
  debugCheck.checked = viewer.showOccluderDebug;
  debugCheck.addEventListener('change', () => {
    viewer.showOccluderDebug = debugCheck.checked;
  });
  debugLabel.append(debugCheck, el('span', undefined, 'Show GNM Head Occluder Wireframe (AR)'));
  arTuneGroup.append(debugLabel);
  addSlider(arTuneGroup, 'AR Scale', 0.75, 1.35, 0.01, viewer.arHeadScale, 2, (v) => {
    viewer.arHeadScale = v;
  });
  addSlider(arTuneGroup, 'AR Depth Z', -0.05, 0.05, 0.002, viewer.arHeadOffset.z, 3, (v) => {
    viewer.arHeadOffset.z = v;
  });
  tabObjects.append(arTuneGroup);

  // =========================================================================
  // TAB 2: CHARACTER CUSTOMIZER (OPTION A)
  // =========================================================================
  const tabChar = document.querySelector<HTMLElement>('#tab-character')!;

  tabChar.append(el('h2', undefined, 'Character Identity Preset'));
  const presetCharGroup = el('div', 'group');
  const charSelect = el('select');
  for (const preset of identityPresets) {
    const opt = el('option');
    opt.value = preset.id;
    opt.textContent = preset.name;
    charSelect.append(opt);
  }
  charSelect.addEventListener('change', () => {
    const found = identityPresets.find((p) => p.id === charSelect.value);
    if (found) applyIdentityVector(found.identity);
  });

  const charBtns = el('div', 'buttons');
  const randomCharBtn = el('button', 'btn', 'Random Character');
  randomCharBtn.addEventListener('click', () => {
    const vec = new Float32Array(model.identityDim);
    for (let i = 0; i < model.identityDim; i++) {
      vec[i] = (Math.random() * 2 - 1) * (i < 16 ? 1.4 : 0.45);
    }
    applyIdentityVector(vec);
  });
  const resetCharBtn = el('button', 'btn', 'Neutral Face');
  resetCharBtn.addEventListener('click', () => {
    charSelect.value = 'neutral';
    applyIdentityVector(new Float32Array(model.identityDim));
  });
  charBtns.append(randomCharBtn, resetCharBtn);
  presetCharGroup.append(charSelect, charBtns);
  tabChar.append(presetCharGroup);

  tabChar.append(el('h2', undefined, 'Identity Proportions'));
  const idSlidersGroup = el('div', 'group');
  for (let i = 0; i < Math.min(IDENTITY_SLIDER_COUNT, model.identityDim); i++) {
    const label = model.manifest.identity.names[i];
    const slider = addSlider(idSlidersGroup, label, -3, 3, 0.05, 0, 2, (v) => {
      state.identity[i] = v;
      retargeter.setIdentity(state.identity);
      invalidate();
    });
    identitySliderUpdaters.push(slider.setDisplay);
  }
  tabChar.append(idSlidersGroup);

  tabChar.append(el('h2', undefined, 'Shader & Material Style'));
  const styleGroup = el('div', 'group');
  const styleSelect = el('select');
  for (const [id, entry] of Object.entries(STYLE_PRESETS)) {
    const opt = el('option');
    opt.value = id;
    opt.textContent = entry.label;
    styleSelect.append(opt);
  }

  const colorPickers: { key: keyof CharacterStyle; label: string; input: HTMLInputElement }[] = [];
  const colorGrid = el('div', 'color-grid');
  const colorFields: [keyof CharacterStyle, string][] = [
    ['skinLit', 'Skin Lit'],
    ['skinDeep', 'Shadow'],
    ['skinFill', 'Fill Light'],
    ['skinRim', 'Rim Glow'],
    ['eyeGlow', 'Eye Glow'],
    ['wireColor', 'Wireframe'],
    ['background', 'Backdrop'],
  ];
  for (const [key, label] of colorFields) {
    const cell = el('div', 'color-item');
    const picker = el('input');
    picker.type = 'color';
    picker.value = String(state.style[key]);
    picker.addEventListener('input', () => {
      (state.style[key] as string) = picker.value;
      state.style.presetId = 'custom';
      viewer.applyStyle(state.style);
    });
    colorPickers.push({ key, label, input: picker });
    cell.append(el('span', undefined, label), picker);
    colorGrid.append(cell);
  }

  const wireCheckLabel = el('label', 'check-row');
  const wireCheck = el('input');
  wireCheck.type = 'checkbox';
  wireCheck.checked = state.style.wireVisible;
  wireCheck.addEventListener('change', () => {
    state.style.wireVisible = wireCheck.checked;
    viewer.applyStyle(state.style);
  });
  wireCheckLabel.append(wireCheck, el('span', undefined, 'Show Quad Wireframe'));

  const wireOpSlider = addSlider(styleGroup, 'Wire Opacity', 0, 1, 0.02, state.style.wireOpacity, 2, (v) => {
    state.style.wireOpacity = v;
    viewer.applyStyle(state.style);
  });
  const bloomSlider = addSlider(styleGroup, 'Bloom', 0, 2, 0.02, state.style.bloomStrength, 2, (v) => {
    state.style.bloomStrength = v;
    viewer.applyStyle(state.style);
  });
  const rimSlider = addSlider(styleGroup, 'Rim Power', 0, 2.5, 0.05, state.style.rimStrength, 2, (v) => {
    state.style.rimStrength = v;
    viewer.applyStyle(state.style);
  });
  const specSlider = addSlider(styleGroup, 'Specular', 0, 2.5, 0.05, state.style.specular, 2, (v) => {
    state.style.specular = v;
    viewer.applyStyle(state.style);
  });

  styleSelect.addEventListener('change', () => {
    const next = STYLE_PRESETS[styleSelect.value];
    if (!next) return;
    state.style = { ...next.style };
    for (const c of colorPickers) c.input.value = String(state.style[c.key]);
    wireCheck.checked = state.style.wireVisible;
    wireOpSlider.setDisplay(state.style.wireOpacity);
    bloomSlider.setDisplay(state.style.bloomStrength);
    rimSlider.setDisplay(state.style.rimStrength);
    specSlider.setDisplay(state.style.specular);
    viewer.applyStyle(state.style);
  });

  styleGroup.prepend(wireCheckLabel);
  styleGroup.prepend(colorGrid);
  styleGroup.prepend(styleSelect);
  tabChar.append(styleGroup);

  // =========================================================================
  // TAB 3: FACE & POSE
  // =========================================================================
  const tabPose = document.querySelector<HTMLElement>('#tab-pose')!;

  tabPose.append(el('h2', undefined, 'Head Pose'));
  const poseGroup = el('div', 'group');
  (['pitch', 'yaw', 'roll'] as const).forEach((axis, idx) => {
    const s = addSlider(poseGroup, axis, -0.7, 0.7, 0.02, 0, 2, (v) => {
      if (headJoint >= 0) state.rotations[headJoint * 3 + idx] = v;
      invalidate();
    });
    poseResetters.push(() => s.setDisplay(0));
  });
  tabPose.append(poseGroup);

  if (model.correctiveDim > 0) {
    tabPose.append(el('h2', undefined, 'Correctives'));
    const corrGroup = el('div', 'group');
    for (let i = 0; i < model.correctiveDim; i++) {
      const name = model.manifest.correctives?.names[i] ?? `corrective_${i}`;
      const s = addSlider(corrGroup, name, 0, 1, 0.02, 0, 2, (v) => {
        state.correctives[i] = v;
        invalidate();
      });
      poseResetters.push(() => s.setDisplay(0));
    }
    tabPose.append(corrGroup);
  }

  tabPose.append(el('h2', undefined, 'Expressions'));
  const exprGroup = el('div', 'group');
  for (const [region, budget] of Object.entries(SLIDER_BUDGET)) {
    const span = model.manifest.expression.regions[region];
    if (!span) continue;
    for (let i = 0; i < Math.min(budget, span.count); i++) {
      const index = span.start + i;
      const label = model.manifest.expression.names[index];
      const s = addSlider(exprGroup, label, -3, 3, 0.05, 0, 2, (v) => {
        state.expression[index] = v;
        invalidate();
      });
      poseResetters.push(() => s.setDisplay(0));
    }
  }
  tabPose.append(exprGroup);

  const poseActions = el('div', 'buttons');
  poseActions.style.marginTop = '14px';
  const resetPoseBtn = el('button', 'btn', 'Reset Pose & Expression');
  resetPoseBtn.addEventListener('click', () => {
    state.expression.fill(0);
    state.rotations.fill(0);
    state.correctives.fill(0);
    poseResetters.forEach((r) => r());
    invalidate();
  });
  poseActions.append(resetPoseBtn);

  if (model.correspondence && viewer.hasCorrespondence) {
    const mapping = model.correspondence;
    let showingCorr = false;
    const corrBtn = el('button', 'btn', `Correspondence (${mapping.landmarks.length})`);
    corrBtn.addEventListener('click', () => {
      showingCorr = !showingCorr;
      viewer.setCorrespondenceVisible(showingCorr);
      camera.correspondence = showingCorr
        ? { landmarks: mapping.landmarks, colors: mapping.colors }
        : null;
      corrBtn.textContent = showingCorr
        ? 'Hide Correspondence'
        : `Correspondence (${mapping.landmarks.length})`;
    });
    poseActions.append(corrBtn);
  }
  tabPose.append(poseActions);

  // =========================================================================
  // AUTO-LOAD DEFAULT .OBJ & MAIN LOOP
  // =========================================================================
  try {
    const defaultGlasses = await accessories.addPreset(OBJ_PRESETS[0]);
    selectedItemId = defaultGlasses.id;
  } catch (err) {
    verifyEl.textContent = `Default .obj warning: ${String(err)}`;
  }
  refreshObjectsUi();

  const resize = () => {
    const rect = canvas.parentElement!.getBoundingClientRect();
    viewer.setSize(rect.width, rect.height);
  };
  window.addEventListener('resize', resize);
  resize();

  statusEl.textContent =
    `${model.vertexCount.toLocaleString()} verts · ${model.identityDim} id · ${model.expressionDim} expr`;

  // Per-frame retargeting loop (~30 Hz)
  setInterval(() => {
    if (!driving || !camera.isRunning) return;
    const frame = camera.latest;
    if (!frame) {
      if (viewer.mode === 'ar') viewer.updateArPose(null);
      return;
    }

    const solution = retargeter.solve(frame.landmarks, camera.aspect);
    state.expression.set(solution.expression);

    if (viewer.mode === 'studio') {
      if (headJoint >= 0) state.rotations.set(solution.headRotation, headJoint * 3);
    } else {
      // In AR mode, arHeadPivot handles full 6-DoF pose via MediaPipe facialTransformationMatrix
      state.rotations.fill(0);
      viewer.updateArPose(frame.transform);
    }

    const driven = model.correctiveBlendshapes;
    for (let i = 0; i < driven.length; i++) {
      const score = frame.blendshapes.get(driven[i]) ?? 0;
      state.correctives[i] += (score - state.correctives[i]) * CORRECTIVE_ALPHA;
    }

    invalidate();
    verifyEl.textContent =
      `${viewer.mode.toUpperCase()} · rms ${solution.rms.toFixed(1)} mm · ${camera.fps.toFixed(0)} fps`;
  }, 33);

  setInterval(() => {
    if (!camera.isRunning) return;
    const frame = camera.latest;
    if (!frame) {
      pipStatus.textContent = `no face · ${camera.fps.toFixed(0)} fps`;
      return;
    }
    const blink = Math.max(
      frame.blendshapes.get('eyeBlinkLeft') ?? 0,
      frame.blendshapes.get('eyeBlinkRight') ?? 0,
    );
    pipStatus.textContent = `${camera.fps.toFixed(0)} fps · blink ${blink.toFixed(2)}`;
  }, 150);

  const startTime = performance.now();
  const loop = () => {
    if (dirty) {
      dirty = false;
      const t0 = performance.now();
      viewer.updateGnm(state.identity, state.expression, state.rotations, state.translation, state.correctives);
      costEl.textContent = `gpu pass ${(performance.now() - t0).toFixed(2)} ms · mode: ${viewer.mode}`;
    }
    viewer.render((performance.now() - startTime) / 1000);
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch((error) => {
  const statusEl = document.querySelector<HTMLElement>('#status');
  if (statusEl) statusEl.textContent = String(error);
  throw error;
});
