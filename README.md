# GNM Virtual Try-On & 3D Character Studio

Built on [GNM Head](https://github.com/google/GNM), Google's open-source parametric model of a human head/face.

Real-time 3D Virtual Try-On and parametric character puppetry built with **Google GNM Head** and **MediaPipe Face Landmarker**.

Includes two versions:
- **Web App (`web/`)** — Browser-native Three.js + WebGL + MediaPipe WASM with dual-mode rendering (**3D Puppet Studio** & **AR Webcam Try-On**), multi-`.obj` accessory try-on, and parametric/shader character customization.
- **Python Desktop Prototype (`main.py`)** — OpenCV + MediaPipe + Pyrender offscreen depth-occluded glasses try-on.

---

## Features (Web Version)

1. **Dual Render Modes**
   - **3D Puppet Studio**: Stylized GPU-evaluated GNM head with view-space lighting, Fresnel rim, quad wireframe, bloom, and OrbitControls.
   - **AR Webcam Try-On**: Live webcam background with the GNM head acting as a real-time **depth occluder** (`colorWrite: false, depthWrite: true`) so `.obj` temples/brims are naturally occluded behind your face and head.
2. **Multi-`.obj` Try-On**
   - Built-in `.obj` + `.mtl` presets (`Round Glasses`, `Cyber Visor`, `Royal Crown`, `Top Hat`, `Cat Ears`).
   - Drag-and-drop or upload custom `.obj` files (optionally paired with `.mtl` and texture images).
   - Auto-normalization to head scale, visibility toggles, color tint override, and per-object **Position / Rotation / Scale** controls.
3. **Custom Character (Parametric & Shader)**
   - Semantic character presets generated from GNM `IdentitySampler`, random identity generator, 12 identity proportion sliders, and **Fit Identity** directly from webcam landmarks.
   - Shader presets (`Cyber Teal`, `Studio Clay`, `Gilded Bronze`, `Neon Synth`) plus color pickers for skin lit/shadow, fill light, rim glow, eye glow, wireframe, and backdrop.
4. **Live Facial Tracking**
   - Real-time expression + head pose retargeting from MediaPipe 478 landmarks and 52 ARKit blendshapes (including `cheekPuff` corrective).

---

## Project Structure

```text
gnm-virtual-try-on/
├── main.py                  # Python desktop VTO entry point
├── config.py                # Camera, FOV, and alignment config
├── core/
│   ├── gnm_head.py          # GNM head mesh wrapper
│   └── tracker.py           # MediaPipe FaceLandmarker wrapper
├── viz/
│   └── renderer.py          # Pyrender 2-pass depth-occluded glasses renderer
├── assets/
│   ├── face_landmarker.task # MediaPipe FaceLandmarker model
│   └── correspondence.npz   # MediaPipe landmark <-> GNM vertex mapping
└── web/
    ├── index.html
    ├── package.json
    ├── tools/
    │   ├── export_assets.py     # Exports GNM bin/json, presets, and sample .obj
    │   └── sync_web_assets.mjs  # Copies MediaPipe WASM & face_landmarker.task
    ├── public/assets/           # Generated browser assets (gnm_head.bin/json, objects/)
    └── src/
        ├── main.ts              # App orchestration & studio UI
        ├── styles.css           # UI styling
        ├── gnm/                 # GNM model, WebGL evaluator, identity fit, retargeter
        ├── scene/               # DualModeViewer, ObjAccessoryManager, character styles
        └── tracking/            # CameraController, FaceTracker, 2D PiP overlay
```

---

## Getting Started

### 1. Prerequisites
- **Python 3.11+**
- **Node.js 20+** and `npm`
- **Google GNM**: Clone the repository this command: 
```bash 
git clone https://github.com/google/GNM.git
```

### 2. Run the Web App (`web/`)

First, export the compact browser assets (~18.5 MB) using your `gnm` Python environment:

```bash
conda activate gnm
python web/tools/export_assets.py
```

Then install JS dependencies, sync MediaPipe WASM assets, and start the Vite dev server:

```bash
cd web
npm install
npm run sync-assets
npm run dev
```

Open `http://localhost:5173` in your browser.

To build for production:

```bash
npm run build
```

### 3. Deploy to Vercel (Web App)

A root [`vercel.json`](vercel.json) is included so you can deploy directly from GitHub with zero extra configuration:

1. Push this repository to GitHub (includes `web/public/assets/gnm_head.bin` & `gnm_head.json`).
2. Import the repository in **Vercel** and leave **Root Directory** as `.` (repo root).
3. Vercel will automatically run `npm --prefix web install`, `npm --prefix web run build` (which syncs MediaPipe WASM and `face_landmarker.task`, typechecks, and bundles with Vite), and serve `web/dist`.

*(Optional)* If you prefer hosting `gnm_head.bin` and `gnm_head.json` on external object storage (e.g., Cloudflare R2 / Vercel Blob with CORS enabled), set the `VITE_GNM_ASSETS_URL` environment variable in Vercel to that base URL.

### 4. Run the Python Desktop Version

```bash
conda activate gnm
python main.py
```

Press `q` in the preview window to quit.

> **Note:** Only one process can open `/dev/video0` at a time. Make sure the desktop script is closed before starting the webcam in the browser (and vice versa).
