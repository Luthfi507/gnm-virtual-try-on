/**
 * MediaPipe FaceLandmarker wrapper producing 478 landmarks, 52 blendshapes,
 * 4x4 facial transformation matrix (in cm, column-major), and iris gaze.
 */

import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

export const EYE_LANDMARKS = {
  left: { irisCentre: 468, outerCorner: 33, innerCorner: 133 },
  right: { irisCentre: 473, outerCorner: 263, innerCorner: 362 },
} as const;

export interface Gaze {
  centre: NormalizedLandmark;
  direction: { x: number; y: number };
}

export interface FaceFrame {
  landmarks: NormalizedLandmark[];
  blendshapes: Map<string, number>;
  transform: Float32Array | null;
  gaze: { left: Gaze; right: Gaze } | null;
}

function eyeGaze(
  landmarks: NormalizedLandmark[],
  indices: { irisCentre: number; outerCorner: number; innerCorner: number },
): Gaze {
  const iris = landmarks[indices.irisCentre];
  const outer = landmarks[indices.outerCorner];
  const inner = landmarks[indices.innerCorner];
  const midpointX = (outer.x + inner.x) / 2;
  const midpointY = (outer.y + inner.y) / 2;
  const halfWidth = Math.hypot(outer.x - inner.x, outer.y - inner.y) / 2;
  const scale = halfWidth > 1e-6 ? halfWidth : 1;

  return {
    centre: iris,
    direction: {
      x: (iris.x - midpointX) / scale,
      y: (iris.y - midpointY) / (scale * 0.5),
    },
  };
}

export class FaceTracker {
  private readonly landmarker: FaceLandmarker;
  private lastTimestamp = -1;

  private constructor(landmarker: FaceLandmarker) {
    this.landmarker = landmarker;
  }

  static async create(
    wasmPath = 'mediapipe/wasm',
    modelPath = 'assets/face_landmarker.task',
  ): Promise<FaceTracker> {
    const fileset = await FilesetResolver.forVisionTasks(wasmPath);
    const landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    return new FaceTracker(landmarker);
  }

  detect(video: HTMLVideoElement, timestampMs: number): FaceFrame | null {
    if (timestampMs <= this.lastTimestamp) return null;
    this.lastTimestamp = timestampMs;

    const result: FaceLandmarkerResult = this.landmarker.detectForVideo(video, timestampMs);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks || landmarks.length === 0) return null;

    const blendshapes = new Map<string, number>();
    for (const category of result.faceBlendshapes?.[0]?.categories ?? []) {
      blendshapes.set(category.categoryName, category.score);
    }
    const matrix = result.facialTransformationMatrixes?.[0];

    return {
      landmarks,
      blendshapes,
      transform: matrix ? Float32Array.from(matrix.data) : null,
      gaze: {
        left: eyeGaze(landmarks, EYE_LANDMARKS.left),
        right: eyeGaze(landmarks, EYE_LANDMARKS.right),
      },
    };
  }

  close(): void {
    this.landmarker.close();
  }
}

export const CONTOURS = {
  faceOval: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
  leftEye: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
  rightEye: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
  leftBrow: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
  rightBrow: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
  leftIris: FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
  rightIris: FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
  lips: FaceLandmarker.FACE_LANDMARKS_LIPS,
} as const;
