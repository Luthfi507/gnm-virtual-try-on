/**
 * 2D landmark & correspondence overlay renderer for the circular webcam PiP.
 */

import { CONTOURS, type FaceFrame } from './tracker.ts';

export const OVERLAY_COLORS = {
  points: 'rgba(124, 222, 197, 0.55)',
  contour: 'rgba(84, 196, 167, 0.35)',
  eye: '#54c4a7',
  iris: '#7cdec5',
  gaze: '#9b3ca1',
} as const;

export interface Projection {
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}

export interface CorrespondenceOverlay {
  landmarks: Uint16Array;
  colors: Uint8Array;
}

export interface OverlayOptions {
  showPoints?: boolean;
  gazeScale?: number;
  lineScale?: number;
  correspondence?: CorrespondenceOverlay | null;
}

export function centreCrop(
  videoWidth: number,
  videoHeight: number,
  size: number,
): Projection & { sourceX: number; sourceY: number; side: number } {
  const side = Math.min(videoWidth, videoHeight);
  const sourceX = (videoWidth - side) / 2;
  const sourceY = (videoHeight - side) / 2;
  const scale = size / side;
  return {
    sourceX,
    sourceY,
    side,
    scaleX: videoWidth * scale,
    scaleY: videoHeight * scale,
    offsetX: -sourceX * scale,
    offsetY: -sourceY * scale,
  };
}

function strokeContour(
  context: CanvasRenderingContext2D,
  frame: FaceFrame,
  connectors: readonly { start: number; end: number }[],
  projection: Projection,
): void {
  context.beginPath();
  for (const { start, end } of connectors) {
    const from = frame.landmarks[start];
    const to = frame.landmarks[end];
    if (!from || !to) continue;
    context.moveTo(from.x * projection.scaleX + projection.offsetX, from.y * projection.scaleY + projection.offsetY);
    context.lineTo(to.x * projection.scaleX + projection.offsetX, to.y * projection.scaleY + projection.offsetY);
  }
  context.stroke();
}

export function drawFace(
  context: CanvasRenderingContext2D,
  frame: FaceFrame,
  projection: Projection,
  options: OverlayOptions = {},
): void {
  const { showPoints = true, gazeScale = 70, lineScale = 1, correspondence = null } = options;

  if (correspondence) {
    const radius = Math.max(1.2, 1.9 * lineScale);
    for (let i = 0; i < correspondence.landmarks.length; i++) {
      const point = frame.landmarks[correspondence.landmarks[i]];
      if (!point) continue;
      const r = correspondence.colors[i * 3];
      const g = correspondence.colors[i * 3 + 1];
      const b = correspondence.colors[i * 3 + 2];
      context.fillStyle = `rgb(${r},${g},${b})`;
      context.beginPath();
      context.arc(point.x * projection.scaleX + projection.offsetX, point.y * projection.scaleY + projection.offsetY, radius, 0, Math.PI * 2);
      context.fill();
    }
  } else if (showPoints) {
    const dot = Math.max(0.8, 1.4 * lineScale);
    context.fillStyle = OVERLAY_COLORS.points;
    for (const point of frame.landmarks) {
      context.fillRect(
        point.x * projection.scaleX + projection.offsetX - dot / 2,
        point.y * projection.scaleY + projection.offsetY - dot / 2,
        dot,
        dot,
      );
    }
  }

  context.lineWidth = lineScale;
  context.strokeStyle = OVERLAY_COLORS.contour;
  for (const contour of [CONTOURS.faceOval, CONTOURS.lips, CONTOURS.leftBrow, CONTOURS.rightBrow]) {
    strokeContour(context, frame, contour, projection);
  }

  context.lineWidth = 1.5 * lineScale;
  context.strokeStyle = OVERLAY_COLORS.eye;
  for (const contour of [CONTOURS.leftEye, CONTOURS.rightEye]) strokeContour(context, frame, contour, projection);

  context.strokeStyle = OVERLAY_COLORS.iris;
  for (const contour of [CONTOURS.leftIris, CONTOURS.rightIris]) strokeContour(context, frame, contour, projection);

  if (!frame.gaze) return;
  for (const eye of [frame.gaze.left, frame.gaze.right]) {
    const x = eye.centre.x * projection.scaleX + projection.offsetX;
    const y = eye.centre.y * projection.scaleY + projection.offsetY;
    context.fillStyle = OVERLAY_COLORS.gaze;
    context.beginPath();
    context.arc(x, y, 2.5 * lineScale, 0, Math.PI * 2);
    context.fill();

    context.strokeStyle = OVERLAY_COLORS.gaze;
    context.lineWidth = 2 * lineScale;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + eye.direction.x * gazeScale, y + eye.direction.y * gazeScale);
    context.stroke();
  }
}
