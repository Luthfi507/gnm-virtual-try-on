/**
 * Manages the webcam stream, MediaPipe detection loop, and optional circular PiP canvas.
 */

import { centreCrop, drawFace, type CorrespondenceOverlay } from './overlay.ts';
import { FaceTracker, type FaceFrame } from './tracker.ts';

export class CameraController {
  latest: FaceFrame | null = null;
  fps = 0;
  mirror = true;
  drawPip = true;
  correspondence: CorrespondenceOverlay | null = null;

  readonly video: HTMLVideoElement;
  private readonly pipCanvas: HTMLCanvasElement;
  private readonly pipContext: CanvasRenderingContext2D;
  private readonly pipSize: number;

  private tracker: FaceTracker | null = null;
  private stream: MediaStream | null = null;
  private running = false;
  private previous = 0;

  constructor(pipCanvas: HTMLCanvasElement, pipSize = 180) {
    this.pipCanvas = pipCanvas;
    this.pipSize = pipSize;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    pipCanvas.width = pipSize * ratio;
    pipCanvas.height = pipSize * ratio;
    pipCanvas.style.width = `${pipSize}px`;
    pipCanvas.style.height = `${pipSize}px`;

    this.pipContext = pipCanvas.getContext('2d')!;
    this.pipContext.scale(ratio, ratio);

    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
  }

  get isRunning(): boolean {
    return this.running;
  }

  get aspect(): number {
    const { videoWidth, videoHeight } = this.video;
    return videoWidth && videoHeight ? videoWidth / videoHeight : 0;
  }

  async start(): Promise<void> {
    if (this.running) return;
    if (!this.tracker) this.tracker = await FaceTracker.create();

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    });
    this.video.srcObject = this.stream;
    await this.video.play();

    this.running = true;
    this.previous = performance.now();
    this.loop();
  }

  stop(): void {
    this.running = false;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.video.srcObject = null;
    this.latest = null;
    this.pipContext.clearRect(0, 0, this.pipSize, this.pipSize);
  }

  private loop = (): void => {
    if (!this.running) return;
    const now = performance.now();
    const delta = now - this.previous;
    this.previous = now;
    if (delta > 0) this.fps = this.fps * 0.9 + (1000 / delta) * 0.1;

    const width = this.video.videoWidth;
    const height = this.video.videoHeight;
    if (width && height && this.tracker) {
      const frame = this.tracker.detect(this.video, now);
      if (frame) this.latest = frame;

      if (this.drawPip && this.pipCanvas.classList.contains('live')) {
        const crop = centreCrop(width, height, this.pipSize);
        const ctx = this.pipContext;
        ctx.save();
        ctx.clearRect(0, 0, this.pipSize, this.pipSize);
        ctx.beginPath();
        ctx.arc(this.pipSize / 2, this.pipSize / 2, this.pipSize / 2, 0, Math.PI * 2);
        ctx.clip();

        if (this.mirror) {
          ctx.translate(this.pipSize, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(this.video, crop.sourceX, crop.sourceY, crop.side, crop.side, 0, 0, this.pipSize, this.pipSize);
        if (this.latest) {
          drawFace(ctx, this.latest, crop, {
            showPoints: true,
            correspondence: this.correspondence,
            gazeScale: this.pipSize * 0.28,
            lineScale: this.pipSize / 320,
          });
        }
        ctx.restore();
      }
    }

    requestAnimationFrame(this.loop);
  };
}
