import time
import cv2
import trimesh

import config
from core.tracker import FacePoseTracker
from core.gnm_head import GNMHeadModel
from viz.renderer import GlassesVTORenderer, create_procedural_glasses


def load_glasses_mesh() -> trimesh.Trimesh:
    if config.GLASSES_MODEL_PATH.exists():
        loaded = trimesh.load(str(config.GLASSES_MODEL_PATH), force="mesh")
        return loaded
    print("[INFO] glasses.glb not found, using procedural 3D glasses.")
    return create_procedural_glasses()


def main():
    tracker = FacePoseTracker(config.FACE_LANDMARKER_PATH)
    gnm_head = GNMHeadModel(scale=config.GNM_SCALE)
    glasses_mesh = load_glasses_mesh()

    renderer = GlassesVTORenderer(
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        fov_y=config.CAMERA_FOV_Y_RAD,
        head_trimesh=gnm_head.mesh,
        glasses_trimesh=glasses_mesh,
        glasses_offset=config.GLASSES_OFFSET,
        glasses_scale=config.GLASSES_SCALE,
        occlusion_eps=config.OCCLUSION_EPSILON,
    )

    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_ID}.")

    start_time = time.monotonic()
    prev_time = start_time

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror view for natural VTO feel
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            pose_matrix, _ = tracker.process(frame, timestamp_ms)
            if pose_matrix is not None:
                frame = renderer.render_overlay(frame, pose_matrix)

            # FPS display
            now = time.monotonic()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            cv2.imshow("GNM Head - 3D Glasses VTO", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        renderer.close()


if __name__ == "__main__":
    main()