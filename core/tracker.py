import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FacePoseTracker:
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Returns 4x4 head pose matrix (in meters) and raw landmarks, or (None, None)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.facial_transformation_matrixes:
            return None, None

        # MediaPipe transformation matrix is right-handed (OpenGL convention) in cm
        pose_matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
        pose_matrix[:3, 3] /= 100.0  # Convert translation from cm to meters

        landmarks = result.face_landmarks[0]
        return pose_matrix, landmarks

    def close(self):
        self.landmarker.close()