from pathlib import Path
from urllib.request import urlretrieve

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

FACE_LANDMARKER_PATH = str(ASSETS_DIR / "face_landmarker.task")
GLASSES_MODEL_PATH = ASSETS_DIR / "glasses.glb"

if not Path(FACE_LANDMARKER_PATH).exists():
	FACE_LANDMARKER_URL = (
		"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
		"face_landmarker/float16/1/face_landmarker.task"
	)
	urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)

# Camera settings
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FOV_Y_RAD = 0.85  # ~48.7 degrees vertical FOV (typical laptop webcam)

# Alignment & occlusion tuning (in meters)
GNM_SCALE = 1.0
OCCLUSION_EPSILON = 0.004  # Depth tolerance to prevent z-fighting on nose bridge

# Glasses offset relative to head center [x, y, z] in meters
GLASSES_OFFSET = [0.0, 0.025, 0.085]
GLASSES_SCALE = 1.0