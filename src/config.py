# EyeWave Configuration Constants

import os

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "calibration.json")
CLICK_SOUND = os.path.join(ASSETS_DIR, "click.wav")

# ── Camera ─────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS_TARGET = 30
TIMER_INTERVAL_MS = 1000 // FPS_TARGET  # ~33ms

# ── MediaPipe Face Mesh ────────────────────────────────────────────────
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Eye landmark indices (MediaPipe face mesh)
LEFT_EYE_CORNERS = [33, 133]       # inner, outer corner
LEFT_EYE_LIDS = [159, 145]         # top, bottom lid
RIGHT_EYE_CORNERS = [362, 263]
RIGHT_EYE_LIDS = [386, 374]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Head pose landmarks (nose tip, chin, left/right eye corners, mouth corners)
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# ── 3D Head Orientation (PCA from nose region) ────────────────────────
# Nose-region landmark indices for stable PCA-based head orientation.
# These landmarks are near the nose and are less affected by lateral
# head movement, providing a stable reference frame.
NOSE_INDICES = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ── Eye Sphere Tracking ───────────────────────────────────────────────
EYE_SPHERE_BASE_RADIUS = 20   # Radius at calibration distance (px-scale units)

# ── Gaze Direction ────────────────────────────────────────────────────
GAZE_SMOOTH_LENGTH = 10       # Number of frames to average for gaze smoothing

# Degree ranges for angular→screen mapping (from MonitorTracking.py)
# At ±GAZE_YAW_RANGE degrees, the gaze reaches the left/right screen edge.
# At ±GAZE_PITCH_RANGE degrees, the gaze reaches the top/bottom screen edge.
GAZE_YAW_RANGE = 12.0         # degrees left/right
GAZE_PITCH_RANGE = 3.0        # degrees up/down


# ── Blink Detection ───────────────────────────────────────────────────
BLINK_EAR_THRESHOLD = 0.18   # Eye Aspect Ratio below this = blink
BLINK_MIN_FRAMES = 3         # Minimum frames for a valid blink
BLINK_MAX_FRAMES = 12        # Maximum frames (longer = intentional close)

# ── Gaze Smoothing ────────────────────────────────────────────────────
SMOOTH_FACTOR = 0.45          # EMA alpha: higher = more responsive, lower = smoother

# ── Calibration ───────────────────────────────────────────────────────
# 16-point grid: more coverage, especially along edges and midpoints
CALIBRATION_POINTS = [
    # (norm_x, norm_y, label)
    # Center
    (0.5,  0.5,  "Center"),
    # Corners
    (0.08, 0.08, "Top-Left"),
    (0.92, 0.08, "Top-Right"),
    (0.08, 0.92, "Bottom-Left"),
    (0.92, 0.92, "Bottom-Right"),
    # Edges
    (0.5,  0.08, "Top-Center"),
    (0.5,  0.92, "Bottom-Center"),
    (0.08, 0.5,  "Left"),
    (0.92, 0.5,  "Right"),
    # Horizontal midpoints (helps with horizontal accuracy)
    (0.25, 0.3,  "Upper-Left Quarter"),
    (0.75, 0.3,  "Upper-Right Quarter"),
    (0.25, 0.7,  "Lower-Left Quarter"),
    (0.75, 0.7,  "Lower-Right Quarter"),
    # Extra horizontal points at center height
    (0.3,  0.5,  "Center-Left Mid"),
    (0.7,  0.5,  "Center-Right Mid"),
    # Center verification
    (0.5,  0.35, "Upper-Center"),
]
CALIBRATION_SAMPLES_PER_POINT = 35    # ~1.2 seconds at 30fps
CALIBRATION_POLY_DEGREE = 2           # Polynomial degree for regression
CALIBRATION_RIDGE_ALPHA = 0.5         # Lower alpha = less regularization = tighter fit

# ── Iris Range Amplification ─────────────────────────────────────────
# Iris ratios typically range 0.35-0.65 (very narrow). These params
# re-center and amplify them before they go into the regression.
IRIS_H_CENTER = 0.50   # Expected center of horizontal iris ratio
IRIS_H_GAIN = 3.0      # Amplify horizontal movement by this factor
IRIS_V_CENTER = 0.50
IRIS_V_GAIN = 2.0

# ── Dwell Selection ───────────────────────────────────────────────────
DWELL_THRESHOLD = 35   # Frames (~1.2s) of continuous gaze to select a key
KEY_HIT_MARGIN = 15    # Pixels to expand each key's gaze hit-area

# ── Virtual Keyboard ──────────────────────────────────────────────────
KEY_LABELS = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y',
    'Z', 'SPACE', 'BACK', 'SPEAK', 'CLR',
]
KEYBOARD_COLS = 5
KEY_SIZE = (120, 90)
SPECIAL_KEY_SIZE = (120, 90)
SPECIAL_KEYS = {'SPACE', 'BACK', 'SPEAK', 'CLR'}

# ── Word Prediction ───────────────────────────────────────────────────
NUM_PREDICTIONS = 3
COMMON_WORDS = [
    "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I",
    "IT", "FOR", "NOT", "ON", "WITH", "HE", "AS", "YOU", "DO", "AT",
    "THIS", "BUT", "HIS", "BY", "FROM", "THEY", "WE", "SAY", "HER", "SHE",
    "OR", "AN", "WILL", "MY", "ONE", "ALL", "WOULD", "THERE", "THEIR", "WHAT",
    "SO", "UP", "OUT", "IF", "ABOUT", "WHO", "GET", "WHICH", "GO", "ME",
    "WHEN", "MAKE", "CAN", "LIKE", "TIME", "NO", "JUST", "HIM", "KNOW", "TAKE",
    "PEOPLE", "INTO", "YEAR", "YOUR", "GOOD", "SOME", "COULD", "THEM", "SEE", "OTHER",
    "THAN", "THEN", "NOW", "LOOK", "ONLY", "COME", "ITS", "OVER", "THINK", "ALSO",
    "BACK", "AFTER", "USE", "TWO", "HOW", "OUR", "WORK", "FIRST", "WELL", "WAY",
    "EVEN", "NEW", "WANT", "BECAUSE", "ANY", "THESE", "GIVE", "DAY", "MOST", "US",
    "YES", "NO", "HELLO", "PLEASE", "THANK", "HELP", "SORRY", "NEED", "WATER", "FOOD",
]

# ── TTS ────────────────────────────────────────────────────────────────
TTS_RATE = 150
