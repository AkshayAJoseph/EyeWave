"""
config.py
=========
All constants, layouts, vocabulary, window geometry, and tuning parameters.
Edit this file to change timing, colours, layouts, or file paths.
"""

import os
import math

try:
    import pyautogui
    MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
except ImportError:
    MONITOR_WIDTH, MONITOR_HEIGHT = 1920, 1080

# ─────────────────────────────────────────────────────────────────────────────
#  FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")
CALIB_FILE      = os.path.join(BASE_DIR, "calibration.json")
GAZE_DATA_FILE  = os.path.join(BASE_DIR, "gaze_data.csv")
CLICK_SOUND     = os.path.join(ASSETS_DIR, "click.wav")
FACE_LANDMARKER = os.path.join(ASSETS_DIR, "face_landmarker.task")

# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD LAYOUTS
# ─────────────────────────────────────────────────────────────────────────────

# QWERTY  — 6 rows × 10 cols
QWERTY_GRID = [
    ['1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',  '0' ],
    ['Q',  'W',  'E',  'R',  'T',  'Y',  'U',  'I',  'O',  'P' ],
    ['A',  'S',  'D',  'F',  'G',  'H',  'J',  'K',  'L',  '?' ],
    ['Z',  'X',  'C',  'V',  'B',  'N',  'M',  '<',  '>',  'BP'],
    ['+',  '-',  ',',  '.',  '/',  '*',  '!',  ' ',  'DL', 'PL'],
    ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
]

# AAC  — 8 rows × 6 cols
# Rows 0-1: large phrase buttons
# Rows 2-6: frequency-ordered letters (E T A O I N … Z)
# Row 7:    special / control keys
AAC_GRID = [
    ['YES',   'NO',    'HELP',  'PAIN',  'WATER', 'TOILET'],
    ['MORE',  'STOP',  'HUNGRY','TIRED', 'OKAY',  'DOCTOR'],
    ['E',     'T',     'A',     'O',     'I',     'N'     ],
    ['S',     'H',     'R',     'D',     'L',     'C'     ],
    ['U',     'M',     'F',     'P',     'G',     'W'     ],
    ['Y',     'B',     'V',     'K',     'X',     'J'     ],
    ['Q',     'Z',     '?',     ',',     '.',     '!'     ],
    ['SPACE', 'NUM',   'BP',    'DL',    'PL',    'SWAP'  ],
]

AAC_PHRASE_ROWS = {0, 1}    # rendered at 2× row height
AAC_SPECIAL_ROW = 7         # rendered at 1.2× height

# ─────────────────────────────────────────────────────────────────────────────
#  PHRASES  (AAC quick-access + QWERTY P1-P10)
# ─────────────────────────────────────────────────────────────────────────────
PHRASES = {
    # AAC top-row phrases
    'YES':    "Yes",
    'NO':     "No",
    'HELP':   "I need help",
    'PAIN':   "I am in pain",
    'WATER':  "I want water",
    'TOILET': "I want to go to the washroom",
    'MORE':   "I want more",
    'STOP':   "Please stop",
    'HUNGRY': "I am hungry",
    'TIRED':  "I am tired",
    'OKAY':   "I am okay",
    'DOCTOR': "Please call a doctor",
    # QWERTY P-keys
    'P1':  "I'm hungry",       'P2':  "I want water",
    'P3':  "I'm satisfied",    'P4':  "I'm not satisfied",
    'P5':  "I want to go to the washroom",
    'P6':  "Can anyone come over here?",
    'P7':  "Could you read something for me?",
    'P8':  "Can we talk a little bit?",
    'P9':  "Can I get more",   'P10': "Thank you",
    # Control aliases
    'SPACE': ' ',
    'SWAP':  '__SWAP__',
    'NUM':   '__NUM__',
}

# ─────────────────────────────────────────────────────────────────────────────
#  AAC WORD PREDICTION VOCABULARY  (AAC-specific, frequency-ordered)
# ─────────────────────────────────────────────────────────────────────────────
AAC_VOCAB = sorted(set([
    "I","am","want","need","help","please","thank","you","yes","no","okay",
    "more","stop","hurt","pain","tired","hungry","thirsty","hot","cold",
    "comfortable","uncomfortable","ready","done","again","wait","hurry",
    "water","food","medicine","doctor","nurse","toilet","bathroom","bed",
    "pillow","blanket","phone","call","read","write","listen","sleep","wake",
    "happy","sad","scared","confused","bored","lonely","anxious","better",
    "worse","good","bad","fine","sick","dizzy","nauseous",
    "sit","stand","walk","move","turn","lift","hold","put","get","go",
    "come","look","see","hear","feel","talk","speak","sign",
    "the","a","in","on","at","to","of","and","but","or","not","can",
    "will","would","could","should","have","has","had","is","are","was",
    "my","your","his","her","their","our","this","that","here","there",
    "what","who","when","where","why","how","much","many","some","all",
]))

# ─────────────────────────────────────────────────────────────────────────────
#  WINDOW / GRID GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
KBD_WIN_W = MONITOR_WIDTH
KBD_WIN_H = MONITOR_HEIGHT
GRID_X    = 16
GRID_Y    = 38
GRID_W    = KBD_WIN_W - 32
GRID_H    = int(KBD_WIN_H * 0.60)
TEXT_Y    = GRID_Y + GRID_H + 18
TEXT_H    = 60
SUGG_Y    = TEXT_Y + TEXT_H + 8

# ─────────────────────────────────────────────────────────────────────────────
#  GAZE PIPELINE TUNING
# ─────────────────────────────────────────────────────────────────────────────
# Adaptive filter
FILTER_ALPHA_SACCADE   = 0.08
FILTER_ALPHA_FIXATION  = 0.55
FILTER_SACCADE_THRESH  = 0.10
FILTER_FIXATION_THRESH = 0.025

# Fixation detector (I-DT)
FIXATION_WINDOW        = 18
FIXATION_DISP_MAX      = 0.028
FIXATION_SPEED_MAX     = 0.045
FIXATION_MIN_SAMPLES   = 8

# Dwell controller
DWELL_TIME             = 1.3    # seconds of accumulated fixation to fire
DWELL_COOLDOWN         = 0.9    # min gap between same-key activations
DWELL_CONFIRM_FRAMES   = 5      # hysteresis: frames before hover switches

# Blink detector
BLINK_EAR_THRESH       = 0.21   # Eye Aspect Ratio closed threshold
BLINK_MIN_MS           = 150    # min closed duration (intentional)
BLINK_MAX_MS           = 500    # max closed duration (above = hold)
BLINK_LONG_MAX_MS      = 1200   # max duration for long-blink undo gesture
BLINK_DOUBLE_GAP_MS    = 700    # window for double-blink detection

# Scanner
SCAN_ROW_RATE          = 1.5    # seconds per row
SCAN_COL_RATE          = 1.2    # seconds per column
SCAN_COL_TIMEOUT       = 8.0    # seconds before col scan auto-returns to row scan

# Audio feedback
AUDIO_ENABLED          = True
AUDIO_ROW_TICK         = (400, 40)    # (freq_hz, duration_ms) — row advance
AUDIO_COL_TICK         = (600, 40)    # column advance
AUDIO_ROW_SELECT       = [(500, 60), (700, 60)]   # rising tone — row confirmed
AUDIO_CANCEL           = [(600, 60), (400, 60)]    # falling tone — cancel/timeout
AUDIO_UNDO             = [(800, 50), (500, 50)]    # undo beep
AUDIO_KEY_ACTIVATE     = (900, 50)    # key activated

# Profiles
PROFILES_DIR           = os.path.join(BASE_DIR, "profiles")

# Calibration (corner)
CALIB_STABLE_DISP_MAX  = 0.025
CALIB_STABLE_WINDOW    = 30
CALIB_MIN_STABLE       = 8

# ─────────────────────────────────────────────────────────────────────────────
#  3D TRACKER / ORBIT
# ─────────────────────────────────────────────────────────────────────────────
BASE_RADIUS  = 20               # eyeball sphere radius at calibration

# Orbit debug camera defaults
ORBIT_YAW    = math.radians(-151.0)
ORBIT_PITCH  = 0.0
ORBIT_RADIUS = 1500.0
ORBIT_FOV    = 50.0

# MediaPipe: nose landmark indices (PCA head pose)
NOSE_IDX = [4,45,275,220,440,1,5,51,281,44,274,241,
            461,125,354,218,438,195,167,393,165,391,3,248]

# MediaPipe: EAR landmark indices (blink detection)
# Order: [outer_corner, upper_lid_1, upper_lid_2, inner_corner, lower_lid_1, lower_lid_2]
LEFT_EYE_EAR  = [263, 385, 387, 362, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
