#!/usr/bin/env python3
"""
Eye-Tracking Controlled Keyboard — 3D Gaze Edition
====================================================
Integrates a full 3D eyeball-sphere gaze tracker (MediaPipe FaceMesh + PCA
head pose) with a dwell-time virtual keyboard.

Gaze accuracy:
  • The virtual "monitor plane" is calibrated once (press C).
  • A 4-point polynomial warp then refines per-corner accuracy (keys 1-4).
  • The keyboard window IS the monitor plane — no angle→pixel conversion needed.

Keyboard window controls:
  C           — Calibrate eye spheres + place monitor plane
  1,2,3,4     — Multi-point corner calibration (TL, TR, BL, BR)
  Space       — Confirm a calibration point (look at corner, then press key)
  S           — Screen-center yaw/pitch trim (legacy fallback)
  X           — Drop a gaze marker on the monitor plane (debug)
  F7          — Toggle OS mouse control
  J/L/I/K     — Orbit debug camera (yaw/pitch)
  [ / ]       — Orbit zoom out / in
  R           — Reset orbit view
  Q           — Quit

Requirements:
  pip install opencv-python mediapipe numpy scipy pyautogui keyboard pyttsx3
"""

import cv2
import numpy as np
import time
import math
import threading
from collections import deque
from scipy.spatial.transform import Rotation as Rscipy

try:
    import pyautogui
    PYAUTOGUI_OK = True
    MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
except ImportError:
    PYAUTOGUI_OK = False
    MONITOR_WIDTH, MONITOR_HEIGHT = 1920, 1080

try:
    import keyboard as kb
    KB_OK = True
except ImportError:
    KB_OK = False

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False

import mediapipe as mp

# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
KEYBOARD = [
    ['1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',  '0' ],
    ['Q',  'W',  'E',  'R',  'T',  'Y',  'U',  'I',  'O',  'P' ],
    ['A',  'S',  'D',  'F',  'G',  'H',  'J',  'K',  'L',  '?' ],
    ['Z',  'X',  'C',  'V',  'B',  'N',  'M',  '<',  '>',  'BP'],
    ['+',  '-',  ',',  '.',  '/',  '*',  '!',  ' ',  'DL', 'PL'],
    ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
]
ROWS = len(KEYBOARD)
COLS = len(KEYBOARD[0])

PHRASES = {
    'P1':  "I'm hungry",
    'P2':  "I want water",
    'P3':  "I'm satisfied",
    'P4':  "I'm not satisfied",
    'P5':  "I want to go to the washroom",
    'P6':  "Can anyone come over here?",
    'P7':  "Could you read something for me?",
    'P8':  "Can we talk a little bit?",
    'P9':  "Can I get more",
    'P10': "Thank you",
}

WORD_DICT = sorted([
    "the","be","to","of","and","a","in","that","have","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her",
    "she","or","an","will","my","one","all","would","there","their","what","so","up",
    "out","if","about","who","get","which","go","me","when","make","can","like","time",
    "no","just","him","know","take","people","into","year","your","good","some","could",
    "them","see","other","than","then","now","look","only","come","its","over","think",
    "also","back","after","use","two","how","our","work","first","well","way","even",
    "new","want","because","any","these","give","day","most","us","water","hungry",
    "help","please","thank","yes","need","feel","pain","tired","okay","bathroom","drink",
    "food","medicine","doctor","nurse","call","phone","read","write","talk","listen",
    "sleep","wake","sit","stand","walk","hot","cold","comfortable","uncomfortable","more",
    "less","stop","start","again","done","ready","wait","hurry","slowly","carefully",
])

# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD WINDOW DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────
KBD_WIN_W  = MONITOR_WIDTH
KBD_WIN_H  = MONITOR_HEIGHT
GRID_X     = 18
GRID_Y     = 32
GRID_W     = KBD_WIN_W - 36
GRID_H     = int(KBD_WIN_H * 0.60)
TEXT_Y     = GRID_Y + GRID_H + 18
TEXT_H     = 62
SUGG_Y     = TEXT_Y + TEXT_H + 10
STATUS_Y   = KBD_WIN_H - 28

KEY_W = GRID_W // COLS
KEY_H = GRID_H // ROWS

DWELL_TIME  = 1.4   # seconds held before activation
COOLDOWN    = 0.7   # min time between same-key activations

# ─────────────────────────────────────────────────────────────────────────────
#  GAZE CONSTANTS (3D tracker)
# ─────────────────────────────────────────────────────────────────────────────
FILTER_LEN  = 12
GAZE_LEN_3D = 350

# ─────────────────────────────────────────────────────────────────────────────
#  ORBIT DEBUG VIEW
# ─────────────────────────────────────────────────────────────────────────────
orbit_yaw    = math.radians(-151.0)
orbit_pitch  = 0.0
orbit_radius = 1500.0
orbit_fov    = 50.0
debug_world_frozen  = False
orbit_pivot_frozen  = None

# Nose landmark indices for PCA head pose
NOSE_INDICES = [4,45,275,220,440,1,5,51,281,44,274,241,
                461,125,354,218,438,195,167,393,165,391,3,248]


# ═════════════════════════════════════════════════════════════════════════════
#  MULTI-POINT CALIBRATION
#  Collect raw (a,b) at 4 known screen corners, then build a bilinear warp.
# ═════════════════════════════════════════════════════════════════════════════
class MultiPointCalib:
    """
    Collects raw (a,b) samples at 4 known positions (TL, TR, BL, BR) and
    computes a bilinear correction that maps raw → corrected (a,b) spanning
    exactly [0,1]×[0,1] across the keyboard grid.
    """
    TARGETS = [          # screen corners (norm): (a_target, b_target)
        (0.0, 0.0),      # 0: Top-Left
        (1.0, 0.0),      # 1: Top-Right
        (0.0, 1.0),      # 2: Bottom-Left
        (1.0, 1.0),      # 3: Bottom-Right
    ]
    LABELS = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_pts   = [None] * 4   # raw (a,b) at each corner
        self.active    = False        # currently collecting?
        self.step      = 0            # which corner (0-3)
        self._samples  = []           # accumulating samples for current step
        self._done     = False
        self._H        = None         # 3x3 homography raw→corrected

    def start(self):
        self.reset()
        self.active = True
        self.step   = 0
        print(f"\n[Calibration] Look at the {self.LABELS[0]} corner of the keyboard.")
        print("  Press SPACE to record, then move to next corner.")

    def record(self, a: float, b: float):
        """Call this while calibration is active to accumulate a sample."""
        if not self.active:
            return
        self._samples.append((a, b))

    def confirm_point(self):
        """User pressed SPACE — commit current averaged sample."""
        if not self.active or not self._samples:
            return
        avg_a = float(np.mean([s[0] for s in self._samples]))
        avg_b = float(np.mean([s[1] for s in self._samples]))
        self.raw_pts[self.step] = (avg_a, avg_b)
        print(f"[Calibration] {self.LABELS[self.step]} recorded: raw=({avg_a:.3f},{avg_b:.3f})")
        self._samples = []
        self.step += 1
        if self.step >= 4:
            self.active = False
            self._done  = True
            self._build_homography()
        else:
            print(f"[Calibration] Now look at {self.LABELS[self.step]}. Press SPACE.")

    def _build_homography(self):
        src = np.array(self.raw_pts,     dtype=np.float32)
        dst = np.array(self.TARGETS,     dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst)
        print("[Calibration] Homography built. Gaze correction active.")

    def correct(self, a: float, b: float):
        """Apply correction warp. Returns (a,b) unchanged if not ready."""
        if self._H is None:
            return a, b
        pt = np.array([[[a, b]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        ca = float(np.clip(out[0, 0, 0], 0.0, 1.0))
        cb = float(np.clip(out[0, 0, 1], 0.0, 1.0))
        return ca, cb

    @property
    def ready(self):
        return self._done and self._H is not None

    @property
    def current_label(self):
        if self.active and self.step < 4:
            return self.LABELS[self.step]
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  KEYBOARD GUI
# ═════════════════════════════════════════════════════════════════════════════
class EyeKeyboard:

    def __init__(self):
        self.typed_text   = ""
        self.suggestions  : list[str] = []

        # Dwell state
        self.hovered : tuple[int,int] | None = None
        self.dwell_t0: float | None          = None
        self.last_key: tuple[int,int] | None = None
        self.last_key_time: float            = 0.0

        # Flash feedback
        self.flash_key: tuple[int,int] | None = None
        self.flash_end: float                  = 0.0

        # Gaze cursor position (a,b) normalised on keyboard
        self.gaze_a = 0.5
        self.gaze_b = 0.5

        # Status line (top of keyboard window)
        self.status = "Press C to calibrate eye spheres, then 1-2-3-4 for accuracy."

    # ── Gaze update (called every frame from main loop) ──────────────────────
    def update_gaze(self, a: float, b: float):
        """a,b in [0,1] → keyboard normalised position."""
        self.gaze_a = float(np.clip(a, 0.0, 1.0))
        self.gaze_b = float(np.clip(b, 0.0, 1.0))

        # Which key?
        col = int(self.gaze_a * COLS)
        row = int(self.gaze_b * ROWS)
        col = min(col, COLS - 1)
        row = min(row, ROWS - 1)
        kp = (row, col)

        self._update_dwell(kp)

    def _update_dwell(self, kp):
        now = time.time()
        if kp != self.hovered:
            self.hovered  = kp
            self.dwell_t0 = now
        elif kp and self.dwell_t0 is not None:
            elapsed = now - self.dwell_t0
            if elapsed >= DWELL_TIME:
                if not (self.last_key == kp and now - self.last_key_time < COOLDOWN):
                    self._activate(kp)
                    self.last_key      = kp
                    self.last_key_time = now
                    self.dwell_t0      = now   # allow repeat

    def _activate(self, kp):
        r, c = kp
        key = KEYBOARD[r][c]
        self.flash_key = kp
        self.flash_end = time.time() + 0.4

        if   key == 'BP':  self.typed_text = self.typed_text[:-1]
        elif key == 'DL':  self.typed_text = ""
        elif key == 'PL':  self._speak(self.typed_text.strip())
        elif key in PHRASES:
            self.typed_text = PHRASES[key]
            self._speak(PHRASES[key])
        elif key == ' ':   self.typed_text += ' '
        else:              self.typed_text += key

        self._update_suggestions()

    def _update_suggestions(self):
        parts = self.typed_text.split()
        prefix = parts[-1].lower() if parts else ""
        if prefix:
            self.suggestions = [w for w in WORD_DICT if w.startswith(prefix)][:5]
        else:
            self.suggestions = []

    def apply_suggestion(self, word: str):
        parts = self.typed_text.rsplit(' ', 1)
        self.typed_text = (parts[0] + ' ' if len(parts) > 1 else '') + word + ' '
        self._update_suggestions()

    def _speak(self, text: str):
        if not text:
            return
        def _do():
            try:
                eng = pyttsx3.init()
                eng.say(text)
                eng.runAndWait()
            except Exception as e:
                print(f"[TTS] {e}")
        threading.Thread(target=_do, daemon=True).start()

    # ── Rendering ─────────────────────────────────────────────────────────────
    def draw(self, calib: MultiPointCalib) -> np.ndarray:
        frame = np.zeros((KBD_WIN_H, KBD_WIN_W, 3), dtype=np.uint8)
        frame[:] = (8, 8, 12)

        self._draw_status(frame, calib)
        self._draw_grid(frame)
        self._draw_text_box(frame)
        self._draw_suggestions(frame)
        self._draw_cursor(frame)
        return frame

    def _draw_status(self, frame, calib: MultiPointCalib):
        # Top bar
        cv2.rectangle(frame, (0, 0), (KBD_WIN_W, GRID_Y - 2), (18, 18, 28), -1)

        # Calibration overlay hint
        if calib.active and calib.current_label:
            msg = f"  ◉ Look at  {calib.current_label}  then press SPACE"
            cv2.putText(frame, msg, (6, GRID_Y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, self.status, (6, GRID_Y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (110, 110, 140), 1, cv2.LINE_AA)

        # Calibration state indicator (top-right)
        if calib.ready:
            cv2.putText(frame, "✓ 4-pt calib active", (KBD_WIN_W - 215, GRID_Y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 220, 80), 1, cv2.LINE_AA)
        elif calib.active:
            pts = sum(1 for p in calib.raw_pts if p is not None)
            cv2.putText(frame, f"● calibrating {pts}/4", (KBD_WIN_W - 215, GRID_Y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1, cv2.LINE_AA)

    def _draw_grid(self, frame):
        now = time.time()
        dwell_p = 0.0
        if self.hovered and self.dwell_t0:
            dwell_p = min((now - self.dwell_t0) / DWELL_TIME, 1.0)

        hov_col = self.hovered[1] if self.hovered else -1

        for r in range(ROWS):
            for c in range(COLS):
                key   = KEYBOARD[r][c]
                x1    = GRID_X + c * KEY_W
                y1    = GRID_Y + r * KEY_H
                x2    = x1 + KEY_W - 2
                y2    = y1 + KEY_H - 2

                is_hov    = self.hovered == (r, c)
                is_flash  = self.flash_key == (r, c) and now < self.flash_end
                is_col    = (c == hov_col) and not is_hov
                is_phrase = r == ROWS - 1

                # ── Background ─────────────────────────────────────
                if is_flash:
                    bg = (40, 210, 40)
                elif is_hov:
                    b_  = int(255 * (1 - dwell_p))
                    g_  = int(180 * dwell_p)
                    r_  = int(255 * dwell_p)
                    bg  = (b_, g_, r_)
                elif is_col:
                    bg  = (50, 50, 90)
                elif is_phrase:
                    bg  = (22, 14, 40)
                else:
                    bg  = (20, 20, 26)

                cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)

                # ── Border ─────────────────────────────────────────
                border = (0, 160, 255) if is_hov else (55, 55, 70)
                cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)

                # ── Label ──────────────────────────────────────────
                fc    = (0, 0, 0) if is_flash else (210, 210, 210)
                fs    = 0.44 if len(key) > 2 else 0.56
                tw, th = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
                tx    = x1 + (KEY_W - tw) // 2
                ty    = y1 + (KEY_H + th) // 2
                cv2.putText(frame, key, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)

                # ── Dwell arc ──────────────────────────────────────
                if is_hov and self.dwell_t0:
                    cx = x1 + KEY_W // 2
                    cy = y1 + KEY_H // 2
                    rad = min(KEY_W, KEY_H) // 2 - 3
                    cv2.ellipse(frame, (cx, cy), (rad, rad),
                                -90, 0, int(360 * dwell_p),
                                (0, 255, 180), 2, cv2.LINE_AA)

    def _draw_text_box(self, frame):
        bx1, by1 = GRID_X, TEXT_Y
        bx2, by2 = GRID_X + GRID_W, TEXT_Y + TEXT_H
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (22, 22, 30), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (70, 70, 90), 1)

        disp = self.typed_text[-90:] if len(self.typed_text) > 90 else self.typed_text
        cv2.putText(frame, disp + "|", (bx1 + 10, by1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (160, 255, 160), 1, cv2.LINE_AA)

    def _draw_suggestions(self, frame):
        if not self.suggestions:
            return
        cv2.putText(frame, "Predict:", (GRID_X, SUGG_Y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 100, 200), 1)
        for i, w in enumerate(self.suggestions):
            sx = GRID_X + 80 + i * 160
            sy = SUGG_Y
            cv2.rectangle(frame, (sx - 4, sy), (sx + 148, sy + 20), (35, 35, 60), -1)
            cv2.rectangle(frame, (sx - 4, sy), (sx + 148, sy + 20), (70, 70, 110), 1)
            cv2.putText(frame, w, (sx, sy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 195, 90), 1, cv2.LINE_AA)

    def _draw_cursor(self, frame):
        gx = int(GRID_X + self.gaze_a * GRID_W)
        gy = int(GRID_Y + self.gaze_b * GRID_H)
        r  = 11
        cv2.circle(frame, (gx, gy), r,      (0, 195, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (gx, gy), 3,      (0, 195, 255), -1)
        cv2.line(frame, (gx - r - 5, gy), (gx - r + 2, gy), (0, 195, 255), 1)
        cv2.line(frame, (gx + r - 2, gy), (gx + r + 5, gy), (0, 195, 255), 1)
        cv2.line(frame, (gx, gy - r - 5), (gx, gy - r + 2), (0, 195, 255), 1)
        cv2.line(frame, (gx, gy + r - 2), (gx, gy + r + 5), (0, 195, 255), 1)


# ═════════════════════════════════════════════════════════════════════════════
#  3D GAZE HELPERS  (your original code, lightly encapsulated)
# ═════════════════════════════════════════════════════════════════════════════
def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

def compute_scale(pts):
    n = len(pts)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            total += np.linalg.norm(pts[i] - pts[j])
            count += 1
    return total / count if count > 0 else 1.0

def pca_orientation(points_3d, ref_container):
    center  = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov     = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    if ref_container[0] is None:
        ref_container[0] = R.copy()
    else:
        for i in range(3):
            if np.dot(R[:, i], ref_container[0][:, i]) < 0:
                R[:, i] *= -1
    return center, R

def create_monitor_plane(head_center, R_final, face_landmarks, fw, fh,
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """Your original function, unchanged."""
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w  = np.array([lm_chin.x*fw, lm_chin.y*fh, lm_chin.z*fw], dtype=float)
        fore_w  = np.array([lm_fore.x*fw, lm_fore.y*fh, lm_fore.z*fw], dtype=float)
        upc     = np.linalg.norm(fore_w - chin_w) / 15.0
    except Exception:
        upc = 5.0

    mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    if gaze_origin is not None and gaze_dir is not None:
        gd = gaze_dir / np.linalg.norm(gaze_dir)
        plane_pt = head_center + head_forward * (50.0 * upc)
        denom = np.dot(head_forward, gd)
        if abs(denom) > 1e-6:
            t = np.dot(head_forward, plane_pt - gaze_origin) / denom
            center_w = gaze_origin + t * gd
        else:
            center_w = head_center + head_forward * (50.0 * upc)
    else:
        center_w = head_center + head_forward * (50.0 * upc)

    world_up  = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc


def ray_plane_ab(O, D, corners, center, normal):
    """
    Intersect ray (O + t*D) with the monitor plane.
    Returns (a, b) normalised 0..1 inside the quad, or None if miss/behind.
    """
    N = _normalize(normal)
    d = float(np.dot(N, D))
    if abs(d) < 1e-6:
        return None
    t = float(np.dot(N, (np.asarray(center) - O)) / d)
    if t < 0.0:
        return None
    P  = O + t * D
    p0, p1, _, p3 = [np.asarray(p, dtype=float) for p in corners]
    u  = p1 - p0
    v  = p3 - p0
    u2 = float(np.dot(u, u))
    v2 = float(np.dot(v, v))
    if u2 < 1e-9 or v2 < 1e-9:
        return None
    wv = P - p0
    a  = float(np.dot(wv, u) / u2)
    b  = float(np.dot(wv, v) / v2)
    return float(np.clip(a, 0.0, 1.0)), float(np.clip(b, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
#  DEBUG ORBIT VIEW  (your original render_debug_view_orbit, unchanged)
# ═════════════════════════════════════════════════════════════════════════════
def render_debug_view_orbit(dh, dw, head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None, combined_dir=None,
    gaze_len=4300, monitor_corners=None,
    monitor_center=None, monitor_normal=None,
    gaze_markers=None, units_per_cm=None):

    if head_center3d is None:
        return

    debug = np.zeros((dh, dw, 3), dtype=np.uint8)
    head_w = np.asarray(head_center3d, dtype=float)

    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    elif monitor_center is not None:
        pivot_w = (head_w + np.asarray(monitor_center)) * 0.5
    else:
        pivot_w = head_w

    f_px = _focal_px(dw, orbit_fov)
    cam_offset = _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch) @ np.array([0., 0., orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0., -1., 0.])
    fwd   = _normalize(pivot_w - cam_pos)
    right = _normalize(np.cross(fwd, up_world))
    up    = _normalize(np.cross(right, fwd))
    V     = np.stack([right, up, fwd], axis=0)

    def proj(P):
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3: return None
        x = f_px * (Pc[0]/Pc[2]) + dw*0.5
        y = -f_px * (Pc[1]/Pc[2]) + dh*0.5
        if not (np.isfinite(x) and np.isfinite(y)): return None
        return (int(x), int(y)), Pc[2]

    def draw_cross(P, sz=12, col=(255,0,255), th=2):
        res = proj(P)
        if not res: return
        (x,y),_ = res
        cv2.line(debug,(x-sz,y),(x+sz,y),col,th)
        cv2.line(debug,(x,y-sz),(x,y+sz),col,th)

    def draw_arrow(P0, P1, col=(0,200,255), th=2):
        a = proj(P0); b = proj(P1)
        if not a or not b: return
        p0,p1 = a[0],b[0]
        cv2.line(debug,p0,p1,col,th)
        v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n>1e-3:
            v/=n; l=np.array([-v[1],v[0]]); ah=9
            cv2.line(debug,p1,(int(p1[0]-v[0]*ah+l[0]*ah*.6),int(p1[1]-v[1]*ah+l[1]*ah*.6)),col,th)
            cv2.line(debug,p1,(int(p1[0]-v[0]*ah-l[0]*ah*.6),int(p1[1]-v[1]*ah-l[1]*ah*.6)),col,th)

    # Landmarks
    if landmarks3d is not None:
        for P in landmarks3d:
            res = proj(P)
            if res: cv2.circle(debug, res[0], 1, (180,180,180), -1)

    # Head center
    draw_cross(head_w, sz=12, col=(255,0,255))
    hc2d = proj(head_w)
    if hc2d:
        cv2.putText(debug,"Head",(hc2d[0][0]+8,hc2d[0][1]-8),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,0,255),1,cv2.LINE_AA)

    # Eyes
    left_dir = right_dir = None
    if left_locked and sphere_world_l is not None:
        res = proj(sphere_world_l)
        if res:
            (cx,cy),z = res
            rp = max(2,int((scaled_radius_l or 6)*f_px/max(z,1e-3)))
            cv2.circle(debug,(cx,cy),rp,(255,255,25),1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l)-np.asarray(sphere_world_l)
                p1_ = proj(np.asarray(sphere_world_l)+_normalize(left_dir)*gaze_len)
                if p1_: cv2.line(debug,(cx,cy),p1_[0],(155,155,25),1)

    if right_locked and sphere_world_r is not None:
        res = proj(sphere_world_r)
        if res:
            (cx,cy),z = res
            rp = max(2,int((scaled_radius_r or 6)*f_px/max(z,1e-3)))
            cv2.circle(debug,(cx,cy),rp,(25,255,255),1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r)-np.asarray(sphere_world_r)
                p1_ = proj(np.asarray(sphere_world_r)+_normalize(right_dir)*gaze_len)
                if p1_: cv2.line(debug,(cx,cy),p1_[0],(25,155,155),1)

    # Combined ray
    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        om = (np.asarray(sphere_world_l)+np.asarray(sphere_world_r))*0.5
        if combined_dir is not None:
            p0_ = proj(om)
            p1_ = proj(om+_normalize(combined_dir)*gaze_len*1.2)
            if p0_ and p1_:
                cv2.line(debug,p0_[0],p1_[0],(155,200,10),2)

    # Monitor plane
    if monitor_corners is not None:
        def dpoly(pts, col, th):
            pp = [proj(p) for p in pts]
            if any(x is None for x in pp): return
            p2 = [p[0] for p in pp]
            for a,b in zip(p2, p2[1:]+[p2[0]]):
                cv2.line(debug,a,b,col,th)
        dpoly(monitor_corners,(0,200,255),2)
        dpoly([monitor_corners[0],monitor_corners[2]],(0,150,210),1)
        dpoly([monitor_corners[1],monitor_corners[3]],(0,150,210),1)
        if monitor_center is not None:
            draw_cross(monitor_center,sz=8,col=(0,200,255))
            if monitor_normal is not None:
                tip = np.asarray(monitor_center)+np.asarray(monitor_normal)*(20.0*(units_per_cm or 1.0))
                draw_arrow(monitor_center, tip, col=(0,220,255))

    # Gaze hit circle on monitor
    if (monitor_corners is not None and monitor_center is not None and
            monitor_normal is not None and combined_dir is not None and
            sphere_world_l is not None and sphere_world_r is not None):
        O = (np.asarray(sphere_world_l)+np.asarray(sphere_world_r))*0.5
        D = _normalize(combined_dir)
        ab = ray_plane_ab(O, D, monitor_corners, monitor_center, monitor_normal)
        if ab:
            a_,b_ = ab
            p0c,p1c,_,p3c = [np.asarray(p,dtype=float) for p in monitor_corners]
            P = p0c + a_*(p1c-p0c) + b_*(p3c-p0c)
            u_hat = _normalize(p1c-p0c)
            r_world = 0.05*np.linalg.norm(p1c-p0c)
            pp = proj(P)
            pr = proj(P+u_hat*r_world)
            if pp and pr:
                rp_ = int(max(1,np.linalg.norm(np.array(pr[0])-np.array(pp[0]))))
                cv2.circle(debug,pp[0],rp_,(0,255,255),2,cv2.LINE_AA)

    # Gaze markers
    if gaze_markers and monitor_corners is not None:
        p0c,p1c,_,p3c = [np.asarray(p,dtype=float) for p in monitor_corners]
        u = p1c-p0c; v = p3c-p0c
        ww = float(np.linalg.norm(u)); u_hat = u/(ww+1e-9)
        for (a_,b_) in gaze_markers:
            Pm = p0c + a_*u + b_*v
            pp = proj(Pm)
            pr = proj(Pm+u_hat*0.01*ww)
            if pp and pr:
                rp_ = int(max(1,np.linalg.norm(np.array(pr[0])-np.array(pp[0]))))
                cv2.circle(debug,pp[0],rp_,(0,255,0),1,cv2.LINE_AA)

    # Help text
    help_lines = ["C=calibrate spheres","1-4=corner calib","SPACE=confirm",
                  "J/L=yaw  I/K=pitch","[/]=zoom  R=reset","X=marker  Q=quit","F7=mouse"]
    for i,t in enumerate(help_lines):
        cv2.putText(debug,t,(8,dh-10-(len(help_lines)-1-i)*18),
                    cv2.FONT_HERSHEY_SIMPLEX,.42,(180,180,180),1,cv2.LINE_AA)

    cv2.imshow("Head/Eye Debug", debug)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    global orbit_yaw, orbit_pitch, orbit_radius
    global debug_world_frozen, orbit_pivot_frozen

    # ── MediaPipe setup ────────────────────────────────────────────────────
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── State ────────────────────────────────────────────────────────────
    keyboard_gui = EyeKeyboard()
    calib        = MultiPointCalib()

    R_ref_nose   = [None]
    combined_gaze_dirs = deque(maxlen=FILTER_LEN)

    # Eye sphere state
    left_sphere_locked  = False
    right_sphere_locked = False
    left_sphere_local_offset  = None
    right_sphere_local_offset = None
    left_cal_scale  = None
    right_cal_scale = None

    # Monitor plane
    monitor_corners  = None
    monitor_center_w = None
    monitor_normal_w = None
    units_per_cm     = None

    # Persistent per-frame values (for debug view)
    sphere_world_l = sphere_world_r = None
    scaled_radius_l = scaled_radius_r = None
    iris_3d_left = iris_3d_right = None
    head_center  = None
    R_final      = None
    nose_pts_3d  = None
    avg_combined = None
    gaze_markers = []

    # Mouse thread
    mouse_enabled = False
    mouse_target  = [MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2]
    mouse_lock    = threading.Lock()

    def mouse_mover():
        while True:
            if mouse_enabled and PYAUTOGUI_OK:
                with mouse_lock:
                    x, y = mouse_target
                pyautogui.moveTo(x, y)
            time.sleep(0.01)
    threading.Thread(target=mouse_mover, daemon=True).start()

    BASE_RADIUS  = 20
    prev_f7      = False
    prev_space   = False

    print("Eye-Tracking Keyboard ready.")
    print("  C  → calibrate eye spheres + monitor plane")
    print("  1-4 → corner calibration (look at corner, press key)")
    print("  Q  → quit")

    # ── Keyboard window: fullscreen ───────────────────────────────────────
    cv2.namedWindow("Eye-Tracking Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye-Tracking Keyboard",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark

            # ── PCA head pose ─────────────────────────────────────────────
            nose_pts_3d = np.array([[lms[i].x*fw, lms[i].y*fh, lms[i].z*fw]
                                     for i in NOSE_INDICES])
            head_center, R_final = pca_orientation(nose_pts_3d, R_ref_nose)

            # ── Iris 3D positions ─────────────────────────────────────────
            li = lms[468]; ri = lms[473]
            iris_3d_left  = np.array([li.x*fw, li.y*fh, li.z*fw])
            iris_3d_right = np.array([ri.x*fw, ri.y*fh, ri.z*fw])

            # ── Draw landmarks on cam frame ───────────────────────────────
            for lm in lms:
                cv2.circle(frame, (int(lm.x*fw), int(lm.y*fh)), 0, (255,255,255), -1)

            # ── Eye sphere update ─────────────────────────────────────────
            current_ns = compute_scale(nose_pts_3d)

            if left_sphere_locked and left_sphere_local_offset is not None:
                sr = current_ns / left_cal_scale if left_cal_scale else 1.0
                sphere_world_l  = head_center + R_final @ (left_sphere_local_offset * sr)
                scaled_radius_l = int(BASE_RADIUS * sr)

            if right_sphere_locked and right_sphere_local_offset is not None:
                sr = current_ns / right_cal_scale if right_cal_scale else 1.0
                sphere_world_r  = head_center + R_final @ (right_sphere_local_offset * sr)
                scaled_radius_r = int(BASE_RADIUS * sr)

            # ── Draw eye circles on cam ───────────────────────────────────
            lx,ly = int(li.x*fw), int(li.y*fh)
            rx,ry = int(ri.x*fw), int(ri.y*fh)
            if not left_sphere_locked:
                cv2.circle(frame, (lx,ly), 10, (255,25,25), 2)
            else:
                cv2.circle(frame, (int(sphere_world_l[0]),int(sphere_world_l[1])),
                           scaled_radius_l, (255,255,25), 2)
            if not right_sphere_locked:
                cv2.circle(frame, (rx,ry), 10, (25,255,25), 2)
            else:
                cv2.circle(frame, (int(sphere_world_r[0]),int(sphere_world_r[1])),
                           scaled_radius_r, (25,255,255), 2)

            # ── Gaze computation ──────────────────────────────────────────
            if left_sphere_locked and right_sphere_locked and \
               sphere_world_l is not None and sphere_world_r is not None:

                lg = _normalize(iris_3d_left  - sphere_world_l)
                rg = _normalize(iris_3d_right - sphere_world_r)
                raw_dir = _normalize(lg + rg)
                combined_gaze_dirs.append(raw_dir)
                avg_combined = _normalize(np.mean(combined_gaze_dirs, axis=0))

                # Draw combined ray on cam frame
                O_cam = ((sphere_world_l + sphere_world_r) * 0.5).astype(int)
                T_cam = (O_cam + avg_combined * GAZE_LEN_3D).astype(int)
                cv2.line(frame, tuple(O_cam[:2]), tuple(T_cam[:2]), (255,255,10), 3)

                # ── Map gaze → keyboard (a,b) ─────────────────────────────
                if monitor_corners is not None:
                    O3 = (sphere_world_l + sphere_world_r) * 0.5
                    ab = ray_plane_ab(O3, avg_combined,
                                      monitor_corners, monitor_center_w, monitor_normal_w)
                    if ab:
                        a_raw, b_raw = ab
                        # Mirror horizontal axis: camera is flipped, iris moves
                        # right in image space when the user looks left.
                        a_raw = 1.0 - a_raw
                        a_corr, b_corr = calib.correct(a_raw, b_raw)

                        # Feed calibration sample if active
                        if calib.active:
                            calib.record(a_raw, b_raw)

                        keyboard_gui.update_gaze(a_corr, b_corr)

                        # Screen coordinates for mouse (optional)
                        sx = int(a_corr * MONITOR_WIDTH)
                        sy = int(b_corr * MONITOR_HEIGHT)
                        sx = max(10, min(sx, MONITOR_WIDTH - 10))
                        sy = max(10, min(sy, MONITOR_HEIGHT - 10))
                        if mouse_enabled:
                            with mouse_lock:
                                mouse_target[0] = sx
                                mouse_target[1] = sy
                        cv2.putText(frame, f"Gaze ({a_corr:.2f},{b_corr:.2f})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)

        # ── Draw keyboard window ──────────────────────────────────────────
        kbd_frame = keyboard_gui.draw(calib)
        cv2.imshow("Eye-Tracking Keyboard", kbd_frame)

        # ── Draw cam window ────────────────────────────────────────────────
        cv2.imshow("Integrated Eye Tracking", frame)

        # ── Debug orbit view ───────────────────────────────────────────────
        lms3d = None
        if results.multi_face_landmarks:
            lm_ = results.multi_face_landmarks[0].landmark
            lms3d = np.array([[p.x*fw, p.y*fh, p.z*fw] for p in lm_])
        render_debug_view_orbit(
            fh, fw,
            head_center3d=head_center,
            sphere_world_l=sphere_world_l if left_sphere_locked else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked else None,
            sphere_world_r=sphere_world_r if right_sphere_locked else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked else None,
            iris3d_l=iris_3d_left, iris3d_r=iris_3d_right,
            left_locked=left_sphere_locked, right_locked=right_sphere_locked,
            landmarks3d=lms3d,
            combined_dir=avg_combined,
            gaze_len=5230,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers,
            units_per_cm=units_per_cm,
        )

        # ── Orbit keyboard controls ───────────────────────────────────────
        if KB_OK:
            yaw_step = math.radians(1.5); pitch_step = math.radians(1.5)
            if kb.is_pressed('j'):   orbit_yaw   -= yaw_step
            if kb.is_pressed('l'):   orbit_yaw   += yaw_step
            if kb.is_pressed('i'):   orbit_pitch += pitch_step
            if kb.is_pressed('k'):   orbit_pitch -= pitch_step
            if kb.is_pressed('['):   orbit_radius += 12
            if kb.is_pressed(']'):   orbit_radius  = max(80., orbit_radius - 12)
            if kb.is_pressed('r'):
                orbit_yaw = 0.; orbit_pitch = 0.; orbit_radius = 600.
            orbit_pitch = max(math.radians(-89), min(math.radians(89), orbit_pitch))

            # F7 debounce
            f7_now = kb.is_pressed('f7')
            if f7_now and not prev_f7:
                mouse_enabled = not mouse_enabled
                print(f"[Mouse] {'ON' if mouse_enabled else 'OFF'}")
            prev_f7 = f7_now

        # ── cv2.waitKey events ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # ── C: calibrate eye spheres + monitor plane ──────────────────────
        elif key == ord('c') and head_center is not None and \
             not (left_sphere_locked and right_sphere_locked):

            cns = compute_scale(nose_pts_3d)
            cam_dir_local = R_final.T @ np.array([0,0,1.])

            left_sphere_local_offset  = R_final.T @ (iris_3d_left  - head_center) + BASE_RADIUS * cam_dir_local
            right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center) + BASE_RADIUS * cam_dir_local
            left_cal_scale  = cns
            right_cal_scale = cns
            left_sphere_locked  = True
            right_sphere_locked = True

            # Compute sphere positions at calibration
            swl = head_center + R_final @ left_sphere_local_offset
            swr = head_center + R_final @ right_sphere_local_offset
            lg  = _normalize(iris_3d_left  - swl)
            rg  = _normalize(iris_3d_right - swr)
            fwd = _normalize(lg + rg)

            go = (swl + swr) * 0.5
            monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = \
                create_monitor_plane(head_center, R_final, lms,
                                     fw, fh, forward_hint=fwd, gaze_origin=go, gaze_dir=fwd)

            debug_world_frozen  = True
            orbit_pivot_frozen  = monitor_center_w.copy()
            keyboard_gui.status = "Spheres locked. Press 1→2→3→4 at each corner to calibrate."
            print("[Calibration] Eye spheres locked. Monitor plane created.")
            print("  Next: look at TL corner and press 1, TR→2, BL→3, BR→4")

        # ── 1/2/3/4: start or confirm a corner calib point ───────────────
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')) and \
             left_sphere_locked and monitor_corners is not None:
            idx = key - ord('1')          # 0-3
            if not calib.active or calib.step != idx:
                calib.step   = idx
                calib.active = True
                calib._samples = []
                print(f"[Calibration] Look at {calib.LABELS[idx]}, then press SPACE.")
                keyboard_gui.status = f"Look at {calib.LABELS[idx]}, then SPACE."

        # ── SPACE: confirm a calibration point ───────────────────────────
        elif key == ord(' ') or (KB_OK and kb.is_pressed('space') and not prev_space):
            if calib.active:
                calib.confirm_point()
                if not calib.active:
                    keyboard_gui.status = "4-pt calibration complete! Type with your eyes."
                else:
                    keyboard_gui.status = f"Look at {calib.current_label}, press SPACE."
            prev_space = True

        # ── S: legacy yaw/pitch trim (when 3D calib unavailable) ─────────
        elif key == ord('s') and left_sphere_locked and avg_combined is not None:
            print("[Legacy trim] Using direct yaw/pitch offsets — prefer 4-pt calib.")

        # ── X: drop debug marker ─────────────────────────────────────────
        elif key == ord('x') and monitor_corners is not None and avg_combined is not None \
             and sphere_world_l is not None and sphere_world_r is not None:
            O3 = (sphere_world_l + sphere_world_r) * 0.5
            ab = ray_plane_ab(O3, avg_combined, monitor_corners, monitor_center_w, monitor_normal_w)
            if ab:
                gaze_markers.append(ab)
                print(f"[Marker] a={ab[0]:.3f}, b={ab[1]:.3f}")

        # Reset space debounce each frame when space is not held
        if not (KB_OK and kb.is_pressed('space')):
            prev_space = False

    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()