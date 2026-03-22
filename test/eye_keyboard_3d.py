#!/usr/bin/env python3
"""
Eye-Tracking Controlled Keyboard — 3D Gaze Edition  (v3 – High Accuracy)
=========================================================================
Integrates the full 3D eyeball-sphere tracker with a precision gaze pipeline:

  RAW 3D direction  →  ray-plane (a,b)  →  AdaptiveGazeFilter (2-D EMA)
    →  FixationDetector (I-DT)  →  SmartDwellController (fixation-gated)
    →  Key activation

The 3D tracker code (PCA head pose, sphere locking, debug orbit view) is
preserved exactly as in the original monitor-tracking program.

Controls
--------
  C          — Calibrate eye spheres + create monitor plane
  1 2 3 4   — Select corner for multi-point calibration (TL TR BL BR)
  SPACE      — Confirm current calibration corner
  X          — Drop a debug gaze marker
  F7         — Toggle OS mouse control
  J/L/I/K    — Orbit debug camera yaw/pitch
  [ / ]      — Zoom orbit in/out
  R          — Reset orbit view
  Q          — Quit

Requirements
------------
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


# ---------------------------------------------------------------------------
#  LAYOUT
# ---------------------------------------------------------------------------
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
    'P1':  "I'm hungry",         'P2':  "I want water",
    'P3':  "I'm satisfied",      'P4':  "I'm not satisfied",
    'P5':  "I want to go to the washroom",
    'P6':  "Can anyone come over here?",
    'P7':  "Could you read something for me?",
    'P8':  "Can we talk a little bit?",
    'P9':  "Can I get more",     'P10': "Thank you",
}

WORD_DICT = sorted([
    "the","be","to","of","and","a","in","that","have","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","they","we",
    "say","her","she","or","an","will","my","one","all","would","there","their",
    "what","so","up","out","if","about","who","get","which","go","me","when",
    "make","can","like","time","no","just","him","know","take","people","into",
    "year","your","good","some","could","them","see","other","than","then","now",
    "look","only","come","its","over","think","also","back","after","use","two",
    "how","our","work","first","well","way","even","new","want","because","any",
    "these","give","day","most","us","water","hungry","help","please","thank",
    "yes","need","feel","pain","tired","okay","bathroom","drink","food","medicine",
    "doctor","nurse","call","phone","read","write","talk","listen","sleep","wake",
    "sit","stand","walk","hot","cold","comfortable","uncomfortable","more","less",
    "stop","start","again","done","ready","wait","hurry","slowly","carefully",
])

# ---------------------------------------------------------------------------
#  WINDOW / GRID GEOMETRY
# ---------------------------------------------------------------------------
KBD_WIN_W  = MONITOR_WIDTH
KBD_WIN_H  = MONITOR_HEIGHT
GRID_X     = 18
GRID_Y     = 36
GRID_W     = KBD_WIN_W - 36
GRID_H     = int(KBD_WIN_H * 0.60)
TEXT_Y     = GRID_Y + GRID_H + 20
TEXT_H     = 64
SUGG_Y     = TEXT_Y + TEXT_H + 10

KEY_W = GRID_W // COLS
KEY_H = GRID_H // ROWS

# ---------------------------------------------------------------------------
#  ORBIT DEBUG GLOBALS
# ---------------------------------------------------------------------------
orbit_yaw    = math.radians(-151.0)
orbit_pitch  = 0.0
orbit_radius = 1500.0
orbit_fov    = 50.0
debug_world_frozen = False
orbit_pivot_frozen = None

NOSE_INDICES = [4,45,275,220,440,1,5,51,281,44,274,241,
                461,125,354,218,438,195,167,393,165,391,3,248]


# ===========================================================================
#  GAZE PIPELINE
# ===========================================================================

class AdaptiveGazeFilter:
    """
    Velocity-aware EMA in (a,b) space — not in 3D direction space.

    Root cause of old jitter: the previous code averaged 3D direction
    vectors over 12 frames, THEN projected onto the monitor plane.
    Angular noise * long projection distance = large pixel displacement.
    Filtering in 2D after the ray-plane intersection is far more stable.

    Two modes:
      SACCADE  (speed > 0.10/s)  alpha = 0.08  — heavy; cursor lags through flight
      FIXATION (speed < 0.025/s) alpha = 0.55  — light; cursor snaps to fixation
    Between thresholds: linearly interpolated.
    """
    SACCADE_THRESH  = 0.10
    FIXATION_THRESH = 0.025
    ALPHA_SACCADE   = 0.08
    ALPHA_FIXATION  = 0.55

    def __init__(self):
        self.a = 0.5
        self.b = 0.5
        self._buf = deque(maxlen=6)
        self.speed = 0.0

    def update(self, a_raw, b_raw):
        now = time.time()
        self._buf.append((a_raw, b_raw, now))
        if len(self._buf) >= 3:
            old = self._buf[-3]
            dt_h = now - old[2]
            if dt_h > 5e-4:
                self.speed = math.hypot(a_raw - old[0], b_raw - old[1]) / dt_h

        s = self.speed
        if s >= self.SACCADE_THRESH:
            alpha = self.ALPHA_SACCADE
        elif s <= self.FIXATION_THRESH:
            alpha = self.ALPHA_FIXATION
        else:
            t = (s - self.FIXATION_THRESH) / (self.SACCADE_THRESH - self.FIXATION_THRESH)
            alpha = self.ALPHA_FIXATION * (1.0 - t) + self.ALPHA_SACCADE * t

        self.a = alpha * a_raw + (1.0 - alpha) * self.a
        self.b = alpha * b_raw + (1.0 - alpha) * self.b
        return self.a, self.b, self.speed

    def reset(self):
        self.a = 0.5
        self.b = 0.5
        self._buf.clear()
        self.speed = 0.0


class FixationDetector:
    """
    I-DT (dispersion-threshold) fixation identification.

    Maintains a rolling window of N filtered samples.
    A fixation is declared when:
      - spatial dispersion of the window < DISPERSION_MAX
      - current filtered speed < SPEED_MAX

    DISPERSION_MAX = 0.028 in (a,b) units ~ 1/3 of one key width.
    This means the eye must be stable within one key before dwell starts.

    During fixation, centroid_a/b gives the mean position — more stable
    than the instantaneous filtered value.
    """
    WINDOW         = 18
    DISPERSION_MAX = 0.028
    SPEED_MAX      = 0.045
    MIN_SAMPLES    = 8

    def __init__(self):
        self._buf = deque(maxlen=self.WINDOW)
        self.is_fixating  = False
        self.centroid_a   = 0.5
        self.centroid_b   = 0.5
        self.dispersion   = 1.0

    def update(self, a, b, speed):
        self._buf.append((a, b))
        if len(self._buf) < self.MIN_SAMPLES:
            self.is_fixating = False
            return False
        arr = np.array(self._buf)
        a_range = float(arr[:, 0].max() - arr[:, 0].min())
        b_range = float(arr[:, 1].max() - arr[:, 1].min())
        self.dispersion  = math.hypot(a_range, b_range)
        self.is_fixating = (self.dispersion < self.DISPERSION_MAX
                            and speed < self.SPEED_MAX)
        if self.is_fixating:
            self.centroid_a = float(arr[:, 0].mean())
            self.centroid_b = float(arr[:, 1].mean())
        return self.is_fixating

    def reset(self):
        self._buf.clear()
        self.is_fixating = False
        self.dispersion  = 1.0


class SmartDwellController:
    """
    Fixation-gated dwell with key hysteresis.

    What the old dwell got wrong
    ----------------------------
    It counted wall-clock time on the same (row,col). Any noise that kept
    the cursor near a key boundary for 1.4 s would fire. Noise at the
    boundary also constantly reset the timer every frame.

    What this version does
    ----------------------
    1. Dwell only accumulates when FixationDetector says fixating.
    2. Key hysteresis: require CONFIRM_FRAMES consecutive fixating frames
       on the same cell before that cell becomes "hovered". Eliminates
       flicker at key boundaries entirely.
    3. Dwell decays (not hard-resets) during saccades: brief blinks or
       noise don't erase all progress.
    4. Per-key cooldown prevents double-firing.
    """
    DWELL_TIME     = 1.3
    COOLDOWN       = 0.9
    CONFIRM_FRAMES = 5

    def __init__(self):
        self.hovered          = None
        self._candidate       = None
        self._candidate_cnt   = 0
        self._dwell_accum     = 0.0
        self._last_act_key    = None
        self._last_act_time   = 0.0
        self.dwell_progress   = 0.0
        self.activated_key    = None

    def update(self, centroid_a, centroid_b, is_fixating, dt):
        self.activated_key = None
        now = time.time()

        col = min(int(centroid_a * COLS), COLS - 1)
        row = min(int(centroid_b * ROWS), ROWS - 1)
        kp  = (row, col)
        on_kbd = (0.0 <= centroid_a <= 1.0 and 0.0 <= centroid_b <= 1.0)

        # Key hysteresis
        if on_kbd and is_fixating:
            if kp == self._candidate:
                self._candidate_cnt += 1
            else:
                self._candidate     = kp
                self._candidate_cnt = 1
            if self._candidate_cnt >= self.CONFIRM_FRAMES:
                if kp != self.hovered:
                    self.hovered      = kp
                    self._dwell_accum = 0.0

        # Dwell accumulation
        if self.hovered and is_fixating and on_kbd:
            self._dwell_accum += dt
        else:
            self._dwell_accum = max(0.0, self._dwell_accum - dt * 1.8)

        self.dwell_progress = (min(self._dwell_accum / self.DWELL_TIME, 1.0)
                               if self.DWELL_TIME > 0 else 0.0)

        # Activation
        if self._dwell_accum >= self.DWELL_TIME and self.hovered:
            ok = not (self._last_act_key == self.hovered
                      and now - self._last_act_time < self.COOLDOWN)
            if ok:
                self.activated_key  = self.hovered
                self._last_act_key  = self.hovered
                self._last_act_time = now
                self._dwell_accum   = 0.0

        return self.activated_key


# ===========================================================================
#  MULTI-POINT CALIBRATION  (stable-sample-only version)
# ===========================================================================
class MultiPointCalib:
    """
    Collects raw (a,b) at 4 corners using only stable (low-dispersion) frames.

    Old problem: accumulated ALL frames — early jittery frames polluted the
    calibration point.
    Fix: only accept samples when FixationDetector.dispersion < threshold.
    Confirmation uses median (robust to outliers).
    """
    TARGETS = [(0.0,0.0),(1.0,0.0),(0.0,1.0),(1.0,1.0)]
    LABELS  = ["Top-Left","Top-Right","Bottom-Left","Bottom-Right"]
    STABLE_DISP_MAX = 0.025
    STABLE_WINDOW   = 30
    MIN_STABLE      = 8

    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_pts         = [None] * 4
        self.active          = False
        self.step            = 0
        self._all_samples    = deque(maxlen=80)
        self._stable_samples = []
        self._done           = False
        self._H              = None

    def record(self, a, b, dispersion):
        if not self.active:
            return
        self._all_samples.append((a, b))
        if dispersion < self.STABLE_DISP_MAX:
            self._stable_samples.append((a, b))
            if len(self._stable_samples) > self.STABLE_WINDOW:
                self._stable_samples.pop(0)

    def confirm_point(self):
        if not self.active:
            return False
        samples = (self._stable_samples
                   if len(self._stable_samples) >= self.MIN_STABLE
                   else list(self._all_samples))
        if not samples:
            print("[Calibration] No samples — keep looking at the corner.")
            return False
        arr = np.array(samples)
        ma  = float(np.median(arr[:, 0]))
        mb  = float(np.median(arr[:, 1]))
        self.raw_pts[self.step] = (ma, mb)
        print(f"[Calibration] {self.LABELS[self.step]} "
              f"raw=({ma:.4f},{mb:.4f})  "
              f"({len(self._stable_samples)} stable frames)")
        self._all_samples.clear()
        self._stable_samples = []
        self.step += 1
        if self.step >= 4:
            self.active = False
            self._done  = True
            self._build_homography()
        else:
            print(f"[Calibration] Now look at {self.LABELS[self.step]}. Press SPACE.")
        return True

    def _build_homography(self):
        src = np.array(self.raw_pts, dtype=np.float32)
        dst = np.array(self.TARGETS, dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst)
        if self._H is not None:
            print("[Calibration] Homography built — correction active.")
        else:
            print("[Calibration] Homography failed — redo corner calibration.")

    def correct(self, a, b):
        if self._H is None:
            return a, b
        pt  = np.array([[[a, b]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return (float(np.clip(out[0,0,0], -0.1, 1.1)),
                float(np.clip(out[0,0,1], -0.1, 1.1)))

    @property
    def ready(self):
        return self._done and self._H is not None

    @property
    def current_label(self):
        return self.LABELS[self.step] if self.active and self.step < 4 else None

    @property
    def stable_count(self):
        return len(self._stable_samples)


# ===========================================================================
#  KEYBOARD GUI  (rendering only — all gaze logic in pipeline classes)
# ===========================================================================
class EyeKeyboard:

    def __init__(self):
        self.typed_text  = ""
        self.suggestions = []
        self.flash_key   = None
        self.flash_end   = 0.0
        self.status      = "Press C to calibrate (look straight at the screen)."

    def activate_key(self, kp):
        r, c = kp
        key  = KEYBOARD[r][c]
        self.flash_key = kp
        self.flash_end = time.time() + 0.4
        if   key == 'BP':  self.typed_text = self.typed_text[:-1]
        elif key == 'DL':  self.typed_text = ""
        elif key == 'PL':  self._speak(self.typed_text.strip())
        elif key in PHRASES:
            self.typed_text = PHRASES[key]; self._speak(PHRASES[key])
        elif key == ' ':   self.typed_text += ' '
        else:              self.typed_text += key
        self._update_suggestions()

    def _update_suggestions(self):
        parts  = self.typed_text.split()
        prefix = parts[-1].lower() if parts else ""
        self.suggestions = ([w for w in WORD_DICT if w.startswith(prefix)][:5]
                            if prefix else [])

    def _speak(self, text):
        if not text or not TTS_OK: return
        def _do():
            try:
                e = pyttsx3.init(); e.say(text); e.runAndWait()
            except Exception as ex:
                print(f"[TTS] {ex}")
        threading.Thread(target=_do, daemon=True).start()

    def draw(self, dwell, fixation, gaze_filter, calib):
        frame = np.zeros((KBD_WIN_H, KBD_WIN_W, 3), dtype=np.uint8)
        frame[:] = (8, 8, 12)
        self._draw_status(frame, calib, fixation)
        self._draw_grid(frame, dwell)
        self._draw_text_box(frame)
        self._draw_suggestions(frame)
        self._draw_cursors(frame, dwell, fixation, gaze_filter)
        return frame

    def _draw_status(self, frame, calib, fixation):
        cv2.rectangle(frame, (0, 0), (KBD_WIN_W, GRID_Y - 2), (14,14,22), -1)
        if calib.active and calib.current_label:
            n   = calib.stable_count
            bar = min(n, calib.MIN_STABLE)
            msg = (f"  Look at {calib.current_label}  "
                   f"({bar}/{calib.MIN_STABLE} stable frames) → SPACE to confirm")
            cv2.putText(frame, msg, (6, GRID_Y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, self.status, (6, GRID_Y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (110,110,140), 1, cv2.LINE_AA)

        badge = ("✓ 4-pt calib" if calib.ready
                 else (f"● calib {sum(1 for p in calib.raw_pts if p is not None)}/4"
                       if calib.active else ""))
        if badge:
            col = (60,220,80) if calib.ready else (0,200,255)
            cv2.putText(frame, badge, (KBD_WIN_W-185, GRID_Y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

        fix_col = (0,255,120) if fixation.is_fixating else (80,80,80)
        cv2.circle(frame, (KBD_WIN_W-14, GRID_Y-12), 7, fix_col, -1)

    def _draw_grid(self, frame, dwell):
        now = time.time()
        dp  = dwell.dwell_progress
        hov = dwell.hovered
        hov_col = hov[1] if hov else -1

        for r in range(ROWS):
            for c in range(COLS):
                key = KEYBOARD[r][c]
                x1  = GRID_X + c * KEY_W
                y1  = GRID_Y + r * KEY_H
                x2  = x1 + KEY_W - 2
                y2  = y1 + KEY_H - 2

                is_hov    = hov == (r, c)
                is_flash  = self.flash_key == (r,c) and now < self.flash_end
                is_col    = (c == hov_col) and not is_hov
                is_phrase = r == ROWS - 1

                if   is_flash:   bg = (30, 200, 30)
                elif is_hov:
                    b_ = int(255*(1-dp)); g_ = int(180*dp); r_ = int(255*dp)
                    bg = (b_, g_, r_)
                elif is_col:     bg = (45, 45, 80)
                elif is_phrase:  bg = (20, 12, 38)
                else:            bg = (18, 18, 24)

                cv2.rectangle(frame, (x1,y1), (x2,y2), bg, -1)
                border = (0,150,255) if is_hov else (50,50,65)
                cv2.rectangle(frame, (x1,y1), (x2,y2), border, 1)

                fc = (0,0,0) if is_flash else (210,210,210)
                fs = 0.44 if len(key) > 2 else 0.58
                tw, th = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
                cv2.putText(frame, key,
                            (x1+(KEY_W-tw)//2, y1+(KEY_H+th)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)

                if is_hov and dp > 0.01:
                    cx = x1 + KEY_W//2;  cy = y1 + KEY_H//2
                    rad = min(KEY_W, KEY_H)//2 - 3
                    cv2.ellipse(frame, (cx,cy), (rad,rad),
                                -90, 0, int(360*dp),
                                (0,255,180), 2, cv2.LINE_AA)

    def _draw_text_box(self, frame):
        bx1,by1 = GRID_X, TEXT_Y
        bx2,by2 = GRID_X+GRID_W, TEXT_Y+TEXT_H
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (20,20,28), -1)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (65,65,88), 1)
        disp = self.typed_text[-90:] if len(self.typed_text) > 90 else self.typed_text
        cv2.putText(frame, disp+"|", (bx1+10, by1+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.76, (150,255,150), 1, cv2.LINE_AA)

    def _draw_suggestions(self, frame):
        if not self.suggestions: return
        cv2.putText(frame, "Predict:", (GRID_X, SUGG_Y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100,100,200), 1)
        for i, w in enumerate(self.suggestions):
            sx = GRID_X + 82 + i*165;  sy = SUGG_Y
            cv2.rectangle(frame, (sx-4,sy), (sx+154,sy+22), (32,32,55), -1)
            cv2.rectangle(frame, (sx-4,sy), (sx+154,sy+22), (65,65,110), 1)
            cv2.putText(frame, w, (sx, sy+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,192,80), 1, cv2.LINE_AA)

    def _draw_cursors(self, frame, dwell, fixation, gaze_filter):
        # 1. Display cursor (large crosshair — follows smooth filter)
        da = float(np.clip(gaze_filter.a, 0.0, 1.0))
        db = float(np.clip(gaze_filter.b, 0.0, 1.0))
        gx = int(GRID_X + da * GRID_W)
        gy = int(GRID_Y + db * GRID_H)
        r  = 13
        col = (0,195,255) if fixation.is_fixating else (0,100,180)
        cv2.circle(frame, (gx,gy), r, col, 1, cv2.LINE_AA)
        cv2.circle(frame, (gx,gy), 2, col, -1)
        cv2.line(frame, (gx-r-6,gy),(gx-r+2,gy), col, 1)
        cv2.line(frame, (gx+r-2,gy),(gx+r+6,gy), col, 1)
        cv2.line(frame, (gx,gy-r-6),(gx,gy-r+2), col, 1)
        cv2.line(frame, (gx,gy+r-2),(gx,gy+r+6), col, 1)

        # 2. Fixation centroid dot (only during fixation — this drives dwell)
        if fixation.is_fixating:
            fa = float(np.clip(fixation.centroid_a, 0.0, 1.0))
            fb = float(np.clip(fixation.centroid_b, 0.0, 1.0))
            fx = int(GRID_X + fa * GRID_W)
            fy = int(GRID_Y + fb * GRID_H)
            cv2.circle(frame, (fx,fy), 5, (0,255,180), -1, cv2.LINE_AA)
            cv2.circle(frame, (fx,fy), 5, (255,255,255),  1, cv2.LINE_AA)


# ===========================================================================
#  3D TRACKER HELPERS  (original code — untouched)
# ===========================================================================
def _rot_x(a):
    ca,sa = math.cos(a),math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def _rot_y(a):
    ca,sa = math.cos(a),math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v/n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    return 0.5*width/math.tan(math.radians(fov_deg)*0.5)

def compute_scale(pts):
    n = len(pts); total = count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += np.linalg.norm(pts[i]-pts[j]); count += 1
    return total/count if count > 0 else 1.0

def pca_orientation(points_3d, ref_container):
    center   = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov      = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs  = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0: eigvecs[:,2] *= -1
    r = Rscipy.from_matrix(eigvecs)
    roll,pitch,yaw = r.as_euler('zyx', degrees=False)
    R = Rscipy.from_euler('zyx', [roll,pitch,yaw]).as_matrix()
    if ref_container[0] is None:
        ref_container[0] = R.copy()
    else:
        for i in range(3):
            if np.dot(R[:,i], ref_container[0][:,i]) < 0: R[:,i] *= -1
    return center, R

def create_monitor_plane(head_center, R_final, face_landmarks, fw, fh,
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    try:
        lc = face_landmarks[152]; lf = face_landmarks[10]
        chin_w = np.array([lc.x*fw,lc.y*fh,lc.z*fw], dtype=float)
        fore_w = np.array([lf.x*fw,lf.y*fh,lf.z*fw], dtype=float)
        upc = np.linalg.norm(fore_w-chin_w)/15.0
    except Exception:
        upc = 5.0
    half_w = 30.0*upc; half_h = 20.0*upc
    head_fwd = -R_final[:,2]
    if forward_hint is not None:
        head_fwd = forward_hint/np.linalg.norm(forward_hint)
    if gaze_origin is not None and gaze_dir is not None:
        gd = gaze_dir/np.linalg.norm(gaze_dir)
        pp = head_center + head_fwd*(50.0*upc)
        dn = np.dot(head_fwd, gd)
        center_w = (gaze_origin + np.dot(head_fwd, pp-gaze_origin)/dn*gd
                    if abs(dn)>1e-6 else head_center+head_fwd*(50.0*upc))
    else:
        center_w = head_center + head_fwd*(50.0*upc)
    world_up  = np.array([0,-1,0], dtype=float)
    head_right = np.cross(world_up, head_fwd); head_right/=np.linalg.norm(head_right)
    head_up    = np.cross(head_fwd, head_right); head_up/=np.linalg.norm(head_up)
    p0 = center_w - head_right*half_w - head_up*half_h
    p1 = center_w + head_right*half_w - head_up*half_h
    p2 = center_w + head_right*half_w + head_up*half_h
    p3 = center_w - head_right*half_w + head_up*half_h
    return [p0,p1,p2,p3], center_w, head_fwd/np.linalg.norm(head_fwd), upc

def ray_plane_ab(O, D, corners, center, normal):
    """Returns UNCLIPPED (a,b) — may exceed [0,1]. Returns None if behind eye."""
    N = _normalize(normal); d = float(np.dot(N,D))
    if abs(d) < 1e-6: return None
    t = float(np.dot(N, np.asarray(center)-O)/d)
    if t < 0.0: return None
    P = O + t*D
    p0,p1,_,p3 = [np.asarray(p,dtype=float) for p in corners]
    u=p1-p0; v=p3-p0
    u2=float(np.dot(u,u)); v2=float(np.dot(v,v))
    if u2<1e-9 or v2<1e-9: return None
    wv=P-p0
    return float(np.dot(wv,u)/u2), float(np.dot(wv,v)/v2)


# ===========================================================================
#  DEBUG ORBIT VIEW  (original — untouched)
# ===========================================================================
def render_debug_view_orbit(dh, dw, head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None, combined_dir=None,
    gaze_len=4300, monitor_corners=None,
    monitor_center=None, monitor_normal=None,
    gaze_markers=None, units_per_cm=None):

    if head_center3d is None: return
    debug  = np.zeros((dh,dw,3), dtype=np.uint8)
    head_w = np.asarray(head_center3d, dtype=float)

    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    elif monitor_center is not None:
        pivot_w = (head_w+np.asarray(monitor_center))*0.5
    else:
        pivot_w = head_w

    f_px   = _focal_px(dw, orbit_fov)
    cam_pos= pivot_w+_rot_y(orbit_yaw)@(_rot_x(orbit_pitch)@np.array([0.,0.,orbit_radius]))
    fwd    = _normalize(pivot_w-cam_pos)
    right  = _normalize(np.cross(fwd,np.array([0.,-1.,0.])))
    up     = _normalize(np.cross(right,fwd))
    V      = np.stack([right,up,fwd],axis=0)

    def proj(P):
        Pc = V@(np.asarray(P,dtype=float)-cam_pos)
        if Pc[2]<=1e-3: return None
        x=f_px*(Pc[0]/Pc[2])+dw*0.5; y=-f_px*(Pc[1]/Pc[2])+dh*0.5
        return ((int(x),int(y)),Pc[2]) if np.isfinite(x) and np.isfinite(y) else None

    def dcross(P,sz=12,col=(255,0,255),th=2):
        r=proj(P)
        if not r: return
        x,y=r[0]; cv2.line(debug,(x-sz,y),(x+sz,y),col,th); cv2.line(debug,(x,y-sz),(x,y+sz),col,th)

    def darrow(P0,P1,col=(0,200,255),th=2):
        a_=proj(P0); b_=proj(P1)
        if not a_ or not b_: return
        p0_,p1_=a_[0],b_[0]; cv2.line(debug,p0_,p1_,col,th)
        v=np.array([p1_[0]-p0_[0],p1_[1]-p0_[1]],dtype=float); n=np.linalg.norm(v)
        if n>1e-3:
            v/=n; l=np.array([-v[1],v[0]]); ah=9
            cv2.line(debug,p1_,(int(p1_[0]-v[0]*ah+l[0]*ah*.6),int(p1_[1]-v[1]*ah+l[1]*ah*.6)),col,th)
            cv2.line(debug,p1_,(int(p1_[0]-v[0]*ah-l[0]*ah*.6),int(p1_[1]-v[1]*ah-l[1]*ah*.6)),col,th)

    if landmarks3d is not None:
        for P in landmarks3d:
            r=proj(P)
            if r: cv2.circle(debug,r[0],1,(180,180,180),-1)

    dcross(head_w,sz=12,col=(255,0,255))
    hc=proj(head_w)
    if hc: cv2.putText(debug,"Head",(hc[0][0]+8,hc[0][1]-8),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,0,255),1)

    if left_locked and sphere_world_l is not None:
        res=proj(sphere_world_l)
        if res:
            (cx,cy),z=res; rp=max(2,int((scaled_radius_l or 6)*f_px/max(z,1e-3)))
            cv2.circle(debug,(cx,cy),rp,(255,255,25),1)
            if iris3d_l is not None:
                ld=np.asarray(iris3d_l)-np.asarray(sphere_world_l)
                p1_=proj(np.asarray(sphere_world_l)+_normalize(ld)*gaze_len)
                if p1_: cv2.line(debug,(cx,cy),p1_[0],(155,155,25),1)
    if right_locked and sphere_world_r is not None:
        res=proj(sphere_world_r)
        if res:
            (cx,cy),z=res; rp=max(2,int((scaled_radius_r or 6)*f_px/max(z,1e-3)))
            cv2.circle(debug,(cx,cy),rp,(25,255,255),1)
            if iris3d_r is not None:
                rd=np.asarray(iris3d_r)-np.asarray(sphere_world_r)
                p1_=proj(np.asarray(sphere_world_r)+_normalize(rd)*gaze_len)
                if p1_: cv2.line(debug,(cx,cy),p1_[0],(25,155,155),1)
    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        om=(np.asarray(sphere_world_l)+np.asarray(sphere_world_r))*0.5
        if combined_dir is not None:
            p0_=proj(om); p1_=proj(om+_normalize(combined_dir)*gaze_len*1.2)
            if p0_ and p1_: cv2.line(debug,p0_[0],p1_[0],(155,200,10),2)

    if monitor_corners is not None:
        def dpoly(pts,col,th):
            pp=[proj(p) for p in pts]
            if any(x is None for x in pp): return
            p2=[p[0] for p in pp]
            for a_,b_ in zip(p2,p2[1:]+[p2[0]]): cv2.line(debug,a_,b_,col,th)
        dpoly(monitor_corners,(0,200,255),2)
        dpoly([monitor_corners[0],monitor_corners[2]],(0,150,210),1)
        dpoly([monitor_corners[1],monitor_corners[3]],(0,150,210),1)
        if monitor_center is not None:
            dcross(monitor_center,sz=8,col=(0,200,255))
            if monitor_normal is not None:
                tip=np.asarray(monitor_center)+np.asarray(monitor_normal)*(20.0*(units_per_cm or 1.0))
                darrow(monitor_center,tip,col=(0,220,255))

    if (monitor_corners and monitor_center is not None and monitor_normal is not None
            and combined_dir is not None
            and sphere_world_l is not None and sphere_world_r is not None):
        O_=(np.asarray(sphere_world_l)+np.asarray(sphere_world_r))*0.5
        ab=ray_plane_ab(O_,_normalize(combined_dir),monitor_corners,monitor_center,monitor_normal)
        if ab:
            a_,b_=ab
            p0c,p1c,_,p3c=[np.asarray(p,dtype=float) for p in monitor_corners]
            P_=p0c+a_*(p1c-p0c)+b_*(p3c-p0c)
            uh=_normalize(p1c-p0c); rw=0.05*np.linalg.norm(p1c-p0c)
            pp=proj(P_); pr=proj(P_+uh*rw)
            if pp and pr:
                rp_=int(max(1,np.linalg.norm(np.array(pr[0])-np.array(pp[0]))))
                cv2.circle(debug,pp[0],rp_,(0,255,255),2,cv2.LINE_AA)

    if gaze_markers and monitor_corners is not None:
        p0c,p1c,_,p3c=[np.asarray(p,dtype=float) for p in monitor_corners]
        u=p1c-p0c; v=p3c-p0c; ww=np.linalg.norm(u); uh=u/(ww+1e-9)
        for (a_,b_) in gaze_markers:
            Pm=p0c+a_*u+b_*v; pp=proj(Pm); pr=proj(Pm+uh*0.01*ww)
            if pp and pr:
                rp_=int(max(1,np.linalg.norm(np.array(pr[0])-np.array(pp[0]))))
                cv2.circle(debug,pp[0],rp_,(0,255,0),1,cv2.LINE_AA)

    help_lines = ["C=calib spheres","1-4=corners","SPACE=confirm",
                  "J/L=yaw I/K=pitch","[/]=zoom R=reset","X=marker Q=quit","F7=mouse"]
    for i,t in enumerate(help_lines):
        cv2.putText(debug,t,(8,dh-10-(len(help_lines)-1-i)*18),
                    cv2.FONT_HERSHEY_SIMPLEX,.42,(180,180,180),1,cv2.LINE_AA)
    cv2.imshow("Head/Eye Debug", debug)


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    global orbit_yaw,orbit_pitch,orbit_radius,debug_world_frozen,orbit_pivot_frozen

    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pipeline
    keyboard_gui = EyeKeyboard()
    calib        = MultiPointCalib()
    gaze_filter  = AdaptiveGazeFilter()
    fixation_det = FixationDetector()
    dwell_ctrl   = SmartDwellController()

    # 3D tracker state
    R_ref_nose   = [None]
    raw_dir_buf  = deque(maxlen=6)   # minimal 3D smoothing for debug view only

    left_sphere_locked  = right_sphere_locked  = False
    left_sphere_local_offset = right_sphere_local_offset = None
    left_cal_scale = right_cal_scale = None

    monitor_corners = monitor_center_w = monitor_normal_w = units_per_cm = None
    sphere_world_l  = sphere_world_r   = None
    scaled_radius_l = scaled_radius_r  = None
    iris_3d_left    = iris_3d_right    = None
    head_center     = R_final          = None
    nose_pts_3d     = lms              = None
    avg_combined    = None
    gaze_markers    = []

    # Mouse
    mouse_enabled = False
    mouse_target  = [MONITOR_WIDTH//2, MONITOR_HEIGHT//2]
    mouse_lock    = threading.Lock()
    def mouse_mover():
        while True:
            if mouse_enabled and PYAUTOGUI_OK:
                with mouse_lock: x,y=mouse_target
                pyautogui.moveTo(x,y)
            time.sleep(0.01)
    threading.Thread(target=mouse_mover, daemon=True).start()

    BASE_RADIUS  = 20
    prev_f7      = False
    prev_space   = False
    last_frame_t = time.time()

    cv2.namedWindow("Eye-Tracking Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye-Tracking Keyboard",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Eye-Tracking Keyboard v3")
    print("  Step 1: C — calibrate spheres (look straight at screen)")
    print("  Step 2: 1→TL, 2→TR, 3→BL, 4→BR + SPACE each corner")
    print("  Q — quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        now  = time.time()
        dt   = min(now - last_frame_t, 0.10)
        last_frame_t = now

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark

            nose_pts_3d = np.array([[lms[i].x*fw, lms[i].y*fh, lms[i].z*fw]
                                     for i in NOSE_INDICES])
            head_center, R_final = pca_orientation(nose_pts_3d, R_ref_nose)

            li = lms[468]; ri = lms[473]
            iris_3d_left  = np.array([li.x*fw, li.y*fh, li.z*fw])
            iris_3d_right = np.array([ri.x*fw, ri.y*fh, ri.z*fw])

            for lm in lms:
                cv2.circle(frame,(int(lm.x*fw),int(lm.y*fh)),0,(255,255,255),-1)

            cns = compute_scale(nose_pts_3d)
            if left_sphere_locked and left_sphere_local_offset is not None:
                sr = cns/left_cal_scale if left_cal_scale else 1.0
                sphere_world_l  = head_center+R_final@(left_sphere_local_offset*sr)
                scaled_radius_l = int(BASE_RADIUS*sr)
            if right_sphere_locked and right_sphere_local_offset is not None:
                sr = cns/right_cal_scale if right_cal_scale else 1.0
                sphere_world_r  = head_center+R_final@(right_sphere_local_offset*sr)
                scaled_radius_r = int(BASE_RADIUS*sr)

            lx,ly=int(li.x*fw),int(li.y*fh)
            rx,ry=int(ri.x*fw),int(ri.y*fh)
            if not left_sphere_locked:
                cv2.circle(frame,(lx,ly),10,(255,25,25),2)
            else:
                cv2.circle(frame,(int(sphere_world_l[0]),int(sphere_world_l[1])),
                           scaled_radius_l,(255,255,25),2)
            if not right_sphere_locked:
                cv2.circle(frame,(rx,ry),10,(25,255,25),2)
            else:
                cv2.circle(frame,(int(sphere_world_r[0]),int(sphere_world_r[1])),
                           scaled_radius_r,(25,255,255),2)

            if (left_sphere_locked and right_sphere_locked
                    and sphere_world_l is not None and sphere_world_r is not None):
                lg = _normalize(iris_3d_left  - sphere_world_l)
                rg = _normalize(iris_3d_right - sphere_world_r)
                raw_dir = _normalize(lg+rg)
                raw_dir_buf.append(raw_dir)
                avg_combined = _normalize(np.mean(raw_dir_buf,axis=0))

                Oc=((sphere_world_l+sphere_world_r)*0.5).astype(int)
                Tc=(Oc+avg_combined*350).astype(int)
                cv2.line(frame,tuple(Oc[:2]),tuple(Tc[:2]),(255,255,10),3)

                if monitor_corners is not None:
                    O3  = (sphere_world_l+sphere_world_r)*0.5
                    ab  = ray_plane_ab(O3, avg_combined,
                                       monitor_corners, monitor_center_w, monitor_normal_w)
                    if ab:
                        a_raw, b_raw = ab
                        a_raw = 1.0 - a_raw   # mirror horizontal (camera is flipped)
                        b_raw = 1.0 - b_raw   # mirror vertical (camera is flipped)

                        # Stage 1: adaptive 2D filter
                        a_f, b_f, speed = gaze_filter.update(a_raw, b_raw)

                        # Stage 2: fixation detection
                        is_fix = fixation_det.update(a_f, b_f, speed)

                        # Feed calibration (stable frames only)
                        if calib.active:
                            calib.record(a_raw, b_raw, fixation_det.dispersion)

                        # Apply homography to fixation centroid
                        fa_c, fb_c = calib.correct(
                            fixation_det.centroid_a, fixation_det.centroid_b)

                        # Stage 3: smart dwell
                        activated = dwell_ctrl.update(fa_c, fb_c, is_fix, dt)
                        if activated:
                            keyboard_gui.activate_key(activated)

                        # OS mouse
                        a_fc, b_fc = calib.correct(a_f, b_f)
                        sx=int(np.clip(a_fc,0,1)*MONITOR_WIDTH)
                        sy=int(np.clip(b_fc,0,1)*MONITOR_HEIGHT)
                        if mouse_enabled:
                            with mouse_lock: mouse_target[0]=sx; mouse_target[1]=sy

                        label = "FIX" if is_fix else f"sac {speed:.3f}"
                        col_  = (0,255,120) if is_fix else (0,120,255)
                        cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,col_,1)

        # Draw windows
        kbd = keyboard_gui.draw(dwell_ctrl, fixation_det, gaze_filter, calib)
        cv2.imshow("Eye-Tracking Keyboard", kbd)
        cv2.imshow("Integrated Eye Tracking", frame)

        lms3d = None
        if results.multi_face_landmarks:
            lms3d=np.array([[p.x*fw,p.y*fh,p.z*fw]
                             for p in results.multi_face_landmarks[0].landmark])
        render_debug_view_orbit(
            fh,fw, head_center3d=head_center,
            sphere_world_l=sphere_world_l if left_sphere_locked else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked else None,
            sphere_world_r=sphere_world_r if right_sphere_locked else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked else None,
            iris3d_l=iris_3d_left, iris3d_r=iris_3d_right,
            left_locked=left_sphere_locked, right_locked=right_sphere_locked,
            landmarks3d=lms3d, combined_dir=avg_combined,
            gaze_len=5230, monitor_corners=monitor_corners,
            monitor_center=monitor_center_w, monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers, units_per_cm=units_per_cm)

        if KB_OK:
            ys=math.radians(1.5); ps=math.radians(1.5)
            if kb.is_pressed('j'): orbit_yaw   -= ys
            if kb.is_pressed('l'): orbit_yaw   += ys
            if kb.is_pressed('i'): orbit_pitch += ps
            if kb.is_pressed('k'): orbit_pitch -= ps
            if kb.is_pressed('['): orbit_radius += 12
            if kb.is_pressed(']'): orbit_radius  = max(80.,orbit_radius-12)
            if kb.is_pressed('r'): orbit_yaw=0.;orbit_pitch=0.;orbit_radius=600.
            orbit_pitch=max(math.radians(-89),min(math.radians(89),orbit_pitch))
            f7_now=kb.is_pressed('f7')
            if f7_now and not prev_f7:
                mouse_enabled=not mouse_enabled
                print(f"[Mouse] {'ON' if mouse_enabled else 'OFF'}")
            prev_f7=f7_now

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c') and head_center is not None and \
                not (left_sphere_locked and right_sphere_locked):
            cns = compute_scale(nose_pts_3d)
            cdl = R_final.T@np.array([0,0,1.])
            left_sphere_local_offset  = R_final.T@(iris_3d_left -head_center)+BASE_RADIUS*cdl
            right_sphere_local_offset = R_final.T@(iris_3d_right-head_center)+BASE_RADIUS*cdl
            left_cal_scale  = cns;  right_cal_scale  = cns
            left_sphere_locked = right_sphere_locked = True

            swl=head_center+R_final@left_sphere_local_offset
            swr=head_center+R_final@right_sphere_local_offset
            fwd=_normalize(_normalize(iris_3d_left-swl)+_normalize(iris_3d_right-swr))
            go =(swl+swr)*0.5

            monitor_corners,monitor_center_w,monitor_normal_w,units_per_cm = \
                create_monitor_plane(head_center,R_final,lms,fw,fh,
                                     forward_hint=fwd,gaze_origin=go,gaze_dir=fwd)
            debug_world_frozen = True
            orbit_pivot_frozen = monitor_center_w.copy()
            gaze_filter.reset(); fixation_det.reset()
            keyboard_gui.status = "Spheres locked. 1→TL, 2→TR, 3→BL, 4→BR + SPACE."
            print("[Cal] Spheres locked. Do corner calibration.")

        elif key in (ord('1'),ord('2'),ord('3'),ord('4')) and \
                left_sphere_locked and monitor_corners is not None:
            idx=key-ord('1')
            calib.step=idx; calib.active=True
            calib._all_samples.clear(); calib._stable_samples=[]
            print(f"[Cal] Look at {calib.LABELS[idx]}. Press SPACE.")
            keyboard_gui.status=f"Look at {calib.LABELS[idx]}, then SPACE."

        elif key==ord(' ') or (KB_OK and kb.is_pressed('space') and not prev_space):
            if calib.active:
                ok = calib.confirm_point()
                if ok and not calib.active:
                    keyboard_gui.status = "Calibration done! Type with your eyes."
                elif ok:
                    keyboard_gui.status = f"Look at {calib.current_label}, press SPACE."
                else:
                    keyboard_gui.status = "Need more stable frames — hold your gaze."

        elif key==ord('x') and monitor_corners is not None \
                and avg_combined is not None and sphere_world_l is not None:
            O3=(sphere_world_l+sphere_world_r)*0.5
            ab=ray_plane_ab(O3,avg_combined,monitor_corners,monitor_center_w,monitor_normal_w)
            if ab:
                a_,b_=ab; a_=1.0-a_
                gaze_markers.append((a_,b_))
                print(f"[Marker] a={a_:.3f}, b={b_:.3f}")

        if not (KB_OK and kb.is_pressed('space')):
            prev_space = False
        elif key == ord(' '):
            prev_space = True

    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()