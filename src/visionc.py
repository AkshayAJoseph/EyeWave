"""
visionc.py
==========
Computer-vision pipeline classes and the 3D debug orbit view.

Classes
-------
AdaptiveGazeFilter    — velocity-aware 2-D EMA in (a,b) space
FixationDetector      — I-DT dispersion-threshold fixation identification
SmartDwellController  — fixation-gated dwell with key hysteresis
MultiPointCalib       — 4-corner homography calibration
BlinkDetector         — EAR-based intentional blink / double-blink detection
ScanningController    — row-column scanning with gaze-assisted jump

Functions
---------
render_debug_view_orbit — 3D orbit camera view of head, eyes, and monitor plane
"""

import math
import time
import collections

import cv2
import numpy as np
from collections import deque

from src.config import (
    # Filter
    FILTER_ALPHA_SACCADE, FILTER_ALPHA_FIXATION,
    FILTER_SACCADE_THRESH, FILTER_FIXATION_THRESH,
    # Fixation
    FIXATION_WINDOW, FIXATION_DISP_MAX, FIXATION_SPEED_MAX, FIXATION_MIN_SAMPLES,
    # Dwell
    DWELL_TIME, DWELL_COOLDOWN, DWELL_CONFIRM_FRAMES,
    # Blink
    BLINK_EAR_THRESH, BLINK_MIN_MS, BLINK_MAX_MS, BLINK_LONG_MAX_MS,
    BLINK_DOUBLE_GAP_MS,
    # Scanner
    SCAN_ROW_RATE, SCAN_COL_RATE, SCAN_COL_TIMEOUT,
    SCAN_SPEED_MIN, SCAN_SPEED_MAX,
    # Audio
    AUDIO_ENABLED, AUDIO_ROW_TICK, AUDIO_COL_TICK,
    AUDIO_ROW_SELECT, AUDIO_CANCEL, AUDIO_KEY_ACTIVATE, AUDIO_UNDO,
    # Calibration
    CALIB_STABLE_DISP_MAX, CALIB_STABLE_WINDOW, CALIB_MIN_STABLE,
    # Orbit defaults
    ORBIT_FOV,
    # Landmark indices
    LEFT_EYE_EAR, RIGHT_EYE_EAR,
)
from src.utils import normalize, focal_px, rot_x, rot_y, ray_plane_ab

import threading

try:
    import winsound
    _WINSOUND_OK = True
except ImportError:
    _WINSOUND_OK = False


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════

class AudioFeedback:
    """Non-blocking audio cues for scanning events using winsound.Beep."""

    def __init__(self):
        self.enabled = AUDIO_ENABLED

    def _beep(self, freq: int, dur: int):
        if self.enabled and _WINSOUND_OK:
            threading.Thread(target=winsound.Beep, args=(freq, dur),
                             daemon=True).start()

    def _multi_beep(self, tones: list):
        if self.enabled and _WINSOUND_OK:
            def _play():
                for freq, dur in tones:
                    winsound.Beep(freq, dur)
            threading.Thread(target=_play, daemon=True).start()

    def tick_row(self):
        self._beep(*AUDIO_ROW_TICK)

    def tick_col(self):
        self._beep(*AUDIO_COL_TICK)

    def row_selected(self):
        self._multi_beep(AUDIO_ROW_SELECT)

    def cancel(self):
        self._multi_beep(AUDIO_CANCEL)

    def key_activated(self):
        self._beep(*AUDIO_KEY_ACTIVATE)

    def undo(self):
        self._multi_beep(AUDIO_UNDO)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTIVE GAZE FILTER
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveGazeFilter:
    """
    Velocity-aware exponential moving average in (a,b) space.

    Why filter in 2-D after the ray-plane, not in 3-D before it?
    Filtering 3-D direction vectors then projecting magnifies angular
    noise by the projection distance.  Filtering the already-projected
    (a,b) coordinates keeps noise in the plane's own units.

    Two modes
    ---------
    SACCADE  (speed > SACCADE_THRESH)  → heavy smooth (alpha=0.08)
    FIXATION (speed < FIXATION_THRESH) → light  smooth (alpha=0.55)
    Linear interpolation between the thresholds.
    """

    def __init__(self):
        self.a = 0.5
        self.b = 0.5
        self._buf   = deque(maxlen=6)
        self.speed  = 0.0

    def update(self, a_raw: float, b_raw: float) -> tuple:
        """
        Returns (a_filtered, b_filtered, speed).
        Call once per frame with raw gaze coordinates.
        """
        now = time.time()
        self._buf.append((a_raw, b_raw, now))

        if len(self._buf) >= 3:
            old = self._buf[-3]
            dt  = now - old[2]
            if dt > 5e-4:
                self.speed = math.hypot(a_raw - old[0], b_raw - old[1]) / dt

        s = self.speed
        if   s >= FILTER_SACCADE_THRESH:  alpha = FILTER_ALPHA_SACCADE
        elif s <= FILTER_FIXATION_THRESH: alpha = FILTER_ALPHA_FIXATION
        else:
            t = ((s - FILTER_FIXATION_THRESH)
                 / (FILTER_SACCADE_THRESH - FILTER_FIXATION_THRESH))
            alpha = FILTER_ALPHA_FIXATION * (1 - t) + FILTER_ALPHA_SACCADE * t

        self.a = alpha * a_raw + (1 - alpha) * self.a
        self.b = alpha * b_raw + (1 - alpha) * self.b
        return self.a, self.b, self.speed

    def reset(self):
        self.a = self.b = 0.5
        self._buf.clear()
        self.speed = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  FIXATION DETECTOR  (I-DT)
# ═══════════════════════════════════════════════════════════════════════════

class FixationDetector:
    """
    Dispersion-threshold fixation identification (Salvucci & Goldberg 2000).

    A fixation is declared when:
      - Spatial dispersion of the last N filtered samples < DISP_MAX
      - Current speed < SPEED_MAX

    centroid_a / centroid_b : mean position during fixation — more stable
    than the instantaneous filtered value and used to drive dwell.
    """

    def __init__(self):
        self._buf = deque(maxlen=FIXATION_WINDOW)
        self.is_fixating = False
        self.centroid_a  = 0.5
        self.centroid_b  = 0.5
        self.dispersion  = 1.0

    def update(self, a: float, b: float, speed: float) -> bool:
        self._buf.append((a, b))
        if len(self._buf) < FIXATION_MIN_SAMPLES:
            self.is_fixating = False
            return False

        arr = np.array(self._buf)
        self.dispersion = math.hypot(
            float(arr[:, 0].max() - arr[:, 0].min()),
            float(arr[:, 1].max() - arr[:, 1].min()),
        )
        self.is_fixating = (self.dispersion < FIXATION_DISP_MAX
                            and speed < FIXATION_SPEED_MAX)
        if self.is_fixating:
            self.centroid_a = float(arr[:, 0].mean())
            self.centroid_b = float(arr[:, 1].mean())
        return self.is_fixating

    def reset(self):
        self._buf.clear()
        self.is_fixating = False
        self.dispersion  = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  SMART DWELL CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class SmartDwellController:
    """
    Fixation-gated dwell with key hysteresis and soft-decay.

    Improvements over naive dwell
    ------------------------------
    1. Only accumulates while FixationDetector says fixating.
    2. Key hysteresis: N consecutive fixating frames required before
       hover switches key — eliminates boundary flicker.
    3. Soft decay during saccades: brief noise / blinks don't erase progress.
    4. Per-key cooldown prevents accidental double-fire.
    """

    def __init__(self):
        self.hovered         = None
        self._candidate      = None
        self._candidate_cnt  = 0
        self._dwell_accum    = 0.0
        self._last_act_key   = None
        self._last_act_time  = 0.0
        self.dwell_progress  = 0.0
        self.activated_key   = None

    def update(self, centroid_a: float, centroid_b: float,
               is_fixating: bool, dt: float,
               rows: int, cols: int):
        """
        Call once per frame.

        Returns activated (row, col) or None.
        dwell_progress (0-1) is updated for the progress arc.
        """
        self.activated_key = None
        now = time.time()

        col = min(int(centroid_a * cols), cols - 1)
        row = min(int(centroid_b * rows), rows - 1)
        kp  = (row, col)
        on_kbd = (0.0 <= centroid_a <= 1.0 and 0.0 <= centroid_b <= 1.0)

        # Key hysteresis
        if on_kbd and is_fixating:
            if kp == self._candidate:
                self._candidate_cnt += 1
            else:
                self._candidate     = kp
                self._candidate_cnt = 1
            if self._candidate_cnt >= DWELL_CONFIRM_FRAMES:
                if kp != self.hovered:
                    self.hovered      = kp
                    self._dwell_accum = 0.0

        # Accumulation / soft decay
        if self.hovered and is_fixating and on_kbd:
            self._dwell_accum += dt
        else:
            self._dwell_accum = max(0.0, self._dwell_accum - dt * 1.8)

        self.dwell_progress = (min(self._dwell_accum / DWELL_TIME, 1.0)
                               if DWELL_TIME > 0 else 0.0)

        # Activation
        if self._dwell_accum >= DWELL_TIME and self.hovered:
            ok = not (self._last_act_key == self.hovered
                      and now - self._last_act_time < DWELL_COOLDOWN)
            if ok:
                self.activated_key  = self.hovered
                self._last_act_key  = self.hovered
                self._last_act_time = now
                self._dwell_accum   = 0.0

        return self.activated_key


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-POINT CALIBRATION  (4-corner homography)
# ═══════════════════════════════════════════════════════════════════════════

class MultiPointCalib:
    """
    4-corner homography calibration.

    Only stable frames (dispersion < threshold) are kept.
    Confirmation uses the median of those frames — robust to outliers.
    The resulting homography maps raw (a,b) → corrected (a,b).
    """

    TARGETS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    LABELS  = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_pts  = [None] * 4
        self.active   = False
        self.step     = 0
        self._all     = deque(maxlen=80)
        self._stable  = []
        self._done    = False
        self._H       = None

    def load_homography(self, H: np.ndarray):
        """Restore a previously saved homography matrix."""
        self._H    = H
        self._done = True

    def record(self, a: float, b: float, dispersion: float):
        if not self.active:
            return
        self._all.append((a, b))
        if dispersion < CALIB_STABLE_DISP_MAX:
            self._stable.append((a, b))
            if len(self._stable) > CALIB_STABLE_WINDOW:
                self._stable.pop(0)

    def confirm_point(self) -> bool:
        if not self.active:
            return False
        samples = (self._stable if len(self._stable) >= CALIB_MIN_STABLE
                   else list(self._all))
        if not samples:
            print("[MultiPointCalib] No samples -- keep looking at the corner.")
            return False

        arr = np.array(samples)
        ma  = float(np.median(arr[:, 0]))
        mb  = float(np.median(arr[:, 1]))
        self.raw_pts[self.step] = (ma, mb)
        print(f"[MultiPointCalib] {self.LABELS[self.step]} "
              f"raw=({ma:.4f},{mb:.4f})  "
              f"({len(self._stable)} stable frames)")

        self._all.clear()
        self._stable = []
        self.step   += 1

        if self.step >= 4:
            self.active = False
            self._done  = True
            self._build_H()
        else:
            print(f"[MultiPointCalib] Now look at {self.LABELS[self.step]}. "
                  f"Press SPACE.")
        return True

    def _build_H(self):
        src = np.array(self.raw_pts, dtype=np.float32)
        dst = np.array(self.TARGETS, dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst)
        if self._H is not None:
            print("[MultiPointCalib] Homography built -- correction active.")
        else:
            print("[MultiPointCalib] Homography failed -- redo corner calibration.")

    def correct(self, a: float, b: float) -> tuple:
        if self._H is None:
            return a, b
        pt  = np.array([[[a, b]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return (float(np.clip(out[0, 0, 0], -0.1, 1.1)),
                float(np.clip(out[0, 0, 1], -0.1, 1.1)))

    @property
    def ready(self) -> bool:
        return self._done and self._H is not None

    @property
    def current_label(self) -> str | None:
        return self.LABELS[self.step] if self.active and self.step < 4 else None

    @property
    def stable_count(self) -> int:
        return len(self._stable)


# ═══════════════════════════════════════════════════════════════════════════
#  BLINK DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class BlinkDetector:
    """
    Eye Aspect Ratio (EAR) based intentional blink and double-blink detection.

    EAR = (||p1-p5|| + ||p2-p4||) / (2 · ||p0-p3||)

    Involuntary blinks (< MIN_MS)  → ignored
    Intentional blinks (MIN-MAX ms) → fires  self.blink = True  for one frame
    Double blink (two blinks within DOUBLE_GAP_MS) → self.double_blink = True
    Long hold (> MAX_MS)           → ignored (fatigue / looking away)
    """

    def __init__(self):
        self.enabled      = True
        self.blink        = False
        self.double_blink = False
        self.long_blink   = False     # long-blink undo gesture
        self.ear          = 0.3      # expose current EAR for debug display
        self.debug_ear    = False    # set True to print EAR values
        self._closed_t    = None    # time (ms) the eye went below threshold
        self._last_blink  = 0.0    # time (ms) of last confirmed blink

    @staticmethod
    def _ear(lms, idx, w: int, h: int) -> float:
        p  = [np.array([lms[i].x * w, lms[i].y * h]) for i in idx]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        ho = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * ho) if ho > 1e-6 else 0.4

    def update(self, lms, w: int, h: int):
        """Call once per frame with MediaPipe face landmarks."""
        self.blink        = False
        self.double_blink = False
        self.long_blink   = False
        if not self.enabled or lms is None:
            return

        ear_l = self._ear(lms, LEFT_EYE_EAR,  w, h)
        ear_r = self._ear(lms, RIGHT_EYE_EAR, w, h)
        ear   = (ear_l + ear_r) / 2.0
        self.ear = ear
        now   = time.time() * 1000   # work in milliseconds

        if self.debug_ear:
            state = "CLOSED" if ear < BLINK_EAR_THRESH else "open"
            print(f"[EAR] L={ear_l:.3f} R={ear_r:.3f} avg={ear:.3f} [{state}]")

        if ear < BLINK_EAR_THRESH:
            if self._closed_t is None:
                self._closed_t = now
        else:
            if self._closed_t is not None:
                dur = now - self._closed_t
                self._closed_t = None
                if BLINK_MIN_MS <= dur <= BLINK_MAX_MS:
                    # Normal intentional blink
                    gap = now - self._last_blink
                    if gap <= BLINK_DOUBLE_GAP_MS:
                        self.double_blink = True
                        if self.debug_ear:
                            print(f"[BLINK] DOUBLE blink ({dur:.0f}ms)")
                    else:
                        self.blink = True
                        if self.debug_ear:
                            print(f"[BLINK] Single blink ({dur:.0f}ms)")
                    self._last_blink = now
                elif BLINK_MAX_MS < dur <= BLINK_LONG_MAX_MS:
                    # Long blink — undo gesture
                    self.long_blink = True
                    if self.debug_ear:
                        print(f"[BLINK] LONG blink / undo ({dur:.0f}ms)")
                elif self.debug_ear:
                    reason = "too short" if dur < BLINK_MIN_MS else "too long"
                    print(f"[BLINK] Rejected ({dur:.0f}ms, {reason})")


# ═══════════════════════════════════════════════════════════════════════════
#  SCANNING CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class ScanningController:
    """
    Row-column scanning with gaze-assisted row/column jump.

    Modes
    -----
    GAZE+SCAN (default) — gaze instantly jumps scanner to the looked-at row/col;
                          blink confirms.  Falls back to auto-advance when gaze
                          is not calibrated or fixation is not detected.
    PURE SCAN           — scanner advances automatically; blink selects.

    State machine
    -------------
    IDLE ──blink──► ROW_SCAN ──blink──► COL_SCAN ──blink──► activate → IDLE
    COL_SCAN ──double_blink──► ROW_SCAN  (go back to re-select row)
    ROW_SCAN ──double_blink──► IDLE
    COL_SCAN ──timeout──► ROW_SCAN  (auto-return after SCAN_COL_TIMEOUT)

    Features
    --------
    - Audio feedback on all state transitions and auto-advance
    - Adaptive scan speed based on rolling average response times
    - Long-blink undo (handled by caller, not in this class)
    """

    ST_IDLE = 'idle'
    ST_ROW  = 'row'
    ST_COL  = 'col'

    def __init__(self):
        self.enabled       = True
        self.gaze_assisted = True          # False = pure scan
        self.state         = self.ST_IDLE
        self.scan_row      = 0
        self.scan_col      = 0
        self._step_t       = time.time()
        self._col_enter    = time.time()   # when column scanning started
        self._n_rows       = 6
        self._n_cols       = 10
        self.activated_key = None

        # Audio feedback
        self.audio = AudioFeedback()

        # Adaptive scan speed
        self.adaptive        = True
        self._row_rate       = SCAN_ROW_RATE
        self._col_rate       = SCAN_COL_RATE
        self._resp_row       = collections.deque(maxlen=10)
        self._resp_col       = collections.deque(maxlen=10)

    @property
    def row_rate(self) -> float:
        return self._row_rate

    @property
    def col_rate(self) -> float:
        return self._col_rate

    def set_layout_size(self, rows: int, cols: int):
        self._n_rows = rows
        self._n_cols = cols
        self.scan_row = min(self.scan_row, rows - 1)
        self.scan_col = 0

    def start(self):
        self.state    = self.ST_ROW
        self.scan_row = 0
        self.scan_col = 0
        self._step_t  = time.time()

    def stop(self):
        self.state = self.ST_IDLE

    def _adapt_speed(self, deque, base_rate):
        """Compute adapted rate from rolling response times."""
        if not self.adaptive or len(deque) < 3:
            return base_rate
        avg = sum(deque) / len(deque)
        # Give 20% headroom — user should have time to react
        adapted = avg * 0.8
        return max(SCAN_SPEED_MIN, min(SCAN_SPEED_MAX, adapted))

    def _update_adaptive(self):
        """Recalculate adaptive rates from collected response data."""
        self._row_rate = self._adapt_speed(self._resp_row, SCAN_ROW_RATE)
        self._col_rate = self._adapt_speed(self._resp_col, SCAN_COL_RATE)

    def update(self, blink: bool, double_blink: bool,
               gaze_row: int | None, gaze_col: int | None):
        """
        Call once per frame.

        Parameters
        ----------
        blink / double_blink : from BlinkDetector
        gaze_row / gaze_col  : from FixationDetector centroid (None = no fix)

        Returns activated (row, col) or None.
        """
        self.activated_key = None
        now = time.time()

        if not self.enabled:
            return None

        # Double blink — context-sensitive cancel
        #   COL mode → go back to ROW (re-select correct row)
        #   ROW mode → go to IDLE (stop scanning)
        if double_blink:
            if self.state == self.ST_COL:
                self.state    = self.ST_ROW
                self.scan_col = 0
                self._step_t  = now
                self.audio.cancel()
                print("[Scanner] Double-blink: back to ROW scanning")
            else:
                self.state = self.ST_IDLE
                self.audio.cancel()
            return None

        if self.state == self.ST_IDLE:
            if blink:
                self.start()
                self.audio.tick_row()
            return None

        elif self.state == self.ST_ROW:
            # Gaze jump to looked-at row
            if self.gaze_assisted and gaze_row is not None:
                if gaze_row != self.scan_row:
                    self.scan_row = gaze_row
                    self._step_t  = now

            if blink:
                # Record response time for adaptive speed
                resp = now - self._step_t
                self._resp_row.append(resp)
                self._update_adaptive()

                self.state       = self.ST_COL
                self.scan_col    = 0
                self._col_enter  = now
                if self.gaze_assisted and gaze_col is not None:
                    self.scan_col = gaze_col
                self._step_t  = now
                self.audio.row_selected()
                return None

            # Auto-advance
            if now - self._step_t >= self._row_rate:
                self.scan_row = (self.scan_row + 1) % self._n_rows
                self._step_t  = now
                self.audio.tick_row()

        elif self.state == self.ST_COL:
            # Timeout: auto-return to row scanning if no selection
            if now - self._col_enter >= SCAN_COL_TIMEOUT:
                self.state    = self.ST_ROW
                self.scan_col = 0
                self._step_t  = now
                self.audio.cancel()
                print("[Scanner] Column timeout: back to ROW scanning")
                return None

            # Gaze jump to looked-at column
            if self.gaze_assisted and gaze_col is not None:
                if gaze_col != self.scan_col:
                    self.scan_col = gaze_col
                    self._step_t  = now

            if blink:
                # Record response time for adaptive speed
                resp = now - self._step_t
                self._resp_col.append(resp)
                self._update_adaptive()

                self.activated_key = (self.scan_row, self.scan_col)
                self.state         = self.ST_IDLE
                self.audio.key_activated()
                return self.activated_key

            # Auto-advance
            if now - self._step_t >= self._col_rate:
                self.scan_col = (self.scan_col + 1) % self._n_cols
                self._step_t  = now
                self.audio.tick_col()

        return self.activated_key


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG ORBIT VIEW  (3-D visualisation of head, eyes, gaze, monitor plane)
# ═══════════════════════════════════════════════════════════════════════════

def render_debug_view_orbit(
    dh: int, dw: int,
    orbit_yaw: float, orbit_pitch: float,
    orbit_radius: float,
    debug_world_frozen: bool, orbit_pivot_frozen,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked: bool = False, right_locked: bool = False,
    landmarks3d=None, combined_dir=None,
    gaze_len: float = 4300,
    monitor_corners=None,
    monitor_center=None, monitor_normal=None,
    gaze_markers=None, units_per_cm=None,
):
    """
    Renders the 3-D orbit debug window showing:
      - Head centre (magenta cross)
      - Eye spheres + per-eye gaze rays
      - Combined gaze ray
      - Monitor plane quad + normal arrow
      - Gaze hit circle on the monitor plane
      - Saved gaze markers (green dots)
    """
    if head_center3d is None:
        return

    debug  = np.zeros((dh, dw, 3), dtype=np.uint8)
    head_w = np.asarray(head_center3d, dtype=float)

    # ── Camera pivot ──────────────────────────────────────────────────────
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    elif monitor_center is not None:
        pivot_w = (head_w + np.asarray(monitor_center)) * 0.5
    else:
        pivot_w = head_w

    fpx    = focal_px(dw, ORBIT_FOV)
    cam_pos = pivot_w + rot_y(orbit_yaw) @ (rot_x(orbit_pitch)
                                            @ np.array([0., 0., orbit_radius]))
    fwd    = normalize(pivot_w - cam_pos)
    right  = normalize(np.cross(fwd, np.array([0., -1., 0.])))
    up     = normalize(np.cross(right, fwd))
    V      = np.stack([right, up, fwd], axis=0)

    def proj(P):
        Pc = V @ (np.asarray(P, dtype=float) - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = fpx * (Pc[0] / Pc[2]) + dw * 0.5
        y = -fpx * (Pc[1] / Pc[2]) + dh * 0.5
        return ((int(x), int(y)), Pc[2]) if np.isfinite(x) and np.isfinite(y) else None

    def dcross(P, sz=12, col=(255, 0, 255), th=2):
        res = proj(P)
        if not res:
            return
        x, y = res[0]
        cv2.line(debug, (x-sz, y), (x+sz, y), col, th)
        cv2.line(debug, (x, y-sz), (x, y+sz), col, th)

    def darrow(P0, P1, col=(0, 200, 255), th=2):
        a_ = proj(P0); b_ = proj(P1)
        if not a_ or not b_:
            return
        p0_, p1_ = a_[0], b_[0]
        cv2.line(debug, p0_, p1_, col, th)
        v = np.array([p1_[0]-p0_[0], p1_[1]-p0_[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n; lv = np.array([-v[1], v[0]]); ah = 9
            cv2.line(debug, p1_,
                     (int(p1_[0]-v[0]*ah+lv[0]*ah*.6),
                      int(p1_[1]-v[1]*ah+lv[1]*ah*.6)), col, th)
            cv2.line(debug, p1_,
                     (int(p1_[0]-v[0]*ah-lv[0]*ah*.6),
                      int(p1_[1]-v[1]*ah-lv[1]*ah*.6)), col, th)

    # Landmarks
    if landmarks3d is not None:
        for P in landmarks3d:
            res = proj(P)
            if res:
                cv2.circle(debug, res[0], 1, (180, 180, 180), -1)

    # Head centre
    dcross(head_w, sz=12, col=(255, 0, 255))
    hc2 = proj(head_w)
    if hc2:
        cv2.putText(debug, "Head", (hc2[0][0]+8, hc2[0][1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 255), 1)

    # Eyes
    for locked, sw, sr, iris3d, sc in [
        (left_locked,  sphere_world_l, scaled_radius_l, iris3d_l, (255, 255, 25)),
        (right_locked, sphere_world_r, scaled_radius_r, iris3d_r, (25, 255, 255)),
    ]:
        if locked and sw is not None:
            res = proj(sw)
            if res:
                (cx, cy), z = res
                rp = max(2, int((sr or 6) * fpx / max(z, 1e-3)))
                cv2.circle(debug, (cx, cy), rp, sc, 1)
                if iris3d is not None:
                    ld = np.asarray(iris3d) - np.asarray(sw)
                    p1_ = proj(np.asarray(sw) + normalize(ld) * gaze_len)
                    if p1_:
                        cv2.line(debug, (cx, cy), p1_[0],
                                 tuple(v//2 for v in sc), 1)

    # Combined gaze ray
    if (left_locked and right_locked
            and sphere_world_l is not None and sphere_world_r is not None
            and combined_dir is not None):
        om  = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) * 0.5
        p0_ = proj(om)
        p1_ = proj(om + normalize(combined_dir) * gaze_len * 1.2)
        if p0_ and p1_:
            cv2.line(debug, p0_[0], p1_[0], (155, 200, 10), 2)

    # Monitor plane
    if monitor_corners is not None:
        def dpoly(pts, col, th):
            pp = [proj(p) for p in pts]
            if any(x is None for x in pp):
                return
            p2 = [p[0] for p in pp]
            for a_, b_ in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a_, b_, col, th)
        dpoly(monitor_corners, (0, 200, 255), 2)
        dpoly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        dpoly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            dcross(monitor_center, sz=8, col=(0, 200, 255))
            if monitor_normal is not None:
                tip = (np.asarray(monitor_center)
                       + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0)))
                darrow(monitor_center, tip, col=(0, 220, 255))

    # Gaze hit circle
    if (monitor_corners and monitor_center is not None
            and monitor_normal is not None and combined_dir is not None
            and sphere_world_l is not None and sphere_world_r is not None):
        O_ = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) * 0.5
        ab = ray_plane_ab(O_, normalize(combined_dir),
                          monitor_corners, monitor_center, monitor_normal)
        if ab:
            a_, b_ = ab
            p0c, p1c, _, p3c = [np.asarray(p, dtype=float)
                                 for p in monitor_corners]
            P_  = p0c + a_ * (p1c - p0c) + b_ * (p3c - p0c)
            uh  = normalize(p1c - p0c)
            rw  = 0.05 * np.linalg.norm(p1c - p0c)
            pp  = proj(P_); pr = proj(P_ + uh * rw)
            if pp and pr:
                rp_ = int(max(1, np.linalg.norm(
                    np.array(pr[0]) - np.array(pp[0]))))
                cv2.circle(debug, pp[0], rp_, (0, 255, 255), 2, cv2.LINE_AA)

    # Gaze markers
    if gaze_markers and monitor_corners is not None:
        p0c, p1c, _, p3c = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1c - p0c; v = p3c - p0c
        ww = np.linalg.norm(u); uh = u / (ww + 1e-9)
        for (a_, b_) in gaze_markers:
            Pm = p0c + a_ * u + b_ * v
            pp = proj(Pm); pr = proj(Pm + uh * 0.01 * ww)
            if pp and pr:
                rp_ = int(max(1, np.linalg.norm(
                    np.array(pr[0]) - np.array(pp[0]))))
                cv2.circle(debug, pp[0], rp_, (0, 255, 0), 1, cv2.LINE_AA)

    # Help overlay
    help_lines = [
        "C=calib  TAB=layout  M=mode  B=blink",
        "1-4=corners  SPACE=confirm",
        "J/L=yaw  I/K=pitch  [/]=zoom  R=reset",
        "X=marker  F7=mouse  Q=quit",
    ]
    for i, t in enumerate(help_lines):
        cv2.putText(debug, t,
                    (8, dh - 10 - (len(help_lines) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("Head/Eye Debug", debug)
