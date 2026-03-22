"""
interface.py
============
LayoutManager  — manages QWERTY / AAC grid, switching, key lookup
EyeKeyboard    — all OpenCV rendering and typed-text state

No gaze logic lives here.  All selection events arrive via activate_key().
"""

import time

import cv2
import numpy as np

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False

try:
    import winsound          # Windows click sound
    WINSOUND_OK = True
except ImportError:
    WINSOUND_OK = False

import threading

from src.config import (
    QWERTY_GRID, AAC_GRID,
    AAC_PHRASE_ROWS, AAC_SPECIAL_ROW,
    PHRASES, AAC_VOCAB,
    KBD_WIN_W, KBD_WIN_H,
    GRID_X, GRID_Y, GRID_W, GRID_H,
    TEXT_Y, TEXT_H, SUGG_Y,
    CLICK_SOUND,
)
from src.visionc import (
    SmartDwellController,
    ScanningController,
    BlinkDetector,
    FixationDetector,
    AdaptiveGazeFilter,
    MultiPointCalib,
)
from src.utils import GazeDataCollector

import os


# ═══════════════════════════════════════════════════════════════════════════
#  LAYOUT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class LayoutManager:
    """
    Manages the active keyboard layout (QWERTY or AAC) and provides
    helpers for key lookup and row classification.
    """

    LAYOUT_QWERTY = 'qwerty'
    LAYOUT_AAC    = 'aac'

    def __init__(self):
        self.current = self.LAYOUT_QWERTY

    def toggle(self):
        self.current = (self.LAYOUT_AAC
                        if self.current == self.LAYOUT_QWERTY
                        else self.LAYOUT_QWERTY)

    @property
    def grid(self):
        return QWERTY_GRID if self.current == self.LAYOUT_QWERTY else AAC_GRID

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0])

    @property
    def is_aac(self) -> bool:
        return self.current == self.LAYOUT_AAC

    def key_at(self, row: int, col: int) -> str | None:
        g = self.grid
        if 0 <= row < len(g) and 0 <= col < len(g[row]):
            return g[row][col]
        return None

    def is_phrase_row(self, row: int) -> bool:
        return self.is_aac and row in AAC_PHRASE_ROWS

    def is_special_row(self, row: int) -> bool:
        return self.is_aac and row == AAC_SPECIAL_ROW


# ═══════════════════════════════════════════════════════════════════════════
#  EYE KEYBOARD  (rendering + text state)
# ═══════════════════════════════════════════════════════════════════════════

class EyeKeyboard:
    """
    Owns the typed text buffer, word suggestions, flash state, and all
    OpenCV rendering.  Selection logic (dwell / scan / blink) lives in
    the pipeline classes; this class only processes activate_key() calls.
    """

    def __init__(self):
        self.typed_text  = ""
        self.suggestions = []
        self.flash_key   = None
        self.flash_end   = 0.0
        self.status      = "Press C to calibrate (loads saved if available)."

    # ── Key activation ────────────────────────────────────────────────────

    def activate_key(self, kp: tuple, layout: LayoutManager):
        """
        Process one key activation.

        Returns
        -------
        '__SWAP__'  — caller should toggle layout
        key string  — the character(s) appended / action taken
        False       — cell out of range
        """
        r, c = kp
        key  = layout.key_at(r, c)
        if key is None:
            return False

        self.flash_key = kp
        self.flash_end = time.time() + 0.35
        self._play_click()

        # Internal routing
        if key == '__SWAP__' or key == 'SWAP':
            return '__SWAP__'

        # Control keys
        if   key == 'BP':    self.typed_text = self.typed_text[:-1]
        elif key == 'DL':    self.typed_text = ""
        elif key == 'PL':    self._speak(self.typed_text.strip())
        elif key in PHRASES:
            val = PHRASES[key]
            if val.startswith('__'):
                return val                      # pass internal commands up
            # Full phrase → replace text and speak; single char → append
            if len(val) > 2:
                self.typed_text = val
                self._speak(val)
            else:
                self.typed_text += val
        elif key == 'SPACE': self.typed_text += ' '
        elif key == 'NUM':   pass               # future: number pad
        else:                self.typed_text += key

        self._update_suggestions()
        return key

    def _update_suggestions(self):
        parts  = self.typed_text.split()
        prefix = parts[-1].lower() if parts else ""
        self.suggestions = ([w for w in AAC_VOCAB if w.startswith(prefix)][:5]
                            if prefix else [])

    def _speak(self, text: str):
        if not text or not TTS_OK:
            return
        def _do():
            try:
                e = pyttsx3.init()
                e.say(text)
                e.runAndWait()
            except Exception as ex:
                print(f"[TTS] {ex}")
        threading.Thread(target=_do, daemon=True).start()

    def _play_click(self):
        """Play a short click sound on key activation."""
        if os.path.exists(CLICK_SOUND):
            try:
                if WINSOUND_OK:
                    winsound.PlaySound(CLICK_SOUND,
                                       winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass

    # ── Master draw entry point ───────────────────────────────────────────

    def draw(self,
             layout:      LayoutManager,
             dwell:       SmartDwellController,
             scanner:     ScanningController,
             blinker:     BlinkDetector,
             fixation:    FixationDetector,
             gaze_filter: AdaptiveGazeFilter,
             calib:       MultiPointCalib,
             collector:   GazeDataCollector,
             sel_mode:    str) -> np.ndarray:
        """Render a complete keyboard frame and return it."""
        frame = np.zeros((KBD_WIN_H, KBD_WIN_W, 3), dtype=np.uint8)
        frame[:] = (8, 8, 12)

        self._draw_topbar(frame, layout, blinker, calib,
                          fixation, collector, sel_mode)
        self._draw_grid(frame, layout, dwell, scanner, sel_mode)
        self._draw_textbox(frame)
        self._draw_suggestions(frame)
        self._draw_cursors(frame, dwell, fixation, gaze_filter)
        return frame

    # ── Top status bar ────────────────────────────────────────────────────

    def _draw_topbar(self, frame, layout, blinker, calib,
                     fixation, collector, sel_mode):
        cv2.rectangle(frame, (0, 0), (KBD_WIN_W, GRID_Y - 2), (12, 12, 20), -1)

        if calib.active and calib.current_label:
            n   = calib.stable_count
            bar = min(n, calib.MIN_STABLE)
            msg = (f"  Look at {calib.current_label}  "
                   f"({bar}/{calib.MIN_STABLE} stable) → SPACE to confirm")
            cv2.putText(frame, msg, (6, GRID_Y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, self.status, (6, GRID_Y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (110, 110, 140), 1, cv2.LINE_AA)

        # Right-side badges
        badges = []
        if calib.ready:
            badges.append(("4pt✓", (60, 220, 80)))
        elif calib.active:
            n = sum(1 for p in calib.raw_pts if p is not None)
            badges.append((f"calib{n}/4", (0, 200, 255)))

        badges.append((
            "SCAN" if sel_mode == 'scan' else "GAZE",
            (0, 200, 255) if sel_mode == 'scan' else (200, 200, 0)
        ))
        badges.append((layout.current.upper(), (180, 120, 255)))
        if blinker.enabled:
            badges.append(("BLINK", (0, 255, 180)))
        badges.append((f"data:{collector.count}", (100, 100, 100)))

        bx = KBD_WIN_W - 12
        for txt, col in reversed(badges):
            tw, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            bx   -= tw + 14
            cv2.putText(frame, txt, (bx, GRID_Y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # Fixation dot
        fix_col = (0, 255, 120) if fixation.is_fixating else (60, 60, 60)
        cv2.circle(frame, (KBD_WIN_W - 8, GRID_Y - 12), 6, fix_col, -1)

    # ── Grid ──────────────────────────────────────────────────────────────

    def _key_dims(self, layout: LayoutManager, row: int, col: int):
        """Return (x1,y1,x2,y2) for a grid cell, with AAC variable row heights."""
        if layout.is_aac:
            ph  = len(AAC_PHRASE_ROWS)
            lr  = layout.rows - ph - 1         # letter rows
            total_u = ph * 2 + lr * 1 + 1 * 1.2
            unit_h  = GRID_H / total_u

            def row_u(r):
                return (2.0 if r in AAC_PHRASE_ROWS
                        else 1.2 if r == AAC_SPECIAL_ROW
                        else 1.0)

            y_off = sum(row_u(r) for r in range(row)) * unit_h
            y1 = GRID_Y + int(y_off)
            y2 = GRID_Y + int(y_off + row_u(row) * unit_h) - 2
        else:
            kh = GRID_H // layout.rows
            y1 = GRID_Y + row * kh
            y2 = y1 + kh - 2

        kw = GRID_W // layout.cols
        x1 = GRID_X + col * kw
        x2 = x1 + kw - 2
        return x1, y1, x2, y2

    def _draw_grid(self, frame, layout, dwell, scanner, sel_mode):
        now = time.time()
        dp  = dwell.dwell_progress
        hov = dwell.hovered if sel_mode == 'gaze' else None
        hov_col = hov[1] if hov else -1

        scan_row    = scanner.scan_row if scanner.state != scanner.ST_IDLE else -1
        scan_col    = scanner.scan_col if scanner.state == scanner.ST_COL  else -1
        in_row_scan = scanner.state == scanner.ST_ROW
        in_col_scan = scanner.state == scanner.ST_COL

        for r in range(layout.rows):
            for c in range(layout.cols):
                key = layout.key_at(r, c)
                if key is None:
                    continue
                x1, y1, x2, y2 = self._key_dims(layout, r, c)
                kw, kh = x2 - x1, y2 - y1

                is_hov    = (hov == (r, c))
                is_flash  = (self.flash_key == (r, c) and now < self.flash_end)
                is_col_h  = (c == hov_col and not is_hov)
                row_lit   = in_row_scan and r == scan_row
                col_lit   = in_col_scan and r == scan_row and c == scan_col

                # Background
                if   is_flash:  bg = (30, 200, 30)
                elif col_lit:   bg = (0, 200, 100)
                elif row_lit:   bg = (0, 80, 140)
                elif is_hov:
                    b_ = int(255 * (1 - dp))
                    g_ = int(180 * dp)
                    r_ = int(255 * dp)
                    bg = (b_, g_, r_)
                elif is_col_h:              bg = (45, 45, 80)
                elif layout.is_phrase_row(r): bg = (28, 12, 50)
                elif layout.is_special_row(r): bg = (12, 28, 28)
                else:                        bg = (18, 18, 24)

                cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)

                # Border
                border = ((0, 255, 150) if col_lit
                          else (0, 150, 255) if row_lit or is_hov
                          else (48, 48, 62))
                cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)

                # Label
                fc = (0, 0, 0) if is_flash else (220, 220, 220)
                fs = (0.60 if layout.is_phrase_row(r)
                      else 0.38 if len(key) > 4
                      else 0.44 if len(key) > 2
                      else 0.58)
                tw, th = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
                cv2.putText(frame, key,
                            (x1 + (kw - tw) // 2, y1 + (kh + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)

                # Dwell arc (gaze mode only)
                if is_hov and dp > 0.01 and sel_mode == 'gaze':
                    cx  = x1 + kw // 2;  cy = y1 + kh // 2
                    rad = min(kw, kh) // 2 - 3
                    cv2.ellipse(frame, (cx, cy), (rad, rad),
                                -90, 0, int(360 * dp),
                                (0, 255, 180), 2, cv2.LINE_AA)

        # Scan state label
        if scanner.state != scanner.ST_IDLE:
            lbl = (f"SCANNING ROW {scan_row+1}/{layout.rows}"
                   if in_row_scan
                   else f"SCANNING COL {scan_col+1}/{layout.cols} "
                        f"in ROW {scan_row+1}")
            cv2.putText(frame, lbl, (GRID_X, GRID_Y + GRID_H + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1, cv2.LINE_AA)

    # ── Text box ──────────────────────────────────────────────────────────

    def _draw_textbox(self, frame):
        bx1, by1 = GRID_X, TEXT_Y
        bx2, by2 = GRID_X + GRID_W, TEXT_Y + TEXT_H
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 28), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (65, 65, 88), 1)
        disp = (self.typed_text[-90:]
                if len(self.typed_text) > 90
                else self.typed_text)
        cv2.putText(frame, disp + "|", (bx1 + 10, by1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.76,
                    (150, 255, 150), 1, cv2.LINE_AA)

    # ── Suggestions ───────────────────────────────────────────────────────

    def _draw_suggestions(self, frame):
        if not self.suggestions:
            return
        cv2.putText(frame, "Predict:", (GRID_X, SUGG_Y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 100, 200), 1)
        for i, w in enumerate(self.suggestions):
            sx = GRID_X + 82 + i * 165
            sy = SUGG_Y
            cv2.rectangle(frame, (sx-4, sy), (sx+154, sy+22), (32, 32, 55), -1)
            cv2.rectangle(frame, (sx-4, sy), (sx+154, sy+22), (65, 65, 110), 1)
            cv2.putText(frame, w, (sx, sy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 192, 80), 1, cv2.LINE_AA)

    # ── Cursors ───────────────────────────────────────────────────────────

    def _draw_cursors(self, frame, dwell, fixation, gaze_filter):
        """
        Two cursors:
          1. Large crosshair  — smooth filter position (display)
          2. Small green dot  — fixation centroid (only during fixation)
             This is what actually drives dwell / scanning.
        """
        da = float(np.clip(gaze_filter.a, 0.0, 1.0))
        db = float(np.clip(gaze_filter.b, 0.0, 1.0))
        gx = int(GRID_X + da * GRID_W)
        gy = int(GRID_Y + db * GRID_H)
        r  = 13
        col = (0, 195, 255) if fixation.is_fixating else (0, 90, 160)

        cv2.circle(frame, (gx, gy), r, col, 1, cv2.LINE_AA)
        cv2.circle(frame, (gx, gy), 2, col, -1)
        cv2.line(frame, (gx-r-6, gy), (gx-r+2, gy), col, 1)
        cv2.line(frame, (gx+r-2, gy), (gx+r+6, gy), col, 1)
        cv2.line(frame, (gx, gy-r-6), (gx, gy-r+2), col, 1)
        cv2.line(frame, (gx, gy+r-2), (gx, gy+r+6), col, 1)

        if fixation.is_fixating:
            fa = float(np.clip(fixation.centroid_a, 0.0, 1.0))
            fb = float(np.clip(fixation.centroid_b, 0.0, 1.0))
            fx = int(GRID_X + fa * GRID_W)
            fy = int(GRID_Y + fb * GRID_H)
            cv2.circle(frame, (fx, fy), 5, (0, 255, 180), -1, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 5, (255, 255, 255),  1, cv2.LINE_AA)
