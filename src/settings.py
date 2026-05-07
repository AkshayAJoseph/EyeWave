"""
settings.py
============
Settings overlay panel for EyeWave.

Draws a semi-transparent overlay on the keyboard window with adjustable
settings. Designed for caregiver operation using keyboard arrow keys.
"""

import cv2
import numpy as np

from src.config import (
    KBD_WIN_W, KBD_WIN_H,
)


class SettingsOverlay:
    """
    On-screen settings panel rendered over the keyboard.

    Navigation (keyboard arrow keys — for caregiver use):
        UP/DOWN     — select setting
        LEFT/RIGHT  — adjust value
        S           — close overlay
    """

    def __init__(self):
        self.is_open = False
        self._cursor = 0   # which setting row is selected

        # Setting definitions: (label, key, min, max, step, fmt)
        self._settings = [
            ("EAR Threshold",     "ear_threshold",    0.10,  0.40,  0.01,  "{:.2f}"),
            ("Scan Row Speed",    "scan_row_rate",    0.5,   4.0,   0.1,   "{:.1f}s"),
            ("Scan Col Speed",    "scan_col_rate",    0.5,   4.0,   0.1,   "{:.1f}s"),
            ("Col Timeout",       "scan_col_timeout", 3.0,   20.0,  1.0,   "{:.0f}s"),
            ("Audio",             "audio_enabled",    0,     1,     1,     "toggle"),
            ("Blink Min (ms)",    "blink_min_ms",     50,    300,   10,    "{:.0f}"),
            ("Blink Max (ms)",    "blink_max_ms",     300,   800,   25,    "{:.0f}"),
            ("Long Blink Max",    "blink_long_max_ms", 800,  2000,  50,    "{:.0f}"),
        ]

    def toggle(self):
        self.is_open = not self.is_open
        self._cursor = 0

    def handle_key(self, key: int, profile_mgr) -> bool:
        """
        Process a keypress while overlay is open.
        Returns True if the key was consumed.
        """
        if not self.is_open:
            return False

        p = profile_mgr.current

        if key == 0:  # UP arrow
            self._cursor = (self._cursor - 1) % len(self._settings)
            return True
        elif key == 1:  # DOWN arrow
            self._cursor = (self._cursor + 1) % len(self._settings)
            return True
        elif key == 2:  # LEFT arrow — decrease
            self._adjust(p, -1)
            return True
        elif key == 3:  # RIGHT arrow — increase
            self._adjust(p, +1)
            return True

        return False

    def _adjust(self, profile, direction: int):
        """Adjust the currently selected setting."""
        label, attr, vmin, vmax, step, fmt = self._settings[self._cursor]
        val = getattr(profile, attr)

        if fmt == "toggle":
            setattr(profile, attr, not val)
        else:
            new_val = val + direction * step
            new_val = max(vmin, min(vmax, new_val))
            # Round to avoid floating point drift
            if isinstance(step, float):
                new_val = round(new_val, 3)
            setattr(profile, attr, new_val)

    def draw(self, frame: np.ndarray, profile_mgr) -> np.ndarray:
        """Render the settings overlay on top of the keyboard frame."""
        if not self.is_open:
            return frame

        p = profile_mgr.current
        overlay = frame.copy()

        # Semi-transparent dark background
        h, w = overlay.shape[:2]
        panel_x = w // 6
        panel_y = 60
        panel_w = w - 2 * panel_x
        panel_h = 50 + len(self._settings) * 48 + 80

        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (15, 15, 25), -1)
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (80, 80, 120), 2)

        # Blend
        frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)

        # Title
        tx = panel_x + 20
        ty = panel_y + 35
        cv2.putText(frame, f"SETTINGS  -  Profile: {p.name}",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 255), 2, cv2.LINE_AA)

        # Settings rows
        for i, (label, attr, vmin, vmax, step, fmt) in enumerate(self._settings):
            y = ty + 50 + i * 48
            is_sel = (i == self._cursor)
            col = (100, 255, 200) if is_sel else (150, 150, 170)
            bg_col = (30, 50, 40) if is_sel else None

            if bg_col:
                cv2.rectangle(frame,
                              (panel_x + 10, y - 22),
                              (panel_x + panel_w - 10, y + 18),
                              bg_col, -1)

            # Label
            cv2.putText(frame, label, (tx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

            # Value
            val = getattr(p, attr)
            if fmt == "toggle":
                val_str = "ON" if val else "OFF"
                val_col = (0, 255, 120) if val else (100, 100, 100)
            else:
                val_str = fmt.format(val)
                val_col = col

            vx = panel_x + panel_w - 200
            cv2.putText(frame, f"< {val_str} >", (vx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, val_col, 1, cv2.LINE_AA)

            # Progress bar for numeric values
            if fmt != "toggle":
                bar_x = vx + 120
                bar_y = y - 8
                bar_w = 60
                bar_h = 10
                pct = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
                fill_w = int(bar_w * pct)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h), val_col, -1)

        # Footer
        fy = ty + 50 + len(self._settings) * 48 + 20
        cv2.putText(frame, "Arrow keys: navigate/adjust  |  S: close & save",
                    (tx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 100, 130), 1, cv2.LINE_AA)

        return frame
