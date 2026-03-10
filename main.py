"""EyeWave — Vision-Based Assistive Communication Interface.

Entry point:  python main.py
"""

import sys
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from src.visionc import EyeTracker
from src.interface import VirtualKeyboard, CalibrationOverlay
from src.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    TIMER_INTERVAL_MS, FPS_TARGET,
    BLINK_MIN_FRAMES, BLINK_MAX_FRAMES,
)
from src.utils import OneEuroFilter


class EyeWaveApp:
    """Main application — ties the eye tracker, calibration, and keyboard together."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tracker = EyeTracker()
        self.gui = VirtualKeyboard()

        # Camera
        print("Initializing camera…")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"Camera {CAMERA_INDEX} unavailable, trying index 1…")
            self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("FATAL: no camera found!")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        print(f"Camera opened: "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        # One Euro Filters for gaze stabilization
        # min_cutoff: lower = smoother during fixation (0.5-2.0 typical)
        # beta: higher = more responsive to fast movement (0.001-0.01)
        self.filter_x = OneEuroFilter(freq=FPS_TARGET, min_cutoff=0.8, beta=0.005)
        self.filter_y = OneEuroFilter(freq=FPS_TARGET, min_cutoff=0.8, beta=0.005)
        self.smooth_x = 0.5
        self.smooth_y = 0.5

        # Blink counter
        self.blink_frames = 0

        # Calibration overlay (created on demand)
        self.cal_overlay = None
        self.is_calibrating = False

        # Wire up calibration button
        self.gui.cal_btn.clicked.connect(self.start_calibration)

        # Main loop timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(TIMER_INTERVAL_MS)

    # ── Calibration ─────────────────────────────────────────────────────

    def start_calibration(self):
        """Open the calibration overlay (non-blocking so timer keeps running)."""
        # Re-lock eye spheres with the current frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            locked = self.tracker.lock_spheres_now(frame)
            if locked:
                print("Eye spheres re-locked for calibration")
            self.filter_x.reset()
            self.filter_y.reset()

        self.cal_overlay = CalibrationOverlay(self.gui)
        self.is_calibrating = True
        self.cal_overlay.finished.connect(self._on_calibration_done)
        self.cal_overlay.show()
        print("Calibration overlay opened")

    def _on_calibration_done(self, result):
        """Called when the calibration dialog closes."""
        if result == 1:  # QDialog.Accepted
            features, targets = self.cal_overlay.get_calibration_data()
            if features and targets:
                self.tracker.calibrate(features, targets)
                self.filter_x.reset()
                self.filter_y.reset()
                self.gui.status_label.setText("Calibration complete -- start typing!")
            else:
                self.gui.status_label.setText("Not enough data -- please recalibrate")
        else:
            self.gui.status_label.setText("Calibration cancelled")

        self.is_calibrating = False
        self.cal_overlay = None

    # ── Main Loop ───────────────────────────────────────────────────────

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # mirror

        try:
            success, is_blink, gx, gy, features = self.tracker.process_frame(frame)
        except Exception as e:
            print(f"Frame error: {e}")
            return

        if success:
            # Smooth gaze with One Euro Filter
            import time as _time
            t = _time.monotonic()
            self.smooth_x = self.filter_x.filter(gx, t)
            self.smooth_y = self.filter_y.filter(gy, t)

            # Feed calibration overlay (if active)
            if self.is_calibrating and self.cal_overlay:
                self.cal_overlay.add_features(features)

            # Update keyboard gaze
            self.gui.update_gaze(self.smooth_x, self.smooth_y)

            # Blink selection
            if is_blink:
                self.blink_frames += 1
            else:
                if BLINK_MIN_FRAMES < self.blink_frames < BLINK_MAX_FRAMES:
                    if self.gui.current_focused_key is not None:
                        idx, is_pred = self.gui.current_focused_key
                        if is_pred:
                            self.gui.select_prediction(idx)
                        else:
                            self.gui.select_key(idx)
                        self.gui.reset_dwell()
                self.blink_frames = 0

            # Debug window
            self._draw_debug(frame, gx, gy, is_blink)
        else:
            cv2.putText(frame, "NO FACE — position yourself in frame",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("EyeWave — Camera (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cleanup()
            sys.exit()

    # ── Debug Visualisation ─────────────────────────────────────────────

    def _draw_debug(self, frame, gx, gy, blink):
        h, w = frame.shape[:2]
        sx, sy = int(self.smooth_x * w), int(self.smooth_y * h)
        color = (0, 0, 255) if blink else (0, 255, 0)

        cv2.circle(frame, (sx, sy), 12, color, 3)
        cv2.line(frame, (sx - 20, sy), (sx + 20, sy), color, 2)
        cv2.line(frame, (sx, sy - 20), (sx, sy + 20), color, 2)

        label = "BLINK!" if blink else "TRACKING"
        cal = "CALIBRATED" if self.tracker.is_calibrated else "UNCALIBRATED"

        cv2.rectangle(frame, (5, 5), (300, 90), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Gaze: ({self.smooth_x:.2f}, {self.smooth_y:.2f})",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, cal, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        ret, _ = self.cap.read()
        if not ret:
            print("ERROR: cannot read from camera!")
            return

        print("✓ Camera ready")
        self.gui.show()
        self.gui.update_key_positions()

        if not self.tracker.is_calibrated:
            self.gui.status_label.setText(
                "⚠ Not calibrated — click 'Run Calibration' to begin"
            )

        print("\n" + "=" * 50)
        print("EYEWAVE — Getting Started")
        print("=" * 50)
        print("\n🎯 Click 'Run Calibration' and follow the dots")
        print("👁 After calibration the gaze cursor will track your eyes")
        print("🔤 Look at a key for ~1.3 seconds or blink to select it")
        print("💡 Good lighting + camera at eye level = best results")
        print("=" * 50 + "\n")

        self.app.aboutToQuit.connect(self.cleanup)
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = EyeWaveApp()
    app.run()
