"""EyeWave Interface Module — Virtual keyboard and calibration overlay.

Contains:
  - WordPredictor:       Simple prefix-based word suggestions.
  - CalibrationOverlay:  Full-screen overlay that guides calibration.
  - VirtualKeyboard:     Gaze-controlled on-screen keyboard.
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QSlider, QFrame, QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient, QFont
import pyttsx3

from .config import (
    KEY_LABELS, KEYBOARD_COLS, KEY_SIZE, SPECIAL_KEY_SIZE, SPECIAL_KEYS,
    DWELL_THRESHOLD, KEY_HIT_MARGIN, NUM_PREDICTIONS, COMMON_WORDS,
    CALIBRATION_POINTS, CALIBRATION_SAMPLES_PER_POINT, TTS_RATE,
)


    # ── UI ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.instruction = QLabel(
            "Look at the RED dot and press SPACEBAR to start collecting.\n"
            "Keep looking until the green ring completes."
        )
        self.instruction.setStyleSheet(
            "font-size: 20px; padding: 15px; font-weight: bold; color: #ecf0f1;"
        )
        self.instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction)

        # Spacer to push progress label to bottom — dots are painted on dialog itself
        layout.addStretch(1)

        # Progress
        label = self._points[0][2] if self._points else ""
        self.progress = QLabel(f"Point 1/{len(self._points)}  --  {label}  |  Press SPACEBAR")
        self.progress.setStyleSheet("font-size: 15px; color: #bdc3c7; font-weight: bold;")
        self.progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress)

        # Cancel button (must NOT accept keyboard focus, otherwise spacebar closes it)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFocusPolicy(Qt.NoFocus)  # Prevent spacebar from activating it
        cancel_btn.setStyleSheet(
            "color: #ecf0f1; background-color: #e74c3c; padding: 8px 20px; "
            "border-radius: 4px; font-size: 13px;"
        )
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #0f0f23;")
        self.setFocusPolicy(Qt.StrongFocus)  # Dialog itself accepts keyboard focus
        self.setFocus()  # Ensure dialog has focus, not any child widget
        self.showMaximized()

    # ── Input ───────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if not self._collecting and not self._done:
                self._collecting = True
                self._current_features = []
                name = self._points[self._current_idx][2]
                self.instruction.setText(f"KEEP LOOKING at the dot!  ({name})")
                self.progress.setText(
                    f"Point {self._current_idx + 1}/{len(self._points)}  --  Collecting..."
                )
            event.accept()  # Consume spacebar — do NOT pass to QDialog
            return
        elif event.key() == Qt.Key_Escape and self._collecting:
            event.accept()  # Don't close while collecting
            return
        super().keyPressEvent(event)

    # ── Drawing ─────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._done or self._current_idx >= len(self._points):
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        tx, ty, _ = self._points[self._current_idx]

        # Draw area: full dialog with margins for instruction/progress labels
        margin_top = 80
        margin_bottom = 80
        draw_w = self.width()
        draw_h = self.height() - margin_top - margin_bottom

        cx = int(tx * draw_w)
        cy = int(margin_top + ty * draw_h)

        # Outer glow
        for i in range(3):
            alpha = 40 - i * 12
            r = 45 + i * 15
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(231, 76, 60, alpha)))
            painter.drawEllipse(QPoint(cx, cy), r, r)

        # Main dot
        painter.setBrush(QBrush(QColor(231, 76, 60)))
        painter.setPen(QPen(QColor(192, 57, 43), 3))
        painter.drawEllipse(QPoint(cx, cy), 28, 28)

        # White center
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(cx, cy), 6, 6)

        # Progress ring
        if self._collecting and self._current_features:
            frac = len(self._current_features) / self._samples_per_point
            painter.setPen(QPen(QColor(46, 204, 113), 5))
            painter.setBrush(Qt.NoBrush)
            span = -int(360 * 16 * frac)
            painter.drawArc(cx - 45, cy - 45, 90, 90, 90 * 16, span)

            # Percentage
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 11))
            painter.drawText(cx - 15, cy + 65, f"{int(frac * 100)}%")

    # ── Data Collection ─────────────────────────────────────────────────

    def add_features(self, features):
        """Called every frame by the main loop while calibrating."""
        if not self._collecting or features is None:
            return

        self._current_features.append(features.copy())

        if len(self._current_features) >= self._samples_per_point:
            tx, ty, name = self._points[self._current_idx]

            # Convert the drawn dot position to normalized *screen* coordinates.
            # paintEvent draws with margin_top/bottom=80 inside the overlay, so
            # the dot's true pixel position on the overlay is:
            #   dot_x = tx * overlay_width
            #   dot_y = margin_top + ty * (overlay_height - margin_top - margin_bottom)
            # We then map that to global screen coords for a proper target.
            margin_top = 80
            margin_bottom = 80
            overlay_w = self.width()
            overlay_h = self.height()
            draw_h = overlay_h - margin_top - margin_bottom

            dot_local_x = int(tx * overlay_w)
            dot_local_y = int(margin_top + ty * draw_h)

            global_pt = self.mapToGlobal(QPoint(dot_local_x, dot_local_y))
            screen = QApplication.primaryScreen().geometry()
            target_x = global_pt.x() / screen.width()
            target_y = global_pt.y() / screen.height()

            # Store all individual samples with the actual screen target
            for feat in self._current_features:
                self._all_features.append(feat)
                self._all_targets.append((target_x, target_y))

            self._current_idx += 1
            self._current_features = []
            self._collecting = False

            if self._current_idx < len(self._points):
                next_name = self._points[self._current_idx][2]
                self.instruction.setText(
                    f"Good!  Now look at the next dot.\n"
                    f"Press SPACEBAR when ready for: {next_name}"
                )
                self.progress.setText(
                    f"Point {self._current_idx + 1}/{len(self._points)}  —  Press SPACEBAR"
                )
            else:
                self._done = True
                self.instruction.setText("✓  Calibration complete!")
                self.progress.setText("All points collected — closing…")
                QTimer.singleShot(1200, self.accept)

        self.update()

    def get_calibration_data(self):
        """Return (features_list, targets_list) collected during calibration."""
        return self._all_features, self._all_targets

    def accept(self):
        """Mimic QDialog.accept() — emit finished(1) and close."""
        self.finished.emit(1)
        self.close()

    def reject(self):
        """Mimic QDialog.reject() — emit finished(0) and close."""
        self.finished.emit(0)
        self.close()


# ════════════════════════════════════════════════════════════════════════
#  Virtual Keyboard
# ════════════════════════════════════════════════════════════════════════

class VirtualKeyboard(QWidget):
    """Gaze-controlled virtual keyboard with word prediction."""

    def __init__(self):
        super().__init__()
        self.keys_widgets = []
        self.key_rects = []
        self.current_focused_key = None
        self.dwell_time = 0
        self.dwell_threshold = DWELL_THRESHOLD

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", TTS_RATE)

        self.predictor = WordPredictor()
        self.prediction_buttons = []

        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.show_gaze_cursor = True

        self._init_ui()

    # ── UI ──────────────────────────────────────────────────────────────

    def _init_ui(self):
        main = QVBoxLayout()

        # Text display
        self.text_display = QLabel("")
        self.text_display.setStyleSheet("""
            font-size: 36px; border: 3px solid #2c3e50;
            padding: 15px; background-color: white; min-height: 80px;
        """)
        self.text_display.setWordWrap(True)
        main.addWidget(self.text_display)

        # Predictions
        pred_layout = QHBoxLayout()
        pred_layout.addWidget(self._styled_label("Suggestions:", 18, True))
        for _ in range(NUM_PREDICTIONS):
            btn = QPushButton("")
            btn.setFixedHeight(50)
            btn.setStyleSheet(self._pred_style(False, 0))
            btn.hide()
            self.prediction_buttons.append(btn)
            pred_layout.addWidget(btn)
        pred_layout.addStretch()
        main.addLayout(pred_layout)

        # Status
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Look at a key to select • Calibrate first")
        self.status_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        status_layout.addWidget(self.status_label)
        self.dwell_indicator = QLabel("")
        self.dwell_indicator.setFixedSize(150, 15)
        self.dwell_indicator.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")
        status_layout.addWidget(self.dwell_indicator)
        main.addLayout(status_layout)

        # Debug
        self.debug_label = QLabel("Gaze: 0.50, 0.50")
        self.debug_label.setStyleSheet("font-size: 11px; color: #95a5a6;")
        main.addWidget(self.debug_label)

        # Keyboard grid
        grid = QGridLayout()
        grid.setSpacing(6)
        row, col = 0, 0
        for label in KEY_LABELS:
            btn = QPushButton(label)
            sz = SPECIAL_KEY_SIZE if label in SPECIAL_KEYS else KEY_SIZE
            btn.setFixedSize(*sz)
            btn.setStyleSheet(self._key_style(False, 0))
            grid.addWidget(btn, row, col)
            self.keys_widgets.append(btn)
            col += 1
            if col >= KEYBOARD_COLS:
                col = 0
                row += 1
        main.addLayout(grid)

        # Buttons row
        btn_row = QHBoxLayout()
        self.cal_btn = QPushButton("🎯 Run Calibration")
        self.cal_btn.setStyleSheet("""
            font-size: 14px; font-weight: bold;
            background-color: #e74c3c; color: white;
            padding: 10px; border-radius: 5px;
        """)
        btn_row.addWidget(self.cal_btn)

        self.toggle_cursor_btn = QPushButton("👁 Toggle Gaze Cursor")
        self.toggle_cursor_btn.setStyleSheet("""
            font-size: 14px; background-color: #3498db;
            color: white; padding: 10px; border-radius: 5px;
        """)
        self.toggle_cursor_btn.clicked.connect(self._toggle_cursor)
        btn_row.addWidget(self.toggle_cursor_btn)

        main.addLayout(btn_row)

        self.setLayout(main)
        self.setWindowTitle("EyeWave — Gaze-Controlled Keyboard")
        self.setGeometry(50, 50, 800, 900)

    # ── Gaze painting ───────────────────────────────────────────────────

    def _gaze_to_local(self, gx, gy):
        """Convert normalized screen gaze (0-1) to widget-local pixel coords.

        Calibration targets are in normalized *screen* coordinates, so we must
        first map to absolute screen pixels, then convert to the widget's
        local coordinate system. This ensures alignment regardless of window
        size, position, or fullscreen state.
        """
        screen = QApplication.primaryScreen().geometry()
        screen_x = int(gx * screen.width())
        screen_y = int(gy * screen.height())
        local = self.mapFromGlobal(QPoint(screen_x, screen_y))
        return local.x(), local.y()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.show_gaze_cursor:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sx, sy = self._gaze_to_local(self.gaze_x, self.gaze_y)

        # Gradient glow
        g = QRadialGradient(sx, sy, 30)
        g.setColorAt(0, QColor(52, 152, 219, 100))
        g.setColorAt(0.7, QColor(52, 152, 219, 50))
        g.setColorAt(1, QColor(52, 152, 219, 0))
        painter.setBrush(QBrush(g))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(sx, sy), 30, 30)

        # Crosshair
        painter.setPen(QPen(QColor(41, 128, 185), 3))
        painter.drawLine(sx - 20, sy, sx + 20, sy)
        painter.drawLine(sx, sy - 20, sx, sy + 20)

        # Center dot
        painter.setBrush(QBrush(QColor(231, 76, 60)))
        painter.setPen(QPen(QColor(192, 57, 43), 2))
        painter.drawEllipse(QPoint(sx, sy), 6, 6)

    # ── Gaze update ─────────────────────────────────────────────────────

    def update_key_positions(self):
        self.key_rects = []
        m = KEY_HIT_MARGIN
        for btn in self.keys_widgets:
            pos = btn.mapToGlobal(btn.rect().topLeft())
            local = self.mapFromGlobal(pos)
            # Expand hit area by margin for easier gaze targeting
            self.key_rects.append(QRect(
                local.x() - m, local.y() - m,
                btn.width() + 2 * m, btn.height() + 2 * m))
        for btn in self.prediction_buttons:
            if btn.isVisible():
                pos = btn.mapToGlobal(btn.rect().topLeft())
                local = self.mapFromGlobal(pos)
                self.key_rects.append(QRect(
                    local.x() - m, local.y() - m,
                    btn.width() + 2 * m, btn.height() + 2 * m))

    def update_gaze(self, gx, gy):
        self.gaze_x = gx
        self.gaze_y = gy
        self.update()  # repaint cursor

        self.debug_label.setText(f"Gaze: {gx:.2f}, {gy:.2f}")

        # Refresh key rects if needed
        expected = len(self.keys_widgets) + sum(1 for b in self.prediction_buttons if b.isVisible())
        if not self.key_rects or len(self.key_rects) != expected:
            self.update_key_positions()

        sx, sy = self._gaze_to_local(gx, gy)

        focused_idx = None
        is_pred = False

        for idx, rect in enumerate(self.key_rects[:len(self.keys_widgets)]):
            if rect.contains(sx, sy):
                focused_idx = idx
                break
        if focused_idx is None:
            offset = len(self.keys_widgets)
            for idx, rect in enumerate(self.key_rects[offset:]):
                if rect.contains(sx, sy):
                    focused_idx = idx
                    is_pred = True
                    break

        if focused_idx is not None:
            key_id = (focused_idx, is_pred)
            if key_id == self.current_focused_key:
                self.dwell_time += 1
                prog = min(1.0, self.dwell_time / self.dwell_threshold)
                if is_pred:
                    self._style_preds(focused_idx, prog)
                else:
                    self._style_keys(focused_idx, prog)
                self._dwell_bar(prog)
                if self.dwell_time >= self.dwell_threshold:
                    if is_pred:
                        self.select_prediction(focused_idx)
                    else:
                        self.select_key(focused_idx)
                    self.dwell_time = 0
                    self.current_focused_key = None
            else:
                self.current_focused_key = key_id
                self.dwell_time = 1
                if is_pred:
                    self._style_preds(focused_idx, 0)
                else:
                    self._style_keys(focused_idx, 0)
        else:
            self.dwell_time = 0
            self.current_focused_key = None
            self._style_keys(None, 0)
            self._style_preds(None, 0)
            self._dwell_bar(0)

    # ── Selection ───────────────────────────────────────────────────────

    def select_key(self, idx):
        char = KEY_LABELS[idx]
        text = self.text_display.text()
        if char == "SPACE":
            self.text_display.setText(text + " ")
        elif char == "BACK":
            self.text_display.setText(text[:-1])
        elif char == "CLR":
            self.text_display.setText("")
        elif char == "SPEAK":
            if text:
                self.engine.say(text)
                self.engine.runAndWait()
        else:
            self.text_display.setText(text + char)
        self._update_predictions()
        self.status_label.setText(f"Selected: {char}")

    def select_prediction(self, idx):
        if idx < len(self.prediction_buttons) and self.prediction_buttons[idx].isVisible():
            word = self.prediction_buttons[idx].text()
            text = self.text_display.text()
            words = text.split()
            if words:
                words[-1] = word
                self.text_display.setText(" ".join(words) + " ")
            else:
                self.text_display.setText(word + " ")
            self._update_predictions()
            self.status_label.setText(f"Selected word: {word}")

    def reset_dwell(self):
        self.dwell_time = 0
        self.current_focused_key = None

    # ── Helpers ─────────────────────────────────────────────────────────

    def _toggle_cursor(self):
        self.show_gaze_cursor = not self.show_gaze_cursor
        self.update()

    def _update_predictions(self):
        text = self.text_display.text()
        words = text.split()
        current = words[-1] if words else ""
        preds = self.predictor.get_predictions(current, NUM_PREDICTIONS)
        for i, btn in enumerate(self.prediction_buttons):
            if i < len(preds):
                btn.setText(preds[i])
                btn.show()
            else:
                btn.hide()
        self.update_key_positions()

    def _style_keys(self, focused_idx, prog):
        for i, btn in enumerate(self.keys_widgets):
            btn.setStyleSheet(self._key_style(i == focused_idx, prog))

    def _style_preds(self, focused_idx, prog):
        for i, btn in enumerate(self.prediction_buttons):
            if btn.isVisible():
                btn.setStyleSheet(self._pred_style(i == focused_idx, prog))

    def _dwell_bar(self, prog):
        self.dwell_indicator.setStyleSheet(f"""
            border: 2px solid #bdc3c7;
            background: qlineargradient(x1:0, x2:1,
                stop:0 #3498db, stop:{prog} #3498db,
                stop:{prog} #ecf0f1, stop:1 #ecf0f1);
        """)

    @staticmethod
    def _key_style(focused, prog):
        if focused:
            r = int(52 + (46 - 52) * prog)
            g = int(152 + (204 - 152) * prog)
            b = int(219 + (113 - 219) * prog)
            return f"""
                font-size: 22px; font-weight: bold;
                background-color: rgb({r},{g},{b}); color: white;
                border: 4px solid #2c3e50; border-radius: 8px;
            """
        return """
            font-size: 20px; background-color: #ecf0f1;
            color: #2c3e50; border: 2px solid #bdc3c7; border-radius: 6px;
        """

    @staticmethod
    def _pred_style(focused, prog):
        if focused:
            r = int(52 + (46 - 52) * prog)
            g = int(152 + (204 - 152) * prog)
            b = int(219 + (113 - 219) * prog)
            return f"""
                font-size: 18px; font-weight: bold;
                background-color: rgb({r},{g},{b}); color: white;
                border: 3px solid #2c3e50; border-radius: 5px;
            """
        return """
            font-size: 16px; background-color: #3498db;
            color: white; border: 2px solid #2980b9; border-radius: 5px;
        """

    @staticmethod
    def _styled_label(text, size, bold=False):
        lbl = QLabel(text)
        w = "bold" if bold else "normal"
        lbl.setStyleSheet(f"font-size: {size}px; font-weight: {w};")
        return lbl
