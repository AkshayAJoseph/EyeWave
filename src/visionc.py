"""EyeWave Vision Module — Eye tracking with polynomial calibration.

Uses MediaPipe Face Mesh (solutions API) for iris / landmark detection
and scikit-learn Ridge regression with polynomial features to learn a
per-user mapping from raw eye features to screen (x, y) coordinates.
"""

import json
import os
import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh as fm

from .config import (
    MAX_NUM_FACES, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    LEFT_EYE_CORNERS, LEFT_EYE_LIDS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
    LEFT_IRIS, RIGHT_IRIS,
    BLINK_EAR_THRESHOLD, CALIBRATION_POLY_DEGREE, CALIBRATION_RIDGE_ALPHA,
    CALIBRATION_FILE,
    IRIS_H_CENTER, IRIS_H_GAIN, IRIS_V_CENTER, IRIS_V_GAIN,
    FACE_3D_MODEL, CAMERA_FOCAL_LENGTH_FACTOR,
)
from .utils import eye_aspect_ratio, estimate_3d_head_pose, compute_gaze_ray_intersection


class EyeTracker:
    """MediaPipe-based eye tracker with polynomial calibration."""

    def __init__(self):
        self.face_mesh = fm.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        # Calibration model (None until calibrated)
        self._model_x = None
        self._model_y = None
        self._poly = None
        self._is_calibrated = False

        # Drift correction state
        self._calib_feature_mean = None   # Mean features at calibration time
        self._running_feature_mean = None # Running EMA of features during live use
        self._drift_alpha = 0.01         # How fast running mean adapts (low = slow)

        # Try to load saved calibration
        self._load_calibration()

    # -- Public API -------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def process_frame(self, frame):
        """Process a camera frame and return tracking results.

        Returns:
            (success, is_blinking, gaze_x, gaze_y, features)
            - success: True if a face was detected.
            - is_blinking: True if the user is blinking.
            - gaze_x, gaze_y: Predicted screen coordinates (0-1) if calibrated,
              otherwise raw iris ratios.
            - features: Raw feature vector (used during calibration).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, False, 0.5, 0.5, None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Convert to pixel coordinates
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

        # Extract features (now needs frame height for solvePnP camera matrix)
        features = self._extract_features(pts, w, h)
        is_blinking = self._detect_blink(pts)

        if self._is_calibrated and features is not None:
            gaze_x, gaze_y = self._predict(features)
        else:
            # Fallback: average iris ratio
            gaze_x = features[0] if features is not None else 0.5
            gaze_y = features[1] if features is not None else 0.5

        return True, is_blinking, float(gaze_x), float(gaze_y), features

    def calibrate(self, feature_samples, screen_targets):
        """Fit the polynomial regression model from calibration data.

        Args:
            feature_samples: list of feature vectors (one per sample).
            screen_targets:  list of (x, y) tuples -- known screen positions.
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge

        X = np.array(feature_samples)
        Y = np.array(screen_targets)

        self._poly = PolynomialFeatures(degree=CALIBRATION_POLY_DEGREE, include_bias=True)
        X_poly = self._poly.fit_transform(X)

        self._model_x = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
        self._model_x.fit(X_poly, Y[:, 0])

        self._model_y = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
        self._model_y.fit(X_poly, Y[:, 1])

        self._is_calibrated = True

        # Store calibration baseline for drift correction
        self._calib_feature_mean = np.mean(X, axis=0)
        self._running_feature_mean = self._calib_feature_mean.copy()

        self._save_calibration()

        print(f"Calibration complete -- {len(feature_samples)} samples, "
              f"degree {CALIBRATION_POLY_DEGREE}")

    # -- Feature Extraction -----------------------------------------------

    def _extract_features(self, pts, frame_w, frame_h=None):
        """Build a feature vector from current face landmarks.

        Features (18-dimensional):
          0:  left iris H ratio (amplified)
          1:  left iris V ratio (amplified)
          2:  right iris H ratio (amplified)
          3:  right iris V ratio (amplified)
          4:  averaged H ratio (L+R)/2 (amplified) — strongest horizontal signal
          5:  averaged V ratio (L+R)/2 (amplified)
          6:  head yaw   (2D estimate)
          7:  head pitch (2D estimate)
          8:  left  eye aspect ratio
          9:  right eye aspect ratio
         10:  iris H pixel delta / face width (scale-invariant horizontal offset)
         11:  iris V pixel delta / face width
         12:  3D head yaw   (solvePnP Euler, degrees, /180 normalised)
         13:  3D head pitch (solvePnP Euler, degrees, /180 normalised)
         14:  3D head roll  (solvePnP Euler, degrees, /180 normalised)
         15:  head distance  (tvec Z / face_width, scale-invariant depth)
         16:  gaze ray intersection X (0-1 screen coordinate)
         17:  gaze ray intersection Y (0-1 screen coordinate)
        """
        try:
            # Default frame_h from aspect ratio if not supplied
            if frame_h is None:
                frame_h = int(frame_w * 480 / 640)

            # Raw iris ratios (both eyes)
            l_iris_h_raw = self._iris_ratio_h(pts, LEFT_EYE_CORNERS, LEFT_IRIS)
            l_iris_v_raw = self._iris_ratio_v(pts, LEFT_EYE_LIDS, LEFT_IRIS)
            r_iris_h_raw = self._iris_ratio_h(pts, RIGHT_EYE_CORNERS, RIGHT_IRIS)
            r_iris_v_raw = self._iris_ratio_v(pts, RIGHT_EYE_LIDS, RIGHT_IRIS)

            # Amplify: re-center around 0.5 and stretch
            l_iris_h = 0.5 + (l_iris_h_raw - IRIS_H_CENTER) * IRIS_H_GAIN
            r_iris_h = 0.5 + (r_iris_h_raw - IRIS_H_CENTER) * IRIS_H_GAIN
            l_iris_v = 0.5 + (l_iris_v_raw - IRIS_V_CENTER) * IRIS_V_GAIN
            r_iris_v = 0.5 + (r_iris_v_raw - IRIS_V_CENTER) * IRIS_V_GAIN

            # Averaged ratios (reduces per-eye noise, strong overall signal)
            avg_h = (l_iris_h + r_iris_h) / 2.0
            avg_v = (l_iris_v + r_iris_v) / 2.0

            # Head pose (lightweight 2D estimate — kept for continuity)
            yaw_2d, pitch_2d = self._estimate_head_pose(pts)

            # Eye Aspect Ratios
            l_ear = eye_aspect_ratio(
                pts[LEFT_EYE_LIDS[0]], pts[LEFT_EYE_LIDS[1]],
                pts[LEFT_EYE_CORNERS[0]], pts[LEFT_EYE_CORNERS[1]])
            r_ear = eye_aspect_ratio(
                pts[RIGHT_EYE_LIDS[0]], pts[RIGHT_EYE_LIDS[1]],
                pts[RIGHT_EYE_CORNERS[0]], pts[RIGHT_EYE_CORNERS[1]])

            # Iris pixel delta (scale-invariant via face width)
            face_width = abs(pts[263][0] - pts[33][0])
            if face_width < 1:
                face_width = 1.0
            l_iris_center = pts[LEFT_IRIS[0]]
            r_iris_center = pts[RIGHT_IRIS[0]]
            avg_iris_x = (l_iris_center[0] + r_iris_center[0]) / 2.0
            avg_iris_y = (l_iris_center[1] + r_iris_center[1]) / 2.0
            face_center_x = (pts[33][0] + pts[263][0]) / 2.0
            face_center_y = (pts[33][1] + pts[263][1]) / 2.0
            iris_dx = (avg_iris_x - face_center_x) / face_width
            iris_dy = (avg_iris_y - face_center_y) / face_width

            # ── 3D head pose (solvePnP) ─────────────────────────────────
            rvec, tvec, euler, pose_ok = estimate_3d_head_pose(
                pts, frame_w, frame_h,
                FACE_3D_MODEL, CAMERA_FOCAL_LENGTH_FACTOR,
            )

            if pose_ok:
                yaw_3d   = euler[0] / 180.0   # normalise to roughly [-1, 1]
                pitch_3d = euler[1] / 180.0
                roll_3d  = euler[2] / 180.0
                head_dist = float(tvec[2, 0]) / face_width  # depth / face width
            else:
                yaw_3d = pitch_3d = roll_3d = 0.0
                head_dist = 0.0

            # ── Gaze ray intersection ───────────────────────────────────
            if pose_ok:
                gaze_x, gaze_y = compute_gaze_ray_intersection(
                    rvec, tvec,
                    l_iris_center, r_iris_center,
                    frame_w, frame_h,
                    CAMERA_FOCAL_LENGTH_FACTOR,
                )
            else:
                gaze_x, gaze_y = 0.5, 0.5

            return np.array([
                l_iris_h, l_iris_v,
                r_iris_h, r_iris_v,
                avg_h, avg_v,
                yaw_2d, pitch_2d,
                l_ear, r_ear,
                iris_dx, iris_dy,
                yaw_3d, pitch_3d, roll_3d,
                head_dist,
                gaze_x, gaze_y,
            ], dtype=np.float64)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def _iris_ratio_h(self, pts, eye_corners, iris_indices):
        """Horizontal iris position ratio (0=inner corner, 1=outer corner)."""
        inner = pts[eye_corners[0]]
        outer = pts[eye_corners[1]]
        iris_center = pts[iris_indices[0]]
        width = outer[0] - inner[0]
        if abs(width) < 1:
            return 0.5
        return float((iris_center[0] - inner[0]) / width)  # Don't clip — let amplification work

    def _iris_ratio_v(self, pts, eye_lids, iris_indices):
        """Vertical iris position ratio (0=top lid, 1=bottom lid)."""
        top = pts[eye_lids[0]]
        bottom = pts[eye_lids[1]]
        iris_center = pts[iris_indices[0]]
        height = bottom[1] - top[1]
        if abs(height) < 1:
            return 0.5
        return float(np.clip((iris_center[1] - top[1]) / height, 0, 1))

    def _estimate_head_pose(self, pts):
        """Lightweight 2D head pose estimate (yaw, pitch) in normalized range."""
        nose = pts[1]
        left_eye = pts[33]
        right_eye = pts[263]
        chin = pts[152]

        # Yaw: nose position between eye corners
        face_width = right_eye[0] - left_eye[0]
        if abs(face_width) < 1:
            yaw = 0.5
        else:
            yaw = float(np.clip((nose[0] - left_eye[0]) / face_width, 0, 1))

        # Pitch: nose position between eye midpoint and chin
        eyes_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        face_height = chin[1] - eyes_mid_y
        if abs(face_height) < 1:
            pitch = 0.5
        else:
            pitch = float(np.clip((nose[1] - eyes_mid_y) / face_height, 0, 1))

        return yaw, pitch

    # -- Blink Detection --------------------------------------------------

    def _detect_blink(self, pts):
        """Return True if the user is currently blinking (both eyes)."""
        l_ear = eye_aspect_ratio(
            pts[LEFT_EYE_LIDS[0]], pts[LEFT_EYE_LIDS[1]],
            pts[LEFT_EYE_CORNERS[0]], pts[LEFT_EYE_CORNERS[1]])
        r_ear = eye_aspect_ratio(
            pts[RIGHT_EYE_LIDS[0]], pts[RIGHT_EYE_LIDS[1]],
            pts[RIGHT_EYE_CORNERS[0]], pts[RIGHT_EYE_CORNERS[1]])
        avg_ear = (l_ear + r_ear) / 2.0
        return avg_ear < BLINK_EAR_THRESHOLD

    # -- Prediction -------------------------------------------------------

    def _predict(self, features):
        """Map raw features -> screen (x, y) with online drift correction."""
        # Update running mean of features (EMA)
        if self._running_feature_mean is not None:
            self._running_feature_mean = (
                (1 - self._drift_alpha) * self._running_feature_mean +
                self._drift_alpha * features
            )

        # Apply drift correction: re-center features around calibration baseline
        if self._calib_feature_mean is not None and self._running_feature_mean is not None:
            drift = self._running_feature_mean - self._calib_feature_mean
            corrected = features - drift
        else:
            corrected = features

        X = self._poly.transform(corrected.reshape(1, -1))
        gx = float(np.clip(self._model_x.predict(X)[0], 0, 1))
        gy = float(np.clip(self._model_y.predict(X)[0], 0, 1))
        return gx, gy

    # -- Persistence ------------------------------------------------------

    def _save_calibration(self):
        """Save model parameters to a JSON file."""
        if not self._is_calibrated:
            return
        data = {
            "poly_degree": CALIBRATION_POLY_DEGREE,
            "model_x_coef": self._model_x.coef_.tolist(),
            "model_x_intercept": float(self._model_x.intercept_),
            "model_y_coef": self._model_y.coef_.tolist(),
            "model_y_intercept": float(self._model_y.intercept_),
            "n_features_in": self._poly.n_features_in_,
            "feature_mean": self._calib_feature_mean.tolist() if self._calib_feature_mean is not None else None,
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Calibration saved to {CALIBRATION_FILE}")

    def _load_calibration(self):
        """Load model parameters from a JSON file (if it exists)."""
        if not os.path.exists(CALIBRATION_FILE):
            return

        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge

            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)

            n_features = data["n_features_in"]
            degree = data["poly_degree"]

            self._poly = PolynomialFeatures(degree=degree, include_bias=True)
            dummy = np.zeros((1, n_features))
            self._poly.fit(dummy)

            self._model_x = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
            self._model_x.coef_ = np.array(data["model_x_coef"])
            self._model_x.intercept_ = data["model_x_intercept"]
            self._model_x.n_features_in_ = self._poly.n_output_features_

            self._model_y = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
            self._model_y.coef_ = np.array(data["model_y_coef"])
            self._model_y.intercept_ = data["model_y_intercept"]
            self._model_y.n_features_in_ = self._poly.n_output_features_

            # Check feature dimension compatibility (12 → 18 migration)
            expected_features = 18
            if n_features != expected_features:
                print(f"⚠ Saved calibration has {n_features} features, "
                      f"expected {expected_features}. Please recalibrate.")
                return

            self._is_calibrated = True

            # Restore drift correction baseline
            if "feature_mean" in data and data["feature_mean"] is not None:
                self._calib_feature_mean = np.array(data["feature_mean"])
                self._running_feature_mean = self._calib_feature_mean.copy()

            print(f"Calibration loaded from {CALIBRATION_FILE}")
        except Exception as e:
            print(f"Could not load calibration: {e}")
            self._is_calibrated = False

    # -- Cleanup ----------------------------------------------------------

    def __del__(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
