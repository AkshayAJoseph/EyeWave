"""EyeWave Vision Module — faithful port of MonitorTracking.py's 3D gaze pipeline.

Pipeline (matching MonitorTracking.py exactly):
  1. MediaPipe Face Mesh → iris + nose landmarks
  2. PCA head orientation from nose region (compute_and_draw_coordinate_box logic)
  3. Eye sphere locking (iris position in head-local coords + camera-dir offset)
  4. Sphere world reconstruction (nose-scale compensation, per-eye)
  5. Binocular gaze direction (iris_3d - sphere_world, averaged, smoothed)
  6. Direct angular gaze→screen mapping (convert_gaze_to_screen_coordinates logic)
  7. Optional polynomial correction (residual refinement)
"""

import json
import os
import math
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
from mediapipe.python.solutions import face_mesh as fm

from .config import (
    MAX_NUM_FACES, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    LEFT_EYE_CORNERS, LEFT_EYE_LIDS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
    LEFT_IRIS, RIGHT_IRIS,
    BLINK_EAR_THRESHOLD, CALIBRATION_POLY_DEGREE, CALIBRATION_RIDGE_ALPHA,
    CALIBRATION_FILE,
    IRIS_H_CENTER, IRIS_H_GAIN, IRIS_V_CENTER, IRIS_V_GAIN,
    NOSE_INDICES, EYE_SPHERE_BASE_RADIUS,
    GAZE_SMOOTH_LENGTH, GAZE_YAW_RANGE, GAZE_PITCH_RANGE,
)
from .utils import eye_aspect_ratio


# ════════════════════════════════════════════════════════════════════════
#  Inline helpers — ported verbatim from MonitorTracking.py
# ════════════════════════════════════════════════════════════════════════

def _compute_scale(points_3d):
    """Average pairwise distance (MonitorTracking.py → compute_scale)."""
    n = len(points_3d)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(points_3d[i] - points_3d[j]))
            count += 1
    return total / count if count > 0 else 1.0


def _pca_orientation(points_3d, ref_matrix_container):
    """PCA head orientation (MonitorTracking.py → compute_and_draw_coordinate_box).

    Returns (center, R_final, points_3d).
    """
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    return center, R_final


def _convert_gaze_to_screen(combined_gaze_direction,
                            calibration_offset_yaw=0.0,
                            calibration_offset_pitch=0.0,
                            yaw_range=12.0, pitch_range=3.0):
    """Direct angular mapping (MonitorTracking.py → convert_gaze_to_screen_coordinates).

    Returns (norm_x, norm_y) in 0–1 range.
    """
    reference_forward = np.array([0, 0, -1], dtype=float)
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

    # Horizontal (yaw)
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]], dtype=float)
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad

    # Vertical (pitch)
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]], dtype=float)
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad

    yaw_deg = float(np.degrees(yaw_rad))
    pitch_deg = float(np.degrees(pitch_rad))

    # MonitorTracking sign convention
    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = -yaw_deg

    # Apply calibration offsets
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch

    # Map to normalised 0–1 screen coordinates
    norm_x = (yaw_deg + yaw_range) / (2 * yaw_range)
    norm_y = (pitch_range - pitch_deg) / (2 * pitch_range)

    norm_x = float(np.clip(norm_x, 0.0, 1.0))
    norm_y = float(np.clip(norm_y, 0.0, 1.0))

    return norm_x, norm_y


# ════════════════════════════════════════════════════════════════════════

class EyeTracker:
    """3D geometric eye tracker — faithful port of MonitorTracking.py."""

    def __init__(self):
        self.face_mesh = fm.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        # --- PCA state (MonitorTracking: R_ref_nose) ---
        self._ref_matrix = [None]

        # --- Eye sphere state (MonitorTracking: per-eye, not combined) ---
        self._left_sphere_locked = False
        self._left_sphere_local_offset = None
        self._left_calibration_nose_scale = None

        self._right_sphere_locked = False
        self._right_sphere_local_offset = None
        self._right_calibration_nose_scale = None

        # --- Gaze smoothing (MonitorTracking: combined_gaze_directions deque) ---
        self._gaze_buffer = deque(maxlen=GAZE_SMOOTH_LENGTH)

        # --- Screen calibration offsets (MonitorTracking: 's' key) ---
        self._calib_yaw_offset = 0.0
        self._calib_pitch_offset = 0.0

        # --- Polynomial correction model (EyeWave addition, optional) ---
        self._model_x = None
        self._model_y = None
        self._poly = None
        self._is_calibrated = False

        # Drift correction
        self._calib_feature_mean = None
        self._running_feature_mean = None
        self._drift_alpha = 0.01

        self._load_calibration()

    # -- Public API -------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def spheres_locked(self) -> bool:
        return self._left_sphere_locked and self._right_sphere_locked

    def process_frame(self, frame):
        """Process a camera frame.  Returns (success, is_blinking, gx, gy, features)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, False, 0.5, 0.5, None

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # 2D pixel coords
        pts = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])

        # Blink detection
        is_blinking = self._detect_blink(pts)

        # ── 1. PCA head orientation (exactly as MonitorTracking) ──────
        nose_points_3d = np.array([
            [face_landmarks[i].x * w,
             face_landmarks[i].y * h,
             face_landmarks[i].z * w]
            for i in NOSE_INDICES
        ], dtype=float)

        try:
            head_center, R_final = _pca_orientation(
                nose_points_3d, self._ref_matrix)
        except Exception as e:
            print(f"PCA error: {e}")
            return True, is_blinking, 0.5, 0.5, None

        # ── 2. Iris 3D positions (exactly as MonitorTracking) ─────────
        left_iris = face_landmarks[468]   # LEFT_IRIS[0]
        right_iris = face_landmarks[473]  # RIGHT_IRIS[0]

        iris_3d_left = np.array([left_iris.x * w,
                                 left_iris.y * h,
                                 left_iris.z * w], dtype=float)
        iris_3d_right = np.array([right_iris.x * w,
                                  right_iris.y * h,
                                  right_iris.z * w], dtype=float)

        # ── 3. Auto-lock spheres on first face detection ──────────────
        if not (self._left_sphere_locked and self._right_sphere_locked):
            self._lock_spheres(head_center, R_final,
                               iris_3d_left, iris_3d_right, nose_points_3d)

        # ── 4. Compute sphere world positions (per-eye, as MonitorTracking) ──
        gaze_x, gaze_y = 0.5, 0.5
        gaze_dir_smoothed = None
        sphere_world_l = None
        sphere_world_r = None

        if self._left_sphere_locked:
            current_nose_scale = _compute_scale(nose_points_3d)
            scale_ratio = (current_nose_scale / self._left_calibration_nose_scale
                           if self._left_calibration_nose_scale else 1.0)
            scaled_offset = self._left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset

        if self._right_sphere_locked:
            current_nose_scale = _compute_scale(nose_points_3d)
            scale_ratio_r = (current_nose_scale / self._right_calibration_nose_scale
                             if self._right_calibration_nose_scale else 1.0)
            scaled_offset_r = self._right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r

        # ── 5. Binocular gaze direction (exactly as MonitorTracking) ──
        if (self._left_sphere_locked and self._right_sphere_locked
                and sphere_world_l is not None and sphere_world_r is not None):

            # Individual gaze directions
            left_gaze_dir = iris_3d_left - sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)

            right_gaze_dir = iris_3d_right - sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)

            # Combined (average)
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            # Smooth via deque
            self._gaze_buffer.append(raw_combined_direction)
            avg_combined_direction = np.mean(self._gaze_buffer, axis=0)
            avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            gaze_dir_smoothed = avg_combined_direction

            # ── 6. Direct angular screen mapping ──────────────────────
            gaze_x, gaze_y = _convert_gaze_to_screen(
                avg_combined_direction,
                self._calib_yaw_offset,
                self._calib_pitch_offset,
                GAZE_YAW_RANGE, GAZE_PITCH_RANGE)

        # ── 7. Feature extraction for polynomial correction ───────────
        features = self._extract_features(
            pts, w, h, head_center, R_final, nose_points_3d,
            gaze_x, gaze_y, gaze_dir_smoothed)

        # ── 8. Apply polynomial correction if calibrated ──────────────
        if self._is_calibrated and features is not None:
            gaze_x, gaze_y = self._predict(features)

        return True, is_blinking, float(gaze_x), float(gaze_y), features

    def calibrate(self, feature_samples, screen_targets):
        """Fit polynomial residual correction model."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge

        X = np.array(feature_samples)
        Y = np.array(screen_targets)

        self._poly = PolynomialFeatures(
            degree=CALIBRATION_POLY_DEGREE, include_bias=True)
        X_poly = self._poly.fit_transform(X)

        self._model_x = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
        self._model_x.fit(X_poly, Y[:, 0])

        self._model_y = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
        self._model_y.fit(X_poly, Y[:, 1])

        self._is_calibrated = True
        self._calib_feature_mean = np.mean(X, axis=0)
        self._running_feature_mean = self._calib_feature_mean.copy()

        # Also compute & store the screen-center calibration offsets
        # (like MonitorTracking's 's' key — zero out yaw/pitch at center)
        if len(self._gaze_buffer) > 0:
            current_dir = np.mean(self._gaze_buffer, axis=0)
            n = np.linalg.norm(current_dir)
            if n > 1e-9:
                current_dir /= n
                _, _, raw_yaw, raw_pitch = self._raw_angles(current_dir)
                # Don't override — the polynomial model handles this

        self._save_calibration()
        print(f"Calibration complete — {len(feature_samples)} samples, "
              f"degree {CALIBRATION_POLY_DEGREE}")

    def lock_spheres_now(self, frame):
        """Re-lock eye spheres from an external call (e.g. calibration start)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        nose_points_3d = np.array([
            [face_landmarks[i].x * w,
             face_landmarks[i].y * h,
             face_landmarks[i].z * w]
            for i in NOSE_INDICES
        ], dtype=float)

        head_center, R_final = _pca_orientation(
            nose_points_3d, self._ref_matrix)

        left_iris = face_landmarks[468]
        right_iris = face_landmarks[473]
        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h,
                                 left_iris.z * w], dtype=float)
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h,
                                  right_iris.z * w], dtype=float)

        self._lock_spheres(head_center, R_final,
                           iris_3d_left, iris_3d_right, nose_points_3d)
        return True

    # -- Internal helpers -------------------------------------------------

    def _lock_spheres(self, head_center, R_final,
                      iris_3d_left, iris_3d_right, nose_points_3d):
        """Lock eye spheres — exactly as MonitorTracking.py 'c' key handler."""
        current_nose_scale = _compute_scale(nose_points_3d)
        base_radius = EYE_SPHERE_BASE_RADIUS

        camera_dir_world = np.array([0, 0, 1], dtype=float)
        camera_dir_local = R_final.T @ camera_dir_world

        # Lock LEFT eye
        self._left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        self._left_sphere_local_offset += base_radius * camera_dir_local
        self._left_calibration_nose_scale = current_nose_scale
        self._left_sphere_locked = True

        # Lock RIGHT eye
        self._right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        self._right_sphere_local_offset += base_radius * camera_dir_local
        self._right_calibration_nose_scale = current_nose_scale
        self._right_sphere_locked = True

        self._gaze_buffer.clear()
        print("[EyeWave] Both eye spheres locked")

    def _raw_angles(self, gaze_dir):
        """Get raw yaw/pitch angles from gaze direction (for calibration offsets)."""
        ref = np.array([0, 0, -1], dtype=float)
        d = gaze_dir / np.linalg.norm(gaze_dir)

        xz = np.array([d[0], 0, d[2]], dtype=float)
        xz /= np.linalg.norm(xz)
        yaw_rad = math.acos(np.clip(np.dot(ref, xz), -1.0, 1.0))
        if d[0] < 0:
            yaw_rad = -yaw_rad
        yaw_deg = float(np.degrees(yaw_rad))
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        yz = np.array([0, d[1], d[2]], dtype=float)
        yz /= np.linalg.norm(yz)
        pitch_rad = math.acos(np.clip(np.dot(ref, yz), -1.0, 1.0))
        if d[1] > 0:
            pitch_rad = -pitch_rad
        pitch_deg = float(np.degrees(pitch_rad))

        return 0, 0, yaw_deg, pitch_deg

    # -- Feature Extraction -----------------------------------------------

    def _extract_features(self, pts, w, h,
                          head_center, R_final, nose_pts_3d,
                          geo_gaze_x, geo_gaze_y, gaze_dir_smoothed):
        """12-dimensional feature vector for polynomial correction.

          0-1:  geometric gaze (x, y) — direct angular mapping
          2-4:  smoothed gaze direction (dx, dy, dz)
          5-6:  left iris H/V ratios (amplified)
          7-8:  right iris H/V ratios (amplified)
           9:   nose scale ratio
         10-11: head yaw, pitch (2D estimate)
        """
        try:
            if gaze_dir_smoothed is not None:
                gd_x = float(gaze_dir_smoothed[0])
                gd_y = float(gaze_dir_smoothed[1])
                gd_z = float(gaze_dir_smoothed[2])
            else:
                gd_x, gd_y, gd_z = 0.0, 0.0, -1.0

            l_h = self._iris_ratio_h(pts, LEFT_EYE_CORNERS, LEFT_IRIS)
            l_v = self._iris_ratio_v(pts, LEFT_EYE_LIDS, LEFT_IRIS)
            r_h = self._iris_ratio_h(pts, RIGHT_EYE_CORNERS, RIGHT_IRIS)
            r_v = self._iris_ratio_v(pts, RIGHT_EYE_LIDS, RIGHT_IRIS)

            l_h = 0.5 + (l_h - IRIS_H_CENTER) * IRIS_H_GAIN
            r_h = 0.5 + (r_h - IRIS_H_CENTER) * IRIS_H_GAIN
            l_v = 0.5 + (l_v - IRIS_V_CENTER) * IRIS_V_GAIN
            r_v = 0.5 + (r_v - IRIS_V_CENTER) * IRIS_V_GAIN

            if (self._left_calibration_nose_scale
                    and self._left_calibration_nose_scale > 0):
                current_scale = _compute_scale(nose_pts_3d)
                scale_ratio = current_scale / self._left_calibration_nose_scale
            else:
                scale_ratio = 1.0

            yaw_2d, pitch_2d = self._estimate_head_pose(pts)

            return np.array([
                geo_gaze_x, geo_gaze_y,
                gd_x, gd_y, gd_z,
                l_h, l_v, r_h, r_v,
                scale_ratio,
                yaw_2d, pitch_2d,
            ], dtype=np.float64)
        except Exception as e:
            print(f"Feature error: {e}")
            return None

    def _iris_ratio_h(self, pts, eye_corners, iris_indices):
        inner, outer = pts[eye_corners[0]], pts[eye_corners[1]]
        iris_c = pts[iris_indices[0]]
        width = outer[0] - inner[0]
        return float((iris_c[0] - inner[0]) / width) if abs(width) > 1 else 0.5

    def _iris_ratio_v(self, pts, eye_lids, iris_indices):
        top, bottom = pts[eye_lids[0]], pts[eye_lids[1]]
        iris_c = pts[iris_indices[0]]
        height = bottom[1] - top[1]
        if abs(height) < 1:
            return 0.5
        return float(np.clip((iris_c[1] - top[1]) / height, 0, 1))

    def _estimate_head_pose(self, pts):
        nose, l_eye, r_eye, chin = pts[1], pts[33], pts[263], pts[152]
        fw = r_eye[0] - l_eye[0]
        yaw = float(np.clip((nose[0] - l_eye[0]) / fw, 0, 1)) if abs(fw) > 1 else 0.5
        mid_y = (l_eye[1] + r_eye[1]) / 2.0
        fh = chin[1] - mid_y
        pitch = float(np.clip((nose[1] - mid_y) / fh, 0, 1)) if abs(fh) > 1 else 0.5
        return yaw, pitch

    # -- Blink Detection --------------------------------------------------

    def _detect_blink(self, pts):
        l_ear = eye_aspect_ratio(
            pts[LEFT_EYE_LIDS[0]], pts[LEFT_EYE_LIDS[1]],
            pts[LEFT_EYE_CORNERS[0]], pts[LEFT_EYE_CORNERS[1]])
        r_ear = eye_aspect_ratio(
            pts[RIGHT_EYE_LIDS[0]], pts[RIGHT_EYE_LIDS[1]],
            pts[RIGHT_EYE_CORNERS[0]], pts[RIGHT_EYE_CORNERS[1]])
        return (l_ear + r_ear) / 2.0 < BLINK_EAR_THRESHOLD

    # -- Prediction -------------------------------------------------------

    def _predict(self, features):
        if self._running_feature_mean is not None:
            self._running_feature_mean = (
                (1 - self._drift_alpha) * self._running_feature_mean +
                self._drift_alpha * features)

        if (self._calib_feature_mean is not None
                and self._running_feature_mean is not None):
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
        if not self._is_calibrated:
            return
        data = {
            "poly_degree": CALIBRATION_POLY_DEGREE,
            "model_x_coef": self._model_x.coef_.tolist(),
            "model_x_intercept": float(self._model_x.intercept_),
            "model_y_coef": self._model_y.coef_.tolist(),
            "model_y_intercept": float(self._model_y.intercept_),
            "n_features_in": self._poly.n_features_in_,
            "feature_mean": (self._calib_feature_mean.tolist()
                            if self._calib_feature_mean is not None else None),
            # Per-eye sphere state
            "left_sphere_offset": (self._left_sphere_local_offset.tolist()
                                   if self._left_sphere_local_offset is not None else None),
            "right_sphere_offset": (self._right_sphere_local_offset.tolist()
                                    if self._right_sphere_local_offset is not None else None),
            "left_calib_nose_scale": self._left_calibration_nose_scale,
            "right_calib_nose_scale": self._right_calibration_nose_scale,
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Calibration saved → {CALIBRATION_FILE}")

    def _load_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            return
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge

            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)

            n_features = data["n_features_in"]
            if n_features != 12:
                print(f"⚠ Saved calibration has {n_features} features, "
                      f"expected 12. Please recalibrate.")
                return

            self._poly = PolynomialFeatures(
                degree=data["poly_degree"], include_bias=True)
            self._poly.fit(np.zeros((1, n_features)))

            self._model_x = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
            self._model_x.coef_ = np.array(data["model_x_coef"])
            self._model_x.intercept_ = data["model_x_intercept"]
            self._model_x.n_features_in_ = self._poly.n_output_features_

            self._model_y = Ridge(alpha=CALIBRATION_RIDGE_ALPHA)
            self._model_y.coef_ = np.array(data["model_y_coef"])
            self._model_y.intercept_ = data["model_y_intercept"]
            self._model_y.n_features_in_ = self._poly.n_output_features_

            self._is_calibrated = True

            if data.get("feature_mean"):
                self._calib_feature_mean = np.array(data["feature_mean"])
                self._running_feature_mean = self._calib_feature_mean.copy()

            # Restore per-eye sphere state
            if data.get("left_sphere_offset"):
                self._left_sphere_local_offset = np.array(data["left_sphere_offset"])
                self._left_calibration_nose_scale = data["left_calib_nose_scale"]
                self._left_sphere_locked = True
            if data.get("right_sphere_offset"):
                self._right_sphere_local_offset = np.array(data["right_sphere_offset"])
                self._right_calibration_nose_scale = data["right_calib_nose_scale"]
                self._right_sphere_locked = True

            if self._left_sphere_locked and self._right_sphere_locked:
                print("Eye sphere state restored from calibration")

            print(f"Calibration loaded from {CALIBRATION_FILE}")
        except Exception as e:
            print(f"Could not load calibration: {e}")
            self._is_calibrated = False

    def __del__(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
