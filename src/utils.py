"""
utils.py
========
Pure utility functions used across the pipeline:
  - 3D math helpers (rotation matrices, normalise, focal length)
  - PCA head-pose estimation
  - Monitor plane creation
  - Ray-plane intersection
  - CalibrationManager  (save / load  calibration.json)
  - GazeDataCollector   (append rows to  gaze_data.csv)

None of these functions depend on OpenCV windows or UI state.
"""

import cv2
import csv
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from src.config import CALIB_FILE, GAZE_DATA_FILE


# ─────────────────────────────────────────────────────────────────────────────
#  ROTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rot_x(a: float) -> np.ndarray:
    """3×3 rotation matrix around X axis by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0,  0  ],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    """3×3 rotation matrix around Y axis by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns v unchanged if near-zero."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def focal_px(width: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels from image width and horizontal FOV."""
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  SCALE ESTIMATION  (used to track head distance changes)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale(pts: np.ndarray) -> float:
    """Mean pairwise distance of a point cloud. Robust distance proxy."""
    n = len(pts)
    total = count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(pts[i] - pts[j])
            count += 1
    return total / count if count > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  PCA HEAD POSE
# ─────────────────────────────────────────────────────────────────────────────

def pca_orientation(points_3d: np.ndarray,
                    ref_container: list) -> tuple:
    """
    Compute head centre and rotation matrix from a point cloud via PCA.

    ref_container : [None] on first call; stores reference matrix to prevent
                    eigenvector sign flips between frames.

    Returns
    -------
    center : (3,) world-space head centre
    R      : (3,3) rotation matrix (head frame)
    """
    center   = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov      = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs  = eigvecs[:, np.argsort(-eigvals)]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # Stabilise sign to prevent eigenvector flips
    if ref_container[0] is None:
        ref_container[0] = R.copy()
    else:
        for i in range(3):
            if np.dot(R[:, i], ref_container[0][:, i]) < 0:
                R[:, i] *= -1

    return center, R


# ─────────────────────────────────────────────────────────────────────────────
#  MONITOR PLANE  (3D world-space quad the gaze ray intersects)
# ─────────────────────────────────────────────────────────────────────────────

def create_monitor_plane(head_center, R_final, face_landmarks,
                         fw: int, fh: int,
                         fwd=None, go=None, gd=None):
    """
    Build a 60 cm × 40 cm plane 50 cm in front of the face.

    Parameters
    ----------
    head_center     : (3,) world-space head centre (from pca_orientation)
    R_final         : (3,3) head rotation matrix
    face_landmarks  : MediaPipe landmark list
    fw, fh          : camera frame width / height (pixels)
    fwd             : optional forward-hint direction (normalised)
    go              : gaze origin (midpoint between eye spheres)
    gd              : gaze direction (combined normalised)

    Returns
    -------
    corners    : [p0, p1, p2, p3]  four world-space corner points
    center_w   : (3,) world-space plane centre
    normal_w   : (3,) unit outward normal
    upc        : float  world-units-per-centimetre
    """
    # Estimate scale: chin-to-forehead ≈ 15 cm
    try:
        lc = face_landmarks[152]
        lf = face_landmarks[10]
        chin_w = np.array([lc.x * fw, lc.y * fh, lc.z * fw], dtype=float)
        fore_w = np.array([lf.x * fw, lf.y * fh, lf.z * fw], dtype=float)
        upc = np.linalg.norm(fore_w - chin_w) / 15.0
    except Exception:
        upc = 5.0

    half_w = 30.0 * upc   # 60 cm wide
    half_h = 20.0 * upc   # 40 cm tall

    head_fwd = -R_final[:, 2]
    if fwd is not None:
        head_fwd = np.asarray(fwd) / np.linalg.norm(fwd)

    # Place centre on gaze ray if available, else 50 cm straight ahead
    if go is not None and gd is not None:
        gdn = gd / np.linalg.norm(gd)
        plane_pt = head_center + head_fwd * (50.0 * upc)
        dn = np.dot(head_fwd, gdn)
        center_w = (go + np.dot(head_fwd, plane_pt - go) / dn * gdn
                    if abs(dn) > 1e-6
                    else head_center + head_fwd * (50.0 * upc))
    else:
        center_w = head_center + head_fwd * (50.0 * upc)

    world_up   = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_fwd)
    head_right /= np.linalg.norm(head_right)
    head_up    = np.cross(head_fwd, head_right)
    head_up    /= np.linalg.norm(head_up)

    p0 = center_w - head_right * half_w - head_up * half_h  # TL
    p1 = center_w + head_right * half_w - head_up * half_h  # TR
    p2 = center_w + head_right * half_w + head_up * half_h  # BR
    p3 = center_w - head_right * half_w + head_up * half_h  # BL

    normal_w = head_fwd / np.linalg.norm(head_fwd)
    return [p0, p1, p2, p3], center_w, normal_w, upc


# ─────────────────────────────────────────────────────────────────────────────
#  RAY–PLANE INTERSECTION
# ─────────────────────────────────────────────────────────────────────────────

def ray_plane_ab(O, D, corners, center, normal):
    """
    Intersect ray  P(t) = O + t·D  with the monitor quad.

    Returns
    -------
    (a, b)  : normalised quad coordinates  (may exceed [0,1])
              a=0 left, a=1 right, b=0 top, b=1 bottom
    None    : ray is parallel or intersection is behind the eye
    """
    N = normalize(normal)
    d = float(np.dot(N, D))
    if abs(d) < 1e-6:
        return None
    t = float(np.dot(N, np.asarray(center) - O) / d)
    if t < 0.0:
        return None
    P = O + t * D

    p0, p1, _, p3 = [np.asarray(p, dtype=float) for p in corners]
    u  = p1 - p0
    v  = p3 - p0
    u2 = float(np.dot(u, u))
    v2 = float(np.dot(v, v))
    if u2 < 1e-9 or v2 < 1e-9:
        return None

    wv = P - p0
    return float(np.dot(wv, u) / u2), float(np.dot(wv, v) / v2)


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationManager:
    """
    Persists sphere offsets + homography to calibration.json.

    What is saved
    -------------
    left_offset / right_offset  — eyeball sphere position in head-local frame
    left_scale  / right_scale   — nose scale at calibration time
    homography                  — 3×3 matrix from 4-point gaze calibration

    What is NOT saved
    -----------------
    Monitor plane corners — these depend on current head position.
    Press C at session start to re-anchor the plane (< 5 seconds).
    """

    def save(self, left_offset, right_offset,
             left_scale, right_scale, H) -> None:
        data = {
            'left_offset':  left_offset.tolist(),
            'right_offset': right_offset.tolist(),
            'left_scale':   float(left_scale),
            'right_scale':  float(right_scale),
            'homography':   H.tolist() if H is not None else None,
        }
        with open(CALIB_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[CalibrationManager] Saved → {CALIB_FILE}")

    def load(self) -> dict | None:
        if not os.path.exists(CALIB_FILE):
            return None
        try:
            with open(CALIB_FILE) as f:
                d = json.load(f)
            return {
                'left_offset':  np.array(d['left_offset']),
                'right_offset': np.array(d['right_offset']),
                'left_scale':   float(d['left_scale']),
                'right_scale':  float(d['right_scale']),
                'homography':   (np.array(d['homography'])
                                 if d['homography'] is not None else None),
            }
        except Exception as e:
            print(f"[CalibrationManager] Load failed: {e}")
            return None

    def exists(self) -> bool:
        return os.path.exists(CALIB_FILE)


# ─────────────────────────────────────────────────────────────────────────────
#  GAZE DATA COLLECTOR  (training data for future personal ML model)
# ─────────────────────────────────────────────────────────────────────────────

class GazeDataCollector:
    """
    Silently logs one CSV row per confirmed key activation.

    Columns
    -------
    ts                    : Unix timestamp
    il_x, il_y, il_z      : left iris 3D position (camera space)
    ir_x, ir_y, ir_z      : right iris 3D position
    roll, pitch, yaw      : head Euler angles (degrees)
    a_raw, b_raw          : uncorrected gaze position on monitor plane
    key_row, key_col, key : ground-truth label

    Usage (future)
    --------------
    Run  scripts/train_gaze_model.py  once enough samples are collected
    (~500+) to train a personal MLP that replaces the geometry pipeline.
    """

    HEADER = [
        'ts',
        'il_x', 'il_y', 'il_z',
        'ir_x', 'ir_y', 'ir_z',
        'roll', 'pitch', 'yaw',
        'a_raw', 'b_raw',
        'key_row', 'key_col', 'key'
    ]

    def __init__(self):
        write_header = not os.path.exists(GAZE_DATA_FILE)
        self._f = open(GAZE_DATA_FILE, 'a', newline='')
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(self.HEADER)
        # Count existing rows
        self.count = 0
        if not write_header:
            try:
                with open(GAZE_DATA_FILE) as tmp:
                    self.count = max(0, sum(1 for _ in tmp) - 1)
            except Exception:
                pass

    def log(self, iris_l, iris_r, R_final,
            a_raw: float, b_raw: float,
            key_row: int, key_col: int, key: str) -> None:
        try:
            r  = Rscipy.from_matrix(R_final)
            roll, pitch, yaw = r.as_euler('zyx', degrees=True)
            self._w.writerow([
                round(float(__import__('time').time()), 4),
                *[round(float(v), 4) for v in iris_l],
                *[round(float(v), 4) for v in iris_r],
                round(roll,  3), round(pitch, 3), round(yaw, 3),
                round(a_raw, 5), round(b_raw, 5),
                key_row, key_col, key,
            ])
            self._f.flush()
            self.count += 1
        except Exception as e:
            print(f"[GazeDataCollector] {e}")

    def close(self) -> None:
        self._f.close()
