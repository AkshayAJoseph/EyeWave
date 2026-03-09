# EyeWave Utility Functions

import math
import time
import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════════
#  One Euro Filter — adaptive low-pass for noisy gaze input
# ════════════════════════════════════════════════════════════════════════

class LowPassFilter:
    """Simple first-order low-pass filter."""

    def __init__(self, alpha=1.0):
        self._y = None
        self._alpha = alpha

    def filter(self, value, alpha=None):
        if alpha is not None:
            self._alpha = alpha
        if self._y is None:
            self._y = value
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._y
        return self._y

    def reset(self):
        self._y = None


class OneEuroFilter:
    """One Euro Filter for adaptive noise reduction.

    During fixations (slow movement) it smooths heavily to eliminate jitter.
    During saccades (fast movement) it responds quickly with minimal lag.

    Args:
        freq:       Sampling frequency in Hz (e.g. 30 for 30fps).
        min_cutoff: Minimum cutoff frequency — lower = more smoothing during fixation.
        beta:       Speed coefficient — higher = less lag during fast movement.
        d_cutoff:   Cutoff for derivative filtering (usually 1.0).
    """

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._last_time = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp=None):
        if self._last_time is not None and timestamp is not None:
            dt = timestamp - self._last_time
            if dt > 0:
                self.freq = 1.0 / dt
        self._last_time = timestamp

        # Estimate derivative
        prev = self._x._y
        if prev is None:
            dx = 0.0
        else:
            dx = (x - prev) * self.freq

        # Filter the derivative
        edx = self._dx.filter(dx, self._alpha(self.d_cutoff, self.freq))

        # Adaptive cutoff: when speed is high, cutoff increases → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(edx)

        return self._x.filter(x, self._alpha(cutoff, self.freq))

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last_time = None


# ════════════════════════════════════════════════════════════════════════
#  Simple helpers
# ════════════════════════════════════════════════════════════════════════

def ema_smooth(current: float, previous: float, alpha: float) -> float:
    """Exponential Moving Average smoothing (legacy, use OneEuroFilter instead)."""
    return alpha * current + (1.0 - alpha) * previous


def normalize_to_range(value: float, src_min: float, src_max: float,
                       dst_min: float = 0.0, dst_max: float = 1.0) -> float:
    """Linearly map *value* from [src_min, src_max] to [dst_min, dst_max], clamped."""
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    normalized = (value - src_min) / (src_max - src_min)
    scaled = dst_min + normalized * (dst_max - dst_min)
    return float(np.clip(scaled, dst_min, dst_max))


def eye_aspect_ratio(top, bottom, left, right) -> float:
    """Compute Eye Aspect Ratio (EAR) from landmark pixel coordinates.

    Each argument is an (x, y) array-like.
    """
    hor = math.hypot(left[0] - right[0], left[1] - right[1])
    ver = math.hypot(top[0] - bottom[0], top[1] - bottom[1])
    return ver / hor if hor > 0 else 0.0


# ════════════════════════════════════════════════════════════════════════
#  3D Head Pose & Gaze Ray Helpers
# ════════════════════════════════════════════════════════════════════════

def estimate_3d_head_pose(pts, frame_w, frame_h, face_3d_model, focal_factor=1.0):
    """Estimate 3D head pose via cv2.solvePnP.

    Args:
        pts:            (N, 2) pixel landmarks from MediaPipe.
        frame_w:        Frame width in pixels.
        frame_h:        Frame height in pixels.
        face_3d_model:  (6, 3) canonical 3D face points (mm).
        focal_factor:   Focal length ≈ focal_factor * frame_w.

    Returns:
        (rvec, tvec, euler_angles, success)
        - rvec: (3,1) Rodrigues rotation vector.
        - tvec: (3,1) translation vector.
        - euler_angles: (yaw, pitch, roll) in degrees.
        - success: True if solvePnP succeeded.
    """
    from .config import HEAD_POSE_LANDMARKS

    # 2D image points for the 6 key landmarks
    image_pts = np.array([pts[i] for i in HEAD_POSE_LANDMARKS], dtype=np.float64)

    # Approximate camera intrinsics (no lens distortion assumed)
    focal_length = frame_w * focal_factor
    cx, cy = frame_w / 2.0, frame_h / 2.0
    camera_matrix = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        face_3d_model, image_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None, None, (0.0, 0.0, 0.0), False

    # Convert rotation vector → rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix to Euler angles (degrees)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
    yaw   = float(euler[1, 0])
    pitch = float(euler[0, 0])
    roll  = float(euler[2, 0])

    return rvec, tvec, (yaw, pitch, roll), True


def compute_gaze_ray_intersection(rvec, tvec, iris_2d_left, iris_2d_right,
                                  frame_w, frame_h, focal_factor=1.0):
    """Compute where the gaze ray intersects the screen plane (z = 0).

    Uses the 3D head pose to estimate gaze direction, blending the
    head-forward direction with the iris offset.

    Args:
        rvec, tvec:      Rotation/translation from solvePnP.
        iris_2d_left:    (x, y) pixel position of left iris center.
        iris_2d_right:   (x, y) pixel position of right iris center.
        frame_w, frame_h: Frame dimensions.
        focal_factor:    Same factor used for the camera matrix.

    Returns:
        (gaze_x, gaze_y) in normalized 0-1 screen coordinates,
        or (0.5, 0.5) on failure.
    """
    try:
        focal_length = frame_w * focal_factor
        cx, cy = frame_w / 2.0, frame_h / 2.0

        # Average iris center in pixels
        iris_x = (iris_2d_left[0] + iris_2d_right[0]) / 2.0
        iris_y = (iris_2d_left[1] + iris_2d_right[1]) / 2.0

        # Ray from camera through the iris in camera coordinates
        ray_dir = np.array([
            (iris_x - cx) / focal_length,
            (iris_y - cy) / focal_length,
            1.0,
        ], dtype=np.float64)
        ray_dir /= np.linalg.norm(ray_dir)

        # Head forward direction from rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        head_forward = rmat[:, 2]  # z-axis of the head in camera space

        # Blend: 70% iris ray (captures eye-in-head) + 30% head forward
        # (captures overall head orientation)
        gaze_dir = 0.7 * ray_dir + 0.3 * head_forward
        gaze_dir /= np.linalg.norm(gaze_dir)

        # Intersect the gaze ray with the z = tvec[2] plane
        # (approximate screen plane at the head's depth)
        if abs(gaze_dir[2]) < 1e-6:
            return 0.5, 0.5

        # Origin of gaze ≈ midpoint between the two iris centers in 3D
        # (projected back from 2D)
        t = float(tvec[2, 0]) / gaze_dir[2]
        hit_x = gaze_dir[0] * t
        hit_y = gaze_dir[1] * t

        # Normalize to 0-1 using the frame dimensions as the viewport
        norm_x = float(np.clip(0.5 + hit_x / frame_w, 0, 1))
        norm_y = float(np.clip(0.5 + hit_y / frame_h, 0, 1))

        return norm_x, norm_y
    except Exception:
        return 0.5, 0.5

