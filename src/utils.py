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
#  3D Head Orientation & Gaze Helpers (from MonitorTracking.py)
# ════════════════════════════════════════════════════════════════════════

def compute_pca_orientation(pts, indices, w, h, ref_matrix):
    """PCA-based head orientation from nose-region landmarks.

    Computes a stable 3D rotation matrix using eigendecomposition of the
    covariance matrix of selected landmarks.  A reference matrix is used
    to stabilise eigenvector sign flips between frames.

    Args:
        pts:        (N, 2) pixel landmarks from MediaPipe.
        indices:    List of landmark indices to use (nose region).
        w, h:       Frame width/height.
        ref_matrix: Single-element list [R_ref | None] — mutated in-place.

    Returns:
        (center_3d, R_final, points_3d)
        - center_3d: (3,) mean position of selected landmarks.
        - R_final:   (3,3) stabilised rotation matrix.
        - points_3d: (K,3) 3D positions of all selected landmarks.
    """
    from scipy.spatial.transform import Rotation as Rscipy

    # Extract 3D positions (MediaPipe provides pseudo-depth via z * w)
    # Need the original landmarks object for z — we store z in pts_3d
    # But pts only has (x,y). We need the caller to provide 3D points.
    # This function expects pts to have already been converted differently.
    # We'll handle this via a separate 3D points array passed from visionc.
    raise NotImplementedError("use compute_pca_orientation_3d instead")


def compute_pca_orientation_3d(points_3d, ref_matrix):
    """PCA-based head orientation from pre-extracted 3D nose landmarks.

    Args:
        points_3d:  (K, 3) array of 3D landmark positions.
        ref_matrix: Single-element list [R_ref | None] — mutated in-place
                    to stabilise eigenvector directions across frames.

    Returns:
        (center_3d, R_final)
        - center_3d: (3,) mean position.
        - R_final:   (3,3) stabilised rotation matrix.
    """
    from scipy.spatial.transform import Rotation as Rscipy

    center = np.mean(points_3d, axis=0)

    # PCA via covariance eigendecomposition
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # major axes first

    # Ensure right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Round-trip through Euler angles (keeps consistent axis convention)
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # Stabilise against eigenvector sign flips using reference matrix
    if ref_matrix[0] is None:
        ref_matrix[0] = R_final.copy()
    else:
        R_ref = ref_matrix[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    return center, R_final


def lock_eye_spheres(head_center, R_final, iris_3d_left, iris_3d_right,
                     nose_points_3d, base_radius=20):
    """Lock eye sphere positions in head-local coordinates.

    During calibration, store the iris position relative to the head center
    in the head's local coordinate system, offset by base_radius along the
    camera direction.  This allows reconstruction at any head pose / distance.

    Args:
        head_center:     (3,) 3D head center from PCA.
        R_final:         (3,3) head rotation matrix.
        iris_3d_left:    (3,) 3D position of left iris center.
        iris_3d_right:   (3,) 3D position of right iris center.
        nose_points_3d:  (K,3) nose landmark positions (for scale).
        base_radius:     Sphere radius at calibration distance.

    Returns:
        (left_offset, right_offset, calib_nose_scale)
    """
    camera_dir_world = np.array([0, 0, 1], dtype=float)
    camera_dir_local = R_final.T @ camera_dir_world

    left_offset = R_final.T @ (iris_3d_left - head_center)
    left_offset += base_radius * camera_dir_local

    right_offset = R_final.T @ (iris_3d_right - head_center)
    right_offset += base_radius * camera_dir_local

    calib_scale = compute_nose_scale(nose_points_3d)
    return left_offset, right_offset, calib_scale


def compute_sphere_positions(head_center, R_final, left_offset, right_offset,
                             nose_points_3d, calib_nose_scale, base_radius=20):
    """Reconstruct eye sphere world positions from stored offsets.

    Scales the offsets by the ratio of current-to-calibration nose landmark
    distances to compensate for the user moving closer/farther.

    Returns:
        (sphere_l, sphere_r, scaled_radius)
    """
    current_scale = compute_nose_scale(nose_points_3d)
    scale_ratio = current_scale / calib_nose_scale if calib_nose_scale > 0 else 1.0

    sphere_l = head_center + R_final @ (left_offset * scale_ratio)
    sphere_r = head_center + R_final @ (right_offset * scale_ratio)
    scaled_radius = int(base_radius * scale_ratio)

    return sphere_l, sphere_r, scaled_radius


def compute_binocular_gaze(sphere_l, sphere_r, iris_3d_left, iris_3d_right):
    """Compute combined gaze direction from both eyes.

    Gaze direction = iris_3d - sphere_center (from eye center toward iris).

    Returns:
        Normalized (3,) gaze direction vector, or None if degenerate.
    """
    left_dir = iris_3d_left - sphere_l
    right_dir = iris_3d_right - sphere_r

    ln = np.linalg.norm(left_dir)
    rn = np.linalg.norm(right_dir)

    if ln < 1e-9 and rn < 1e-9:
        return None

    parts = []
    if ln > 1e-9:
        parts.append(left_dir / ln)
    if rn > 1e-9:
        parts.append(right_dir / rn)

    combined = np.mean(parts, axis=0)
    n = np.linalg.norm(combined)
    if n < 1e-9:
        return None
    return combined / n


def gaze_to_screen(gaze_dir, yaw_range=15.0, pitch_range=5.0):
    """Direct angular mapping from 3D gaze direction to screen coordinates.

    Ported directly from MonitorTracking.py's convert_gaze_to_screen_coordinates.

    Args:
        gaze_dir:    Normalized (3,) gaze direction vector.
        yaw_range:   Degrees at which the gaze reaches screen edges horizontally.
        pitch_range: Degrees at which the gaze reaches screen edges vertically.

    Returns:
        (norm_x, norm_y) in 0–1 range (clamped).
    """
    reference_forward = np.array([0, 0, -1], dtype=float)
    avg_dir = gaze_dir / np.linalg.norm(gaze_dir)

    # Horizontal (yaw) angle from reference Z-axis
    xz_proj = np.array([avg_dir[0], 0, avg_dir[2]], dtype=float)
    xz_n = np.linalg.norm(xz_proj)
    if xz_n < 1e-9:
        yaw_deg = 0.0
    else:
        xz_proj /= xz_n
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_dir[0] < 0:
            yaw_rad = -yaw_rad
        yaw_deg = float(np.degrees(yaw_rad))

    # Vertical (pitch) angle from reference Z-axis
    yz_proj = np.array([0, avg_dir[1], avg_dir[2]], dtype=float)
    yz_n = np.linalg.norm(yz_proj)
    if yz_n < 1e-9:
        pitch_deg = 0.0
    else:
        yz_proj /= yz_n
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_dir[1] > 0:
            pitch_rad = -pitch_rad
        pitch_deg = float(np.degrees(pitch_rad))

    # MonitorTracking sign convention for yaw
    yaw_deg = -yaw_deg

    # Map to normalised 0–1 screen coordinates
    norm_x = (yaw_deg + yaw_range) / (2 * yaw_range)
    norm_y = (pitch_range - pitch_deg) / (2 * pitch_range)

    norm_x = float(np.clip(norm_x, 0, 1))
    norm_y = float(np.clip(norm_y, 0, 1))

    return norm_x, norm_y


def compute_nose_scale(points_3d):
    """Average pairwise distance of landmark points — used for scale estimation.

    From MonitorTracking.py's compute_scale.
    """
    n = len(points_3d)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(points_3d[i] - points_3d[j]))
            count += 1
    return total / count if count > 0 else 1.0


