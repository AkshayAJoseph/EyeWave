"""
EyeWave/src
===========
Public package API.  Import from here for clean usage:

    from src import EyeKeyboard, LayoutManager
    from src import AdaptiveGazeFilter, FixationDetector
"""

from src.interface import EyeKeyboard, LayoutManager
from src.visionc  import (
    AdaptiveGazeFilter,
    FixationDetector,
    SmartDwellController,
    MultiPointCalib,
    BlinkDetector,
    ScanningController,
    render_debug_view_orbit,
)
from src.utils import (
    CalibrationManager,
    GazeDataCollector,
    normalize,
    compute_scale,
    pca_orientation,
    create_monitor_plane,
    ray_plane_ab,
)

__all__ = [
    # Interface
    "EyeKeyboard", "LayoutManager",
    # Vision pipeline
    "AdaptiveGazeFilter", "FixationDetector", "SmartDwellController",
    "MultiPointCalib", "BlinkDetector", "ScanningController",
    "render_debug_view_orbit",
    # Utils
    "CalibrationManager", "GazeDataCollector",
    "normalize", "compute_scale", "pca_orientation",
    "create_monitor_plane", "ray_plane_ab",
]
