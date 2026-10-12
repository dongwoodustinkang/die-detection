"""측면 검사 알고리즘 패키지."""

from .pipeline import (
    DetectionResult,
    SideDetectionResult,
    create_detection_visualization,
    extract_surface_tiles,
    run_side_detection,
)
from .surface_crop import SURFACE_CROP_VERSION, SurfaceTiles

__all__ = [
    "SideDetectionResult",
    "run_side_detection",
    "DetectionResult",
    "create_detection_visualization",
    "extract_surface_tiles",
    "SurfaceTiles",
    "SURFACE_CROP_VERSION",
]
