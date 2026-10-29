"""Side 표면 Crop 타일을 학습 데이터 생성과 추론에서 같은 방식으로 만든다."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from general import to_bgr, to_grayscale

from .surface import crop_polygon_tile, get_polygon_crop_bounds


# Crop 방식이 바뀌면 올려서, 저장된 학습 데이터와 추론 입력이 어긋나지 않았는지 확인한다.
SURFACE_CROP_VERSION = "1"


@dataclass
class SurfaceTiles:
    """A 밴드 컨투어로 자른 A/B 표면 타일."""

    tile_a: np.ndarray  # BGRA, 컨투어 바깥은 alpha 0
    tile_b: np.ndarray  # BGRA, tile_a와 같은 크기·영역
    bounds: Tuple[int, int, int, int]  # 원본 기준 (left, top, right, bottom)

    @property
    def gray_a(self):
        """A 타일의 그레이스케일 픽셀(컨투어 바깥 픽셀도 원본 그대로)."""
        return to_grayscale(self.tile_a)

    @property
    def gray_b(self):
        """B 타일의 그레이스케일 픽셀(컨투어 바깥 픽셀도 원본 그대로)."""
        return to_grayscale(self.tile_b)


def create_surface_tiles(image_a, image_b, page_a_contour) -> Optional[SurfaceTiles]:
    """A 밴드 컨투어 하나로 A/B 페이지를 같은 영역만큼 자른다. 실패하면 None."""
    if page_a_contour is None:
        return None
    bounds = get_polygon_crop_bounds(page_a_contour, image_a.shape[:2])
    tile_a = crop_polygon_tile(to_bgr(image_a), page_a_contour)
    tile_b = crop_polygon_tile(image_b, page_a_contour)
    if bounds is None or tile_a is None or tile_b is None:
        return None
    return SurfaceTiles(tile_a=tile_a, tile_b=tile_b, bounds=bounds)
