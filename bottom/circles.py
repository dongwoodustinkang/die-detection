"""칩 기준 6개 ROI에서 검은 성분의 실제 윤곽과 원 후보 조건을 검사한다."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from general import to_bgr


ROI_SIZE = 40.0
MIN_CIRCLE_AREA = 300.0
MAX_CIRCLE_AREA = 450.0
MIN_CIRCULARITY = 0.70
MIN_CIRCLE_ASPECT_RATIO = 0.70
MAX_CENTER_DISTANCE = 12.0
# 작은 픽셀 잡음만 제외한다. 조건에 미달한 큰 성분은 검토용으로 남긴다.
MIN_COMPONENT_AREA = 10.0
ROI_POSITIONS = (
    ("왼쪽 위", 0.1, 0.1), ("오른쪽 위", 0.9, 0.1),
    ("왼쪽 중간", 0.1, 0.5), ("오른쪽 중간", 0.9, 0.5),
    ("왼쪽 아래", 0.1, 0.9), ("오른쪽 아래", 0.9, 0.9),
)
ROI_COLOR = (230, 180, 20)
CANDIDATE_COLOR = (115, 165, 0)
REVIEW_COLOR = (25, 115, 240)
MISSING_COLOR = (65, 65, 220)


@dataclass
class CircleComponent:
    contour: np.ndarray
    center: Tuple[float, float]
    area: float
    circularity: float
    aspect_ratio: float
    center_distance: float
    rejection_reasons: Tuple[str, ...]

    @property
    def is_candidate(self):
        return not self.rejection_reasons


@dataclass
class CircleRegion:
    index: int
    name: str
    expected_center: Tuple[float, float]
    corners: np.ndarray
    components: List[CircleComponent] = field(default_factory=list)
    primary: Optional[CircleComponent] = None

    @property
    def status(self):
        if self.primary is None:
            return "missing"
        accepted = sum(component.is_candidate for component in self.components)
        return "candidate" if accepted == 1 else "review"

    @property
    def review_reasons(self):
        if self.primary is None:
            return ("유효한 검은 성분 없음",)
        if sum(component.is_candidate for component in self.components) > 1:
            return ("조건을 통과한 성분이 여러 개임",)
        return self.primary.rejection_reasons


def _measure_component(contour, expected_center, search_boundary):
    area = float(cv2.contourArea(contour))
    if area < MIN_COMPONENT_AREA:
        return None
    moments = cv2.moments(contour)
    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    perimeter = cv2.arcLength(contour, True)
    circularity = float(4 * np.pi * area / perimeter ** 2) if perimeter else 0.0
    _, _, width, height = cv2.boundingRect(contour)
    aspect = min(width, height) / max(width, height)
    distance = float(np.linalg.norm(np.asarray(center) - expected_center))
    points = contour[:, 0]
    touches_boundary = bool(np.any(search_boundary[points[:, 1], points[:, 0]]))
    reasons = []
    if not MIN_CIRCLE_AREA <= area <= MAX_CIRCLE_AREA:
        reasons.append(f"면적 범위 밖 ({MIN_CIRCLE_AREA:g}–{MAX_CIRCLE_AREA:g} px²)")
    if circularity < MIN_CIRCULARITY:
        reasons.append(f"원형도 부족 (< {MIN_CIRCULARITY:.2f})")
    if aspect < MIN_CIRCLE_ASPECT_RATIO:
        reasons.append(f"가로세로 비율 부족 (< {MIN_CIRCLE_ASPECT_RATIO:.2f})")
    if distance > MAX_CENTER_DISTANCE:
        reasons.append(f"예상 중심에서 벗어남 (> {MAX_CENTER_DISTANCE:g} px)")
    if touches_boundary:
        reasons.append("검색 경계에 닿아 윤곽 확인 필요")
    return CircleComponent(contour, center, area, circularity, aspect, distance, tuple(reasons))


def find_circle_regions(binary, chip, chip_mask):
    """Closing/내부 채움을 하지 않은 이진 영상에서 원 후보를 추출한다.

    칩 마스크는 외부 배경을 배제하는 데만 사용한다. 각 ROI의 중심은
    칩의 두 변을 따라 계산하며, 고정된 영상 좌표나 Hough 원 피팅을 쓰지 않는다.
    """
    if chip is None:
        return []
    top_edge = chip.corners[1] - chip.corners[0]
    left_edge = chip.corners[3] - chip.corners[0]
    half_x = top_edge / np.linalg.norm(top_edge) * ROI_SIZE / 2
    half_y = left_edge / np.linalg.norm(left_edge) * ROI_SIZE / 2
    regions = []
    for index, (name, fx, fy) in enumerate(ROI_POSITIONS, 1):
        center = chip.corners[0] + fx * top_edge + fy * left_edge
        corners = np.asarray((
            center - half_x - half_y, center + half_x - half_y,
            center + half_x + half_y, center - half_x + half_y,
        ), dtype=np.float32)
        roi = np.zeros_like(binary)
        cv2.fillConvexPoly(roi, np.rint(corners).astype(np.int32), 255)
        search_mask = cv2.bitwise_and(roi, chip_mask)
        eroded = cv2.erode(search_mask, np.ones((3, 3), dtype=np.uint8))
        search_boundary = (search_mask > 0) & (eroded == 0)
        dark = np.where((search_mask > 0) & (binary == 0), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        components = []
        for contour in contours:
            component = _measure_component(contour, center, search_boundary)
            if component is not None:
                components.append(component)
        # 적합한 후보를 우선하되, 탈락한 성분도 원본 윤곽과 사유를 유지한다.
        components.sort(key=lambda c: (not c.is_candidate, c.center_distance, -c.area))
        regions.append(CircleRegion(
            index, name, tuple(float(value) for value in center), corners,
            components, components[0] if components else None,
        ))
    return regions


def _region_color(region):
    return {"candidate": CANDIDATE_COLOR, "review": REVIEW_COLOR, "missing": MISSING_COLOR}[region.status]


def draw_circle_regions(image, regions):
    """ROI와 실제 윤곽만 그린다. 칩 외곽선이나 이상적인 원으로 덮지 않는다."""
    result = to_bgr(image)
    for region in regions:
        corners = np.rint(region.corners).astype(np.int32)
        color = _region_color(region)
        cv2.polylines(result, [corners], True, ROI_COLOR, 1, cv2.LINE_AA)
        for component in region.components:
            contour_color = color if component is region.primary else REVIEW_COLOR
            cv2.drawContours(result, [component.contour], -1, contour_color, 1, cv2.LINE_AA)
        x, y = corners[0]
        cv2.putText(result, str(region.index), (max(0, x + 2), max(9, y + 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    return result


def create_circle_roi_preview(image_a, regions):
    """폭을 활용한 3열 × 2행 상세 보기로 여섯 ROI와 실제 윤곽을 확대한다."""
    if not regions:
        return None
    tile_size, cell_width, cell_height = 128, 152, 184
    preview = np.full((cell_height * 2, cell_width * 3, 3), 245, dtype=np.uint8)
    target = np.asarray(((0, 0), (tile_size, 0), (tile_size, tile_size), (0, tile_size)), np.float32)
    source = to_bgr(image_a)
    for region in regions:
        transform = cv2.getPerspectiveTransform(region.corners, target)
        tile = cv2.warpPerspective(source, transform, (tile_size, tile_size), flags=cv2.INTER_NEAREST)
        color = _region_color(region)
        for component in region.components:
            points = cv2.perspectiveTransform(component.contour.astype(np.float32), transform)
            cv2.polylines(tile, [np.rint(points).astype(np.int32)], True,
                          color if component is region.primary else REVIEW_COLOR, 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), color, 2)
        x = ((region.index - 1) % 3) * cell_width + 12
        y = ((region.index - 1) // 3) * cell_height + 8
        preview[y:y + tile_size, x:x + tile_size] = tile
        component = region.primary
        captions = (f'{region.index}  C={component.circularity:.2f}', f'A={component.area:g} px2') if component else (f'{region.index}  --', '')
        for line, caption in enumerate(captions):
            cv2.putText(preview, caption, (x, y + tile_size + 18 + line * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    return preview
