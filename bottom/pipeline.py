"""A 영상의 칩 위치를 찾고 6개 ROI에서 원 후보를 검사한다."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from general import load_ab_tiff_pages, to_bgr, to_grayscale
from .circles import CircleRegion, create_circle_roi_preview, draw_circle_regions, find_circle_regions


# 현재 Bottom 영상은 검은 배경에 사각 칩 한 개가 놓인 구성을 전제로 한다.
MIN_CHIP_AREA_RATIO = 0.03
MAX_CHIP_AREA_RATIO = 0.80
MIN_CHIP_ASPECT_RATIO = 0.65
MIN_CHIP_RECTANGULARITY = 0.80
MIN_BACKGROUND_CONTRAST = 20.0
CHIP_CENTER_COLOR = (255, 145, 0)  # BGR, 파란 중심 표시


@dataclass
class ChipLocation:
    """원본 A 좌표계의 칩 외곽, 회전 사각형과 위치 정보."""

    contour: np.ndarray
    corners: np.ndarray  # 좌상, 우상, 우하, 좌하 순서
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle_degrees: float
    bounding_rect: Tuple[int, int, int, int]
    area: float


@dataclass
class BottomDetectionResult:
    """칩 위치와 원 후보 결과. 원 후보 통과는 양품 판정을 뜻하지 않는다."""

    raw_image_a: np.ndarray
    raw_image_b: np.ndarray
    chip: Optional[ChipLocation]
    source_visualization: np.ndarray
    chip_mask: np.ndarray
    threshold: float
    chip_crop: Optional[np.ndarray] = None
    circle_regions: List[CircleRegion] = field(default_factory=list)
    circle_preview: Optional[np.ndarray] = None
    roi_preview: Optional[np.ndarray] = None

    @property
    def is_detected(self):
        return self.chip is not None

    @property
    def candidate_count(self):
        return sum(region.status == "candidate" for region in self.circle_regions)

    @property
    def review_count(self):
        return sum(region.status == "review" for region in self.circle_regions)

    @property
    def missing_count(self):
        return sum(region.status == "missing" for region in self.circle_regions)

    @property
    def all_regions_accepted(self):
        return len(self.circle_regions) == 6 and self.candidate_count == 6


def _ordered_corners(rect):
    """OpenCV 버전별 회전각 표현과 무관하게 네 꼭짓점의 순서를 고정한다."""
    corners = cv2.boxPoints(rect)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    return np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0)


def _binarize_a(image_a):
    """배경 밝기와 Otsu 값을 이용해 회색 기판까지 포함하는 이진 영상을 만든다."""
    if (
        image_a.dtype != np.uint8
        or image_a.ndim not in (2, 3)
        or (image_a.ndim == 3 and image_a.shape[2] not in (3, 4))
        or min(image_a.shape[:2]) < 3
    ):
        raise ValueError("Bottom 검출에는 8비트 A 영상이 필요합니다.")

    gray = to_grayscale(image_a)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate((blurred[0], blurred[-1], blurred[:, 0], blurred[:, -1]))
    background = float(np.median(border))
    otsu_threshold, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(
        background + MIN_BACKGROUND_CONTRAST, (background + otsu_threshold) / 2
    )
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return binary, threshold


def _locate_bottom_chip(binary):
    """칩 위치 추정용 복사본에만 Closing과 내부 채움을 적용한다."""
    height, width = binary.shape
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    image_area = height * width
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not MIN_CHIP_AREA_RATIO * image_area <= area <= MAX_CHIP_AREA_RATIO * image_area:
            continue
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        rect_area = rect_width * rect_height
        if rect_area <= 0:
            continue
        if min(rect_width, rect_height) / max(rect_width, rect_height) < MIN_CHIP_ASPECT_RATIO:
            continue
        if area / rect_area < MIN_CHIP_RECTANGULARITY:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        # 화면에 잘린 칩은 완전한 외곽 위치로 승인하지 않는다.
        if x <= 0 or y <= 0 or x + box_width >= width or y + box_height >= height:
            continue
        candidates.append((area, contour, rect, (x, y, box_width, box_height)))

    mask = np.zeros_like(binary)
    if not candidates:
        return None, mask

    area, contour, rect, bounds = max(candidates, key=lambda candidate: candidate[0])
    corners = _ordered_corners(rect)
    top_edge = corners[1] - corners[0]
    left_edge = corners[3] - corners[0]
    chip = ChipLocation(
        contour=contour,
        corners=corners,
        center=tuple(float(value) for value in rect[0]),
        size=(float(np.linalg.norm(top_edge)), float(np.linalg.norm(left_edge))),
        angle_degrees=float(np.degrees(np.arctan2(top_edge[1], top_edge[0]))),
        bounding_rect=bounds,
        area=area,
    )
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return chip, mask


def find_bottom_chip(image_a):
    """기존 호출과 동일하게 (칩 위치 또는 None, 외곽 마스크, 임계값)을 반환한다."""
    binary, threshold = _binarize_a(image_a)
    chip, mask = _locate_bottom_chip(binary)
    return chip, mask, threshold


def run_bottom_detection(image_path):
    """이진화 → 칩 기준 ROI → 검은 성분 → 원 후보 조건 검사. B는 참조용이다."""
    image_a, image_b = load_ab_tiff_pages(image_path)
    binary, threshold = _binarize_a(image_a)
    chip, mask = _locate_bottom_chip(binary)
    regions = find_circle_regions(binary, chip, mask)
    visualization = draw_circle_regions(image_a, regions)
    crop = None
    circle_preview = None
    if chip is not None:
        cv2.drawMarker(
            visualization, tuple(int(round(value)) for value in chip.center),
            CHIP_CENTER_COLOR, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA,
        )
        x, y, width, height = chip.bounding_rect
        crop = to_bgr(image_a[y:y + height, x:x + width])
        circle_preview = visualization[y:y + height, x:x + width].copy()
    return BottomDetectionResult(
        raw_image_a=image_a, raw_image_b=image_b, chip=chip,
        source_visualization=visualization, chip_mask=mask,
        threshold=threshold, chip_crop=crop, circle_regions=regions,
        circle_preview=circle_preview, roi_preview=create_circle_roi_preview(image_a, regions),
    )
