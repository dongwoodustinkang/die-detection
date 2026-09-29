"""A 영상의 칩 위치를 찾고 6개 ROI에서 원 후보를 검사한다."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from general import load_ab_tiff_pages, to_bgr, to_grayscale
from .circles import CircleRegion, create_circle_roi_preview, draw_circle_regions, find_circle_regions


# 현재 Bottom 영상은 검은 배경에 사각 칩 한 개가 놓인 구성을 전제로 한다.
MIN_CHIP_AREA_RATIO = 0.03 # 3%
MAX_CHIP_AREA_RATIO = 0.80 # 80%
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
    chip: Optional[ChipLocation] # 칩 위치 정보
    source_visualization: np.ndarray 
    chip_mask: np.ndarray # 칩 내부 255 영역 마스크
    threshold: float # 이진화시 밝기 임계값
    chip_crop: Optional[np.ndarray] = None # 원 검출을 위한 크롭된 이미지
    circle_regions: List[CircleRegion] = field(default_factory=list) # 원 검출 결과 리스트
    circle_preview: Optional[np.ndarray] = None # 원 검출 프리뷰 이미지
    roi_preview: Optional[np.ndarray] = None # ROI 프리뷰 이미지

    @property
    def is_detected(self):
        # 칩 검출 성공 여부
        return self.chip is not None

    @property
    def candidate_count(self):
        # 6개의 ROI 중 후보 조건을 통과한 개수
        return sum(region.status == "candidate" for region in self.circle_regions)

    @property
    def review_count(self):
        # 조건 미달 계수
        return sum(region.status == "review" for region in self.circle_regions)

    @property
    def missing_count(self):
        # 미검출 영역 개수
        return sum(region.status == "missing" for region in self.circle_regions)

    @property
    def all_regions_accepted(self):
        # 6개 ROI가 모두 후보 조건을 통과했는지
        return len(self.circle_regions) == 6 and self.candidate_count == 6


def _ordered_corners(rect):
    """OpenCV 버전별 회전각 표현과 무관하게 네 꼭짓점의 순서를 고정한다."""
    corners = cv2.boxPoints(rect) # 회전된 사각형의 4개 꼭짓점 2D 좌표
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]) # 회전각 계산
    corners = corners[np.argsort(angles)] # 계산한 회전각을 기준으로 정렬
    return np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0) # 정렬된 꼭짓점을 반환


def _binarize(image):
    """회색 기판만 남기는 이진화 작업 수행"""

    # 입력된 이미지가 적합한 규격인지 검사
    if (
        image.dtype != np.uint8 # 픽셀당 8비트 정수형인지(0~255)
        or image.ndim not in (2, 3) # 2채널 또는 3채널 이미지인지
        or (image.ndim == 3 and image.shape[2] not in (3, 4)) # 3채널이면서 3채널 또는 4채널인지
    ):
        raise ValueError("Bottom 검출에는 8비트 영상이 필요합니다.")

    gray = to_grayscale(image) # 그레이 스케일 변환
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # 3x3 가우시안 블러 적용
    border = np.concatenate((blurred[0], blurred[-1], blurred[:, 0], blurred[:, -1])) # 영상 4개의 테두리픽셀 추출
    background = float(np.median(border)) # 테두리 픽셀의 중앙값으로 밝기 산출

    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 오츠 이진화 적용
    threshold = max(background + MIN_BACKGROUND_CONTRAST, (background + otsu_threshold) / 2) # 이진화 임계값 설정
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY) # 이진화 수행
    return binary, threshold


def _locate_bottom_chip(binary):
    """이진화된 영상에서 칩 위치를 찾는다."""
    height, width = binary.shape
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)) # 닫힘 연산으로 결점 방지
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 가장 바깥쪽 외곽선만 추출

    candidates = []
    image_area = height * width 
    for contour in contours:
        # 1. 면적 비율 검사 (전체 면적 크기 대비 3% ~80%에 해당하는지)
        area = float(cv2.contourArea(contour))
        if not MIN_CHIP_AREA_RATIO * image_area <= area <= MAX_CHIP_AREA_RATIO * image_area:
            continue

        # 2. 최소 영역 회전 사각형 산출
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        rect_area = rect_width * rect_height
        if rect_area <= 0:
            continue

        # 3. 종횡비 검사
        if min(rect_width, rect_height) / max(rect_width, rect_height) < MIN_CHIP_ASPECT_RATIO:
            continue

        # 4. 직사각형도 검사
        if area / rect_area < MIN_CHIP_RECTANGULARITY:
            continue

        # 5. 화면 잘림 검사
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x + box_width >= width or y + box_height >= height:
            continue

        candidates.append((area, contour, rect, (x, y, box_width, box_height))) # 칩 형태 후보군 등록

    mask = np.zeros_like(binary)
    if not candidates:
        return None, mask
    
    # 통과한 후보 중 가장 면적이 큰(max) 객체를 최종 칩을 선택
    area, contour, rect, bounds = max(candidates, key=lambda candidate: candidate[0])

    corners = _ordered_corners(rect)
    top_edge = corners[1] - corners[0]
    left_edge = corners[3] - corners[0]
    chip = ChipLocation(
        contour=contour,
        corners=corners,
        center=tuple(float(value) for value in rect[0]), # 중심점
        size=(float(np.linalg.norm(top_edge)), float(np.linalg.norm(left_edge))), # 실제 가로/세로 길이
        angle_degrees=float(np.degrees(np.arctan2(top_edge[1], top_edge[0]))), # 기울어진 각도
        bounding_rect=bounds, # 정방형 바운딩 박스
        area=area, # 침 면적
    )
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED) # 칩 내부를 채운 이진 마스크
    return chip, mask


def find_bottom_chip(image_a):
    binary, threshold = _binarize(image_a)
    chip, mask = _locate_bottom_chip(binary)
    return chip, mask, threshold


def run_bottom_detection(image_path):
    # 파이프라인 진입점
    image_a, image_b = load_ab_tiff_pages(image_path)
    chip, mask, threshold = find_bottom_chip(image_a) # 칩 위치, 마스크, 임계값 검출
    binary, _ = _binarize(image_a) # 원 검출용 이진 영상

    regions = find_circle_regions(binary, chip, mask) # 원 후보 찾기
    visualization = draw_circle_regions(image_a, regions) # 시각화

    crop = None
    circle_preview = None

    if chip is not None:
        cv2.drawMarker(
            visualization, tuple(int(round(value)) for value in chip.center),
            CHIP_CENTER_COLOR, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA,
        ) # 칩 중심점에 십자 표시
        x, y, width, height = chip.bounding_rect
        crop = to_bgr(image_a[y:y + height, x:x + width]) # 칩 영역 자르기
        circle_preview = visualization[y:y + height, x:x + width].copy() # 원 검출 시각화 (draw_circle_regions)

    return BottomDetectionResult(
        raw_image_a=image_a, raw_image_b=image_b, chip=chip,
        source_visualization=visualization, chip_mask=mask,
        threshold=threshold, chip_crop=crop, circle_regions=regions,
        circle_preview=circle_preview, 
        roi_preview=create_circle_roi_preview(image_a, regions),
    )
