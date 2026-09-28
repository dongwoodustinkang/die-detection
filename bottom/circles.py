"""칩 기준 6개 ROI에서 검은 성분의 실제 윤곽과 원 후보 조건을 검사한다."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from general import to_bgr


ROI_SIZE = 40.0

# 원형 조건 판정을 위한 파라미터
MIN_CIRCLE_AREA = 300.0
MAX_CIRCLE_AREA = 450.0
MIN_CIRCULARITY = 0.70
MIN_CIRCLE_ASPECT_RATIO = 0.70
MAX_CENTER_DISTANCE = 12.0
# 작은 픽셀 잡음만 제외한다. 조건에 미달한 큰 성분은 검토용으로 남긴다.
MIN_COMPONENT_AREA = 10.0

# 칩 기준 6개 ROI 위치 (상단 변으로부터의 비율)
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
    """검출된 원의 수치 결과 및 조건 판정 정보"""
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
    """칩 기준 6개 ROI의 검사 결과"""
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
    """검사 대상(dark)의 면적이나 원형도, 거리등의 특징을 측정하고 부합하는지 판정하는 함수"""
    # 노이즈 필터링
    area = float(cv2.contourArea(contour)) # 실제 면적을 계산하여 기준치 미만인지 확인
    if area < MIN_COMPONENT_AREA:
        return None

    # 특징 측정(무게중심, 둘레 길이, 원형도, 종횡비, 중심거리)
    moments = cv2.moments(contour)
    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]) # 덩어리의 중심점 계산
    perimeter = cv2.arcLength(contour, True) # 덩어리의 테두리 길이 계산
    circularity = float(4 * np.pi * area / perimeter ** 2) if perimeter else 0.0 # 원형도 계산(얼마나 둥근지)
    _, _, width, height = cv2.boundingRect(contour)
    aspect = min(width, height) / max(width, height) # 종횡비(가로 세로 비율로 비슷한지 아닌지)
    distance = float(np.linalg.norm(np.asarray(center) - expected_center)) # 중심거리

    # 검색 경계에 닿아 윤곽 확인 필요 조건
    points = contour[:, 0]
    touches_boundary = bool(np.any(search_boundary[points[:, 1], points[:, 0]]))

    # 조건 미달 사유 추가
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
    if chip is None:
        return []

    top_edge = chip.corners[1] - chip.corners[0] # 좌상단의 변 벡터 (상단)
    left_edge = chip.corners[3] - chip.corners[0] # 좌상,좌하의 변 벡터 (좌단)
    half_x = top_edge / np.linalg.norm(top_edge) * ROI_SIZE / 2 # 탐색 영역의 절반 크기 (top_edge의 실제 픽셀 길이)
    half_y = left_edge / np.linalg.norm(left_edge) * ROI_SIZE / 2

    # chip.corners[N] : 좌상, 우상, 우하, 좌하단의 모서리 좌표
    # 단위 벡터(top_edge / np.linalg.norm(top_edge))에 탐색 영역의 절반 크기(ROI_SIZE / 2)를 곱함
    # 이를 통해 상단 변 방향으로 ROI_SIZE/2 만큼의 벡터를 얻는다.

    roi = np.zeros_like(binary) # roi 마스크 배열 1회 생성 후 재사용
    regions = []

    # 6개 위치별 회전 ROI 박스 생성
    for index, (name, fx, fy) in enumerate(ROI_POSITIONS, 1): # fx, fy = (0.1, 0.1), (0.9, 0.1), ... -> 칩 모서리에서 (0.1, 0.1)만큼 떨어진 지점
        center = chip.corners[0] + fx * top_edge + fy * left_edge # ROI 예상 중심 좌표로 가로 방형으로 10% , 세로 방향으로 10% 지점의 중심 좌표를 말한다. 
        corners = np.asarray((
            center - half_x - half_y, center + half_x - half_y, 
            center + half_x + half_y, center - half_x + half_y,
        ), dtype=np.float32) # ROI의 모서리 좌표 (좌상, 우상, 우하, 좌하 순서)

        roi.fill(0) # 기존 배열 재사용하여 0으로 초기화
        cv2.fillConvexPoly(roi, np.rint(corners).astype(np.int32), 255) # 회전된 ROI 영역 채우기 (roi 마스크)
        search_mask = cv2.bitwise_and(roi, chip_mask) # ROI 영역과 칩 마스크 교집합
        eroded = cv2.erode(search_mask, np.ones((3, 3), dtype=np.uint8)) # 침 마스크 침식
        search_boundary = (search_mask > 0) & (eroded == 0) # 침 마스크의 경계선
        dark = cv2.bitwise_and(cv2.bitwise_not(binary), search_mask) # 비트 연산으로 검은 픽셀(0)을 255로 반전 시켜 객체 추출

        # 검은 성분(dark) 윤곽선 검출 및 측정
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        components = []
        for contour in contours:
            component = _measure_component(contour, center, search_boundary) # 검출된 원의 수치 결과 정보
            if component is not None:
                components.append(component)

        # 검은 성분(dark)의 중심거리, 면적 순서로 정렬
        components.sort(key=lambda c: (not c.is_candidate, c.center_distance, -c.area))
        regions.append(CircleRegion(
            index, name, tuple(float(value) for value in center), corners,
            components, components[0] if components else None,
        ))
    return regions


def _region_color(region):
    return {"candidate": CANDIDATE_COLOR, "review": REVIEW_COLOR, "missing": MISSING_COLOR}[region.status]


def draw_circle_regions(image, regions):
    """칩 영역 표시"""
    result = to_bgr(image)
    for region in regions:
        corners = np.rint(region.corners).astype(np.int32) # 모서리 좌표를 반올림(rint)하여 정수로 변경
        color = _region_color(region) # 초록: 원 후보, 주황: 검토, 빨강: 성분 없음
        cv2.polylines(result, [corners], True, ROI_COLOR, 1, cv2.LINE_AA) # result에 ROI 영역 그리기

        # 원 그리기
        for component in region.components:
            contour_color = color if component is region.primary else REVIEW_COLOR
            cv2.drawContours(result, [component.contour], -1, contour_color, 1, cv2.LINE_AA) # 원 윤곽선
        x, y = corners[0]
        cv2.putText(result, str(region.index), (max(0, x + 2), max(9, y + 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA) # ROI 식별 번호
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
