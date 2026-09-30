"""Side 표면·볼 검출 단계를 조립하는 실행 파이프라인."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from contour import (
    ContourMeasurement,
    find_b_contours,
    find_first_contact_points,
    get_contour_box_center_x,
)
from general import load_ab_tiff_pages, to_bgr, to_grayscale
from .ball import (
    draw_surface_slot_guides,
    find_surface_slot_brightness_lines,
    draw_surface_slot_brightness_lines,
    find_surface_slot_deepest_bright_points,
    draw_surface_slot_bright_points,
    draw_representative_point_rois,
    create_representative_point_rois,
    fit_representative_rois_to_contours,
    create_measured_ball_roi_preview,
    find_surface_slot_ball_contours,
    draw_surface_slot_ball_contours,
    get_bottom_contour_y_by_x,
    get_surface_slot_ranges,
    get_surface_slot_boundaries,
)
from .surface import (
    CONTOUR_COLOR,
    SOURCE_PREVIEW_MASK_MODE,
    create_polygon_preview,
    create_preview_comparison,
    draw_frequency_contact_points,
    draw_left_right_contour_reference_lines,
    draw_side_cutting_boundary,
    draw_side_cutting_guides,
    draw_top_pillar_reference_points,
    find_contour_contact_color_change_points,
    find_contour_contact_points,
    find_downward_color_change_points,
    find_left_right_contour_reference_points,
    find_top_pillar_reference_points,
    get_concentrated_cut_line,
    get_contour_side_reference_line,
    get_first_contact_side_lines,
    get_pillar_bottom_cut_line,
    get_pillar_outer_reference_points,
    get_side_cut_line,
)


@dataclass
class SideDetectionResult:
    """Side 검출 과정에서 생성되는 화면·좌표·성능 결과."""

    raw_image_a: Optional[np.ndarray] = None
    raw_image_b: Optional[np.ndarray] = None
    contours: List[np.ndarray] = field(default_factory=list)
    measurements: List[ContourMeasurement] = field(default_factory=list)
    analysis_preview: Optional[np.ndarray] = None
    source_preview: Optional[np.ndarray] = None
    source_visualization: Optional[np.ndarray] = None
    density_log_lines: List[str] = field(default_factory=list)
    center_split_x: Optional[int] = None
    surface_bottom_y_by_x: Dict[int, int] = field(default_factory=dict)
    ball_slot_count: int = 0
    ball_slot_ranges: List[Tuple[int, int]] = field(default_factory=list)
    ball_roi_polygons: List[np.ndarray] = field(default_factory=list)
    ball_roi_preview: Optional[np.ndarray] = None
    ball_contours: List[np.ndarray] = field(default_factory=list)
    ball_roi_measurements: List[Tuple[Optional[int], Optional[int], Optional[int]]] = field(default_factory=list)
    pillar_with_ball_ms: float = 0.0
    frequency_with_ball_ms: float = 0.0

    @property
    def is_detected(self):
        """컨투어와 볼 검사 ROI가 준비됐을 때 검출 준비 상태로 판단한다."""
        return bool(self.contours and self.ball_roi_polygons)


@dataclass
class _SurfaceCutAnalysis:
    """최상단/빈도 기준의 상·하면 컷선과 계산 시간."""

    top_boundary: Optional[dict]
    bottom_boundary: Optional[dict]
    elapsed_seconds: float

@dataclass
class _PillarAnalysis:
    """기둥 기준 검출에서 미리보기 생성에 필요한 중간 결과."""

    source_preview_image: np.ndarray
    top_cut_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    bottom_cut_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    left_reference_line: Tuple[Tuple[int, int], ...]
    right_reference_line: Tuple[Tuple[int, int], ...]
    elapsed_seconds: float


def _create_base_result(image_a, image_b, gray):
    contours = find_b_contours(image_b) # 표면 컨투어 추출
    result = SideDetectionResult(raw_image_a=image_a, raw_image_b=image_b, contours=contours, measurements=[find_first_contact_points(contour, gray.shape) for contour in contours],) # 접점 계산.
    result.center_split_x = get_contour_box_center_x( contours, gray.shape[1]) # 컨투어의 중앙위치를 기반으로 중심선 구함.
    return result


def _find_cut_boundary(
    measurements,
    point_attribute,
    image_width,
    use_maximum,
    side_name,
    midpoint_x,
):
    """밀집 컷선을 우선 사용하고 없으면 좌·우 빈도 컷선으로 대체한다."""
    boundary = get_concentrated_cut_line(
        measurements,
        point_attribute,
        image_width,
        use_maximum=use_maximum,
        side_name=side_name,
        midpoint_x=midpoint_x,
    )
    if boundary is not None:
        return boundary
    return get_side_cut_line(
        measurements,
        point_attribute,
        image_width,
        use_maximum=use_maximum,
        midpoint_x=midpoint_x,
    )


def _record_density_log(result, boundary):
    """컷선 계산 중 만들어진 밀집도 설명을 결과에 남긴다."""
    if boundary is None:
        return
    log_line = boundary.get("density_log_line")
    if log_line:
        result.density_log_lines.append(log_line)


def _analyze_surface_cuts(result, image_width):
    """항상 활성화되는 상·하면 측면 커팅 기준선을 계산한다."""
    started_at = perf_counter()
    top_boundary = _find_cut_boundary(
        result.measurements,
        "top_points",
        image_width,
        use_maximum=True,
        side_name="상면",
        midpoint_x=result.center_split_x,
    )
    top_elapsed = perf_counter() - started_at
    _record_density_log(result, top_boundary)

    started_at = perf_counter()
    bottom_boundary = _find_cut_boundary(
        result.measurements,
        "bottom_points",
        image_width,
        use_maximum=False,
        side_name="하면",
        midpoint_x=result.center_split_x,
    )
    bottom_elapsed = perf_counter() - started_at
    _record_density_log(result, bottom_boundary)
    return _SurfaceCutAnalysis(
        top_boundary=top_boundary,
        bottom_boundary=bottom_boundary,
        elapsed_seconds=top_elapsed + bottom_elapsed,
    )


def _create_result_image(gray, result, cuts):
    """B 페이지 위에 접점·기준선·측면 컷선을 그린다."""
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw_frequency_contact_points(
        result_image, result.measurements, "top_points", use_maximum=True
    )
    draw_frequency_contact_points(
        result_image, result.measurements, "bottom_points", use_maximum=False
    )
    draw_side_cutting_guides(result_image, result.center_split_x)
    draw_side_cutting_boundary(result_image, cuts.top_boundary)
    draw_side_cutting_boundary(result_image, cuts.bottom_boundary)
    return result_image


def _create_ball_rois(image_shape, result, ball_slot_count):
    """표면 하면의 슬롯 구간을 계산한다."""
    started_at = perf_counter()
    result.surface_bottom_y_by_x = get_bottom_contour_y_by_x(
        image_shape, result.contours
    )
    result.ball_slot_count = ball_slot_count
    result.ball_slot_ranges = get_surface_slot_ranges(
        result.surface_bottom_y_by_x,
        3 if result.ball_slot_count == 2 else result.ball_slot_count,
    )
    return perf_counter() - started_at


def _analyze_pillar_surface(image_a, image_shape, result, cuts=None):
    """기둥 기준점과 컨투어 접점을 계산하고 A 페이지에 표시한다."""
    started_at = perf_counter()
    pillar_reference_points = find_top_pillar_reference_points(image_a)
    pillar_outer_reference_points = get_pillar_outer_reference_points(
        image_a, pillar_reference_points
    )
    pillar_downward_points = find_downward_color_change_points(
        image_a, pillar_outer_reference_points
    )
    contour_outline = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_outline, result.contours, -1, 255, thickness=1)
    left_contour_points, right_contour_points = (
        find_left_right_contour_reference_points(contour_outline)
    )
    left_reference_line = get_contour_side_reference_line(
        left_contour_points, image_shape[0]
    )
    right_reference_line = get_contour_side_reference_line(
        right_contour_points, image_shape[0]
    )
    if not left_reference_line or not right_reference_line:
        left_reference_line, right_reference_line = get_first_contact_side_lines(
            result.measurements, image_shape[0]
        )
    pillar_downward_bright_points = find_contour_contact_color_change_points(
        image_a, pillar_downward_points, contour_outline
    )
    pillar_contour_contact_points = find_contour_contact_points(
        pillar_downward_points, contour_outline
    )

    if len(pillar_downward_points) == 2:
        top_cut_line = pillar_downward_points
    elif len(pillar_downward_points) == 1:
        point_y = pillar_downward_points[0][1]
        top_cut_line = ((0, point_y), (image_shape[1] - 1, point_y))
    else:
        top_cut_line = None

    frequency_bottom_cut_line = (
        (cuts.bottom_boundary or {}).get("extended_line") if cuts else None
    )
    bottom_cut_line = (
        frequency_bottom_cut_line
        if frequency_bottom_cut_line is not None
        else get_pillar_bottom_cut_line(
            pillar_contour_contact_points, image_shape[1]
        )
    )

    source_visualization = to_bgr(image_a)
    cv2.drawContours(source_visualization, result.contours, -1, CONTOUR_COLOR, 1)
    source_visualization = draw_left_right_contour_reference_lines(
        source_visualization, left_contour_points, right_contour_points
    )
    source_visualization = draw_top_pillar_reference_points(
        source_visualization,
        pillar_reference_points,
        pillar_downward_points,
        pillar_downward_bright_points,
        bottom_cut_line=bottom_cut_line,
    )

    result.source_visualization = source_visualization
    draw_surface_slot_guides(
        result.source_visualization, result.ball_slot_ranges, bottom_cut_line
    )
    return _PillarAnalysis(
        source_preview_image=to_bgr(image_a),
        top_cut_line=top_cut_line,
        bottom_cut_line=bottom_cut_line,
        left_reference_line=left_reference_line,
        right_reference_line=right_reference_line,
        elapsed_seconds=perf_counter() - started_at,
    )


def _create_pillar_preview(image_b, result, pillar):
    """기둥(상면)·빈도(하면) 기준의 A/B Crop 비교 이미지를 만든다."""
    a_preview = create_polygon_preview(
        pillar.source_preview_image,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=pillar.top_cut_line,
        bottom_cut_line=pillar.bottom_cut_line,
        left_reference_line=pillar.left_reference_line,
        right_reference_line=pillar.right_reference_line,
    )
    b_preview = create_polygon_preview(
        image_b,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=pillar.top_cut_line,
        bottom_cut_line=pillar.bottom_cut_line,
        left_reference_line=pillar.left_reference_line,
        right_reference_line=pillar.right_reference_line,
    )
    return create_preview_comparison(
        a_preview, "A Page crop", b_preview, "B Page crop"
    )


def run_side_detection(image_path, ball_slot_count=3):
    image_a, image_b = load_ab_tiff_pages(image_path) # 이미지 불러오기

    # 두 이미지의 크기가 일치하지 않는 경우
    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(f"A/B 페이지 크기가 일치하지 않습니다: {image_path}")

    # B 페이지를 그레이스케일로 변환
    gray = to_grayscale(image_b)
    
    # B 페이지의 표면 컨투어 추출
    result = _create_base_result(image_a, image_b, gray)
    
    # 하면 기준 컷선 산출(최상단/빈도 계산 기반)
    cuts = _analyze_surface_cuts(result, gray.shape[1])
    result_image = _create_result_image(gray, result, cuts)

    # 표면 기준 볼 ROI 생성 및 시각화
    ball_seconds = _create_ball_rois(gray.shape, result, ball_slot_count)
    draw_surface_slot_guides(
        result_image, result.ball_slot_ranges,
        (cuts.bottom_boundary or {}).get("extended_line"),
    )

    # 상면 기준 기둥 탐색 및 시각화(기둥 계산 기반)
    pillar = _analyze_pillar_surface(image_a, gray.shape, result, cuts=cuts)

    brightness_lines = find_surface_slot_brightness_lines(
        image_a, result.ball_slot_ranges, pillar.bottom_cut_line
    )
    if result.ball_slot_count == 2:
        boundaries = get_surface_slot_boundaries(result.ball_slot_ranges)
        if len(boundaries) == 4:
            brightness_lines = [
                line for line in brightness_lines
                if not boundaries[1] <= line[0][0] < boundaries[2]
            ]
    draw_surface_slot_brightness_lines(result.source_visualization, brightness_lines)
    draw_surface_slot_brightness_lines(result_image, brightness_lines)
    bright_points = find_surface_slot_deepest_bright_points(image_a, brightness_lines)
    result.ball_contours = find_surface_slot_ball_contours(
        image_a, brightness_lines
    )
    draw_surface_slot_ball_contours(result.source_visualization, result.ball_contours)
    draw_surface_slot_ball_contours(result_image, result.ball_contours)
    draw_surface_slot_bright_points(result.source_visualization, bright_points)
    draw_surface_slot_bright_points(result_image, bright_points)
    point_rois = create_representative_point_rois(
        brightness_lines, bright_points, result.ball_slot_ranges
    )
    point_rois = fit_representative_rois_to_contours(point_rois, result.ball_contours)
    result.ball_roi_polygons = [
        np.asarray(((left, top), (right, top), (right, bottom), (left, bottom)), dtype=np.int32)
        for _, (left, top, right, bottom) in point_rois
    ]
    draw_representative_point_rois(result.source_visualization, point_rois)
    draw_representative_point_rois(result_image, point_rois)
    result.ball_roi_preview, result.ball_roi_measurements = create_measured_ball_roi_preview(
        image_a, point_rois, result.ball_contours
    )


    started_at = perf_counter() # 알고리즘 소요 시간 계산(ms)
    result.analysis_preview = _create_pillar_preview(image_b, result, pillar)

    # 표면 분석 소요 시간 합산
    # 상단 + 하면 + 크롭 미리보기 이미지 합성 시간 
    pillar_seconds = (
        pillar.elapsed_seconds + cuts.elapsed_seconds + (perf_counter() - started_at)
    )

    # A컷 이미지 준비
    result.source_preview = None 
    # 최종 결과 반환
    result.pillar_with_ball_ms = (pillar_seconds + ball_seconds) * 1000
    result.frequency_with_ball_ms = None
    return result_image, result


# 기존 호출부와 외부 사용 코드가 바로 깨지지 않도록 이전 이름을 유지한다.
DetectionResult = SideDetectionResult
create_detection_visualization = run_side_detection
