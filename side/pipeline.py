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
    create_ball_square_crop_preview,
    draw_ball_squares,
    find_downward_contour_white_points,
    get_bottom_contour_y_by_x,
    get_distant_bottommost_points,
)
from .ball_arc import BallArcMeasurement, analyze_ball_crops, create_ball_arc_preview
from .surface import (
    CONTOUR_COLOR,
    SOURCE_PREVIEW_MASK_MODE,
    append_density_log_when_merge_exists,
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
    a_ball_white_points: List[Tuple[int, int]] = field(default_factory=list)
    a_ball_bottommost_y: Optional[int] = None
    a_ball_bottommost_count: int = 0
    a_ball_bottommost_points_by_third: List[Optional[Tuple[int, int]]] = field(
        default_factory=list
    )
    selected_ball_bottommost_points: List[Tuple[int, int]] = field(
        default_factory=list
    )
    ball_square_crop_preview: Optional[np.ndarray] = None
    ball_square_surface_bottom_y_by_x: Dict[int, int] = field(default_factory=dict)
    ball_arc_measurements: List[BallArcMeasurement] = field(default_factory=list)
    ball_arc_preview: Optional[np.ndarray] = None
    ball_arc_ms: Optional[float] = None
    pillar_with_ball_ms: float = 0.0
    frequency_with_ball_ms: float = 0.0

    @property
    def is_detected(self):
        """컨투어와 볼이 모두 검출됐을 때만 최종 검출로 판단한다."""
        return bool(self.contours and self.selected_ball_bottommost_points)


@dataclass
class _SurfaceCutAnalysis:
    """최상단/빈도 기준의 상·하면 컷선과 계산 시간."""

    top_boundary: Optional[dict]
    bottom_boundary: Optional[dict]
    elapsed_seconds: float


def measure_ball_arcs(result):
    """요청 시 한 번만 원호를 분석해 동일한 원본/크롭 결과와 비교한다."""
    if result.ball_arc_ms is not None:
        return
    started_at = perf_counter()
    if result.raw_image_a is not None:
        result.ball_arc_measurements = analyze_ball_crops(
            result.raw_image_a, result.selected_ball_bottommost_points,
            result.ball_square_surface_bottom_y_by_x,
        )
    result.ball_arc_preview = create_ball_arc_preview(result.ball_arc_measurements)
    result.ball_arc_ms = (perf_counter() - started_at) * 1000


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
    """컨투어와 첫 접점을 계산해 Side 검출 결과의 공통 기반을 만든다."""
    contours = find_b_contours(image_b)
    result = SideDetectionResult(
        raw_image_a=image_a,
        raw_image_b=image_b,
        contours=contours,
        measurements=[
            find_first_contact_points(contour, gray.shape)
            for contour in contours
        ],
    )
    result.center_split_x = get_contour_box_center_x(
        contours, gray.shape[1]
    )
    append_density_log_when_merge_exists(
        result,
        "top_points",
        result.center_split_x,
        use_maximum=True,
        side_name="상면",
    )
    append_density_log_when_merge_exists(
        result,
        "bottom_points",
        result.center_split_x,
        use_maximum=False,
        side_name="하면",
    )
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
    """컷선 계산 중 만들어진 밀집도 설명을 결과와 터미널에 남긴다."""
    if boundary is None:
        return
    log_line = boundary.get("density_log_line")
    if log_line:
        result.density_log_lines.append(log_line)
        print(log_line)


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


def _detect_ball(image_a, image_shape, result):
    """A 페이지에서 볼 후보를 찾고 관련 결과 필드를 채운다."""
    started_at = perf_counter()
    result.a_ball_white_points = find_downward_contour_white_points(
        image_a, result.contours
    )
    result.selected_ball_bottommost_points = get_distant_bottommost_points(
        result.a_ball_white_points, min_x_distance=40, max_y_difference=10
    )
    result.ball_square_surface_bottom_y_by_x = get_bottom_contour_y_by_x(
        image_shape, result.contours
    )
    result.ball_square_crop_preview = create_ball_square_crop_preview(
        image_a,
        result.selected_ball_bottommost_points,
        result.ball_square_surface_bottom_y_by_x,
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
    left_reference_line, right_reference_line = get_first_contact_side_lines(
        result.measurements, image_shape[0]
    )
    pillar_downward_bright_points = find_contour_contact_color_change_points(
        image_a, pillar_downward_points, contour_outline
    )
    pillar_contour_contact_points = find_contour_contact_points(
        pillar_downward_points, contour_outline
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
    )

    result.source_visualization = draw_ball_squares(
        source_visualization,
        result.selected_ball_bottommost_points,
        result.ball_square_surface_bottom_y_by_x,
    )

    if len(pillar_downward_points) == 2:
        top_cut_line = pillar_downward_points
    elif len(pillar_downward_points) == 1:
        point_y = pillar_downward_points[0][1]
        top_cut_line = ((0, point_y), (image_shape[1] - 1, point_y))
    else:
        top_cut_line = None
    bottom_cut_line = get_pillar_bottom_cut_line(
        pillar_contour_contact_points, image_shape[1]
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
    """기둥 기준의 A/B Crop 비교 이미지를 만든다."""
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


def _create_frequency_preview(image_b, result, pillar, cuts):
    """최상단/빈도 기준의 A/B Crop 비교 이미지를 만든다."""
    top_cut_line = (cuts.top_boundary or {}).get("extended_line")
    bottom_cut_line = (cuts.bottom_boundary or {}).get("extended_line")
    a_preview = create_polygon_preview(
        pillar.source_preview_image,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=top_cut_line,
        bottom_cut_line=bottom_cut_line,
    )
    b_preview = create_polygon_preview(
        image_b,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=top_cut_line,
        bottom_cut_line=bottom_cut_line,
    )
    return create_preview_comparison(
        a_preview, "A Page crop", b_preview, "B Page crop"
    )


def run_side_detection(image_path):
    """Side 검출을 실행하고 화면 표시용 이미지와 결과를 반환한다."""

    image_a, image_b = load_ab_tiff_pages(image_path)
    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(f"A/B 페이지 크기가 일치하지 않습니다: {image_path}")

    gray = to_grayscale(image_b)
    result = _create_base_result(image_a, image_b, gray)
    cuts = _analyze_surface_cuts(result, gray.shape[1])
    result_image = _create_result_image(gray, result, cuts)
    ball_seconds = _detect_ball(image_a, gray.shape, result)
    pillar = _analyze_pillar_surface(image_a, gray.shape, result, cuts=cuts)

    started_at = perf_counter()
    result.analysis_preview = _create_pillar_preview(image_b, result, pillar)
    pillar_seconds = pillar.elapsed_seconds + (perf_counter() - started_at)

    started_at = perf_counter()
    result.source_preview = _create_frequency_preview(
        image_b, result, pillar, cuts
    )
    frequency_seconds = cuts.elapsed_seconds + (perf_counter() - started_at)

    result.pillar_with_ball_ms = (pillar_seconds + ball_seconds) * 1000
    result.frequency_with_ball_ms = (frequency_seconds + ball_seconds) * 1000
    return result_image, result


# 기존 호출부와 외부 사용 코드가 바로 깨지지 않도록 이전 이름을 유지한다.
DetectionResult = SideDetectionResult
create_detection_visualization = run_side_detection
