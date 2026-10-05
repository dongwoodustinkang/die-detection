"""Side 표면·볼 검출 단계를 조립하는 실행 파이프라인."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from contour import (
    ContourMeasurement,
    MIN_CONTOUR_AREA,
    MAX_CONTOUR_AREA,
    find_b_contours,
    find_first_contact_points,
    get_contour_box_center_x,
)
from general import load_ab_tiff_pages, to_bgr, to_grayscale
from .ball import (
    draw_surface_slot_guides,
    find_surface_slot_brightness_lines,
    draw_surface_slot_brightness_lines,
    offset_surface_slot_lines,
    draw_surface_slot_offset_lines,
    find_surface_slot_ball_rois,
    draw_surface_slot_ball_boxes,
    pad_ball_roi,
    create_slot_ball_preview,
    get_bottom_contour_y_by_x,
    get_surface_slot_ranges,
    get_surface_slot_boundaries,
    split_slot_ranges,
)
from .surface import (
    CONTOUR_COLOR,
    SOURCE_PREVIEW_MASK_MODE,
    apply_side_cutting_mask,
    create_polygon_preview,
    create_preview_comparison,
    crop_polygon_tile,
    draw_frequency_contact_points,
    draw_left_right_contour_reference_lines,
    draw_side_cutting_boundary,
    draw_side_cutting_guides,
    draw_top_pillar_reference_points,
    find_contour_contact_color_change_points,
    find_contour_contact_points,
    find_downward_color_change_points,
    find_left_right_contour_reference_points,
    find_page_a_band_contours,
    get_page_a_bottom_x_range,
    find_top_pillar_reference_points,
    get_concentrated_cut_line,
    get_contour_side_reference_line,
    get_first_contact_side_lines,
    get_pillar_bottom_cut_line,
    get_pillar_outer_reference_points,
    get_side_cut_line,
    extend_line_to_image_edges,
)


# 임시 실험: False로 바꾸면 Page A 크롭 재검출을 사용하지 않는다.
USE_CROPPED_PAGE_A_FALLBACK = True
# 임시 실험: True면 처음부터 A 상단 기준선 아래로 자른 B에서 컨투어를 따서
# 하단 기준선을 만든다. False로 바꾸면 기존(B 전체 → 실패 시 크롭) 흐름으로 돌아간다.
USE_TOP_CROPPED_PAGE_B = True


@dataclass
class SideDetectionResult:
    """Side 검출 과정에서 생성되는 화면·좌표·성능 결과."""

    raw_image_a: Optional[np.ndarray] = None
    raw_image_b: Optional[np.ndarray] = None
    contours: List[np.ndarray] = field(default_factory=list)
    page_a_contours: List[np.ndarray] = field(default_factory=list)
    used_cropped_page_a: bool = False
    fallback_pillar_x_range: Optional[Tuple[int, int]] = None
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
    # (슬롯 번호, 너비 px, 높이 px, (left, top, right, bottom), 판정)
    ball_slot_measurements: List[Tuple[Optional[int], Optional[int], Optional[int], Optional[Tuple[int, int, int, int]], str]] = field(default_factory=list)
    pillar_with_ball_ms: float = 0.0
    frequency_with_ball_ms: float = 0.0

    @property
    def is_detected(self):
        """표면 컨투어가 준비됐을 때 검출 준비 상태로 판단한다."""
        return bool(self.contours)


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
    page_a_contour: Optional[np.ndarray] = None


def _create_base_result(image_a, image_b, gray):
    contours = find_b_contours(image_b) # 표면 컨투어 추출
    result = SideDetectionResult(raw_image_a=image_a, raw_image_b=image_b, contours=contours, measurements=[find_first_contact_points(contour, gray.shape) for contour in contours],) # 접점 계산.
    result.center_split_x = get_contour_box_center_x( contours, gray.shape[1]) # 컨투어의 중앙위치를 기반으로 중심선 구함.
    return result


def _retry_surface_from_page_a(image_a, result, image_shape):
    """A 상단 기준선으로 B를 크롭해 재검출하고 성공한 경우에만 교체한다."""
    reference_points = find_top_pillar_reference_points(image_a)
    outer_points = get_pillar_outer_reference_points(image_a, reference_points)
    downward_points = find_downward_color_change_points(image_a, outer_points)
    if len(downward_points) == 2:
        top_cut_line = downward_points
    elif len(downward_points) == 1:
        point_y = downward_points[0][1]
        top_cut_line = ((0, point_y), (image_shape[1] - 1, point_y))
    else:
        return False

    surface_image = to_grayscale(result.raw_image_b)
    mask = np.full(image_shape, 255, dtype=np.uint8)
    crop_cut_line = tuple((x, y + 3) for x, y in top_cut_line)
    apply_side_cutting_mask(mask, 0, 0, crop_cut_line, None)
    valid_rows = np.flatnonzero(np.any(mask, axis=1))
    if not valid_rows.size:
        return False
    crop_top = int(valid_rows[0])
    crop = surface_image[crop_top:]
    crop_mask = mask[crop_top:] != 0
    # 잘라낸 영역의 검은 픽셀을 Otsu 이진화 통계에 포함하지 않는다.
    threshold, _ = cv2.threshold(
        crop[crop_mask].reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    contours = []
    # 크롭 경계의 밝은 배경과 붙으면 임계값을 조금씩 높여 표면을 분리한다.
    for candidate_threshold in range(int(threshold), 256, 5):
        binary = np.where(
            crop_mask & (crop > candidate_threshold), 255, 0
        ).astype(np.uint8)
        candidates, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            offset=(0, crop_top),
        )
        contours = [
            contour for contour in candidates
            if MIN_CONTOUR_AREA <= cv2.contourArea(contour) <= MAX_CONTOUR_AREA
            and cv2.boundingRect(contour)[0] > 0
            and sum(cv2.boundingRect(contour)[::2]) < image_shape[1]
        ]
        if contours:
            break
    if not contours:
        return False
    result.contours = contours
    result.measurements = [
        find_first_contact_points(contour, image_shape) for contour in contours
    ]
    result.center_split_x = get_contour_box_center_x(contours, image_shape[1])
    result.density_log_lines.clear()
    result.used_cropped_page_a = True
    if len(downward_points) == 2:
        result.fallback_pillar_x_range = tuple(sorted(point[0] for point in downward_points))
    return True


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
    """컨투어 컷선을 계산한다. 크롭 재검출 컨투어는 하면만 사용한다."""
    started_at = perf_counter()
    top_boundary = None if result.used_cropped_page_a else _find_cut_boundary(
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
    if result.used_cropped_page_a and result.fallback_pillar_x_range and bottom_boundary:
        # 컨투어로 구한 하단 높이/기울기를 유지하고 기둥 x에서 양 끝을 잡는다.
        (x1, y1), (x2, y2) = bottom_boundary["extended_line"]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        line = tuple(
            (x, int(round(y1 + (x - x1) * slope)))
            for x in result.fallback_pillar_x_range
        )
        bottom_boundary = dict(bottom_boundary, line=line,
            extended_line=extend_line_to_image_edges(line, image_width),
            markers=tuple((point, False) for point in line))
    _record_density_log(result, bottom_boundary)
    return _SurfaceCutAnalysis(
        top_boundary=top_boundary,
        bottom_boundary=bottom_boundary,
        elapsed_seconds=top_elapsed + bottom_elapsed,
    )


def _create_result_image(gray, result, cuts):
    """B 페이지 위에 접점·기준선·측면 컷선을 그린다."""
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not result.used_cropped_page_a:
        draw_frequency_contact_points(
            result_image, result.measurements, "top_points", use_maximum=True
        )
    draw_frequency_contact_points(
        result_image, result.measurements, "bottom_points", use_maximum=False
    )
    draw_side_cutting_guides(result_image, result.center_split_x)
    draw_side_cutting_boundary(result_image, cuts.bottom_boundary)
    return result_image


def _create_ball_rois(image_shape, result, ball_slot_count):
    """표면 하면의 슬롯 구간을 계산한다."""
    started_at = perf_counter()
    result.surface_bottom_y_by_x = get_bottom_contour_y_by_x(
        image_shape, result.contours
    )
    if result.used_cropped_page_a and result.fallback_pillar_x_range:
        left_x, right_x = result.fallback_pillar_x_range
        result.surface_bottom_y_by_x = {
            x: y for x, y in result.surface_bottom_y_by_x.items()
            if left_x <= x <= right_x
        }
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
    if result.used_cropped_page_a and result.fallback_pillar_x_range:
        left_x, right_x = result.fallback_pillar_x_range
        left_reference_line = ((left_x, 0), (left_x, image_shape[0] - 1))
        right_reference_line = ((right_x, 0), (right_x, image_shape[0] - 1))
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

    # 상단~하단 기준선 사이에서 A 페이지 컨투어를 다시 따서 표시한다.
    result.page_a_contours = find_page_a_band_contours(
        image_a, top_cut_line, bottom_cut_line
    )
    page_a_contour = (
        max(result.page_a_contours, key=cv2.contourArea)
        if result.page_a_contours else None
    )
    source_visualization = to_bgr(image_a)
    cv2.drawContours(
        source_visualization, result.page_a_contours, -1, CONTOUR_COLOR, 1
    )
    if not result.used_cropped_page_a:
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
    return _PillarAnalysis(
        source_preview_image=to_bgr(image_a),
        top_cut_line=top_cut_line,
        bottom_cut_line=bottom_cut_line,
        left_reference_line=left_reference_line,
        right_reference_line=right_reference_line,
        elapsed_seconds=perf_counter() - started_at,
        page_a_contour=page_a_contour,
    )


def _fit_slot_ranges_to_page_a(result, pillar):
    """A 컨투어 하단의 좌·우 끝 사이로 슬롯 구간을 다시 나눈다."""
    if pillar.page_a_contour is None:
        return
    x_range = get_page_a_bottom_x_range(
        pillar.page_a_contour, pillar.source_preview_image.shape
    )
    if x_range is None:
        return
    result.ball_slot_ranges = split_slot_ranges(
        *x_range,
        3 if result.ball_slot_count == 2 else result.ball_slot_count,
    )


def _create_pillar_preview(image_b, result, pillar):
    """기둥(상면)·빈도(하면) 기준의 A/B Crop 비교 이미지를 만든다."""
    if pillar.page_a_contour is not None:
        # A 밴드 컨투어(상·하단 기준선 + 좌·우 컨투어 외곽) 그대로 두 페이지를 자른다.
        return create_preview_comparison(
            crop_polygon_tile(pillar.source_preview_image, pillar.page_a_contour),
            "이미지 A",
            crop_polygon_tile(image_b, pillar.page_a_contour),
            "이미지 B",
        )
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
        a_preview, "이미지 A", b_preview, "이미지 B"
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
    if USE_TOP_CROPPED_PAGE_B:
        # A 상단 기준선 아래 B 크롭 컨투어로 교체한다(실패하면 B 전체 컨투어 유지).
        _retry_surface_from_page_a(image_a, result, gray.shape)
    cuts = _analyze_surface_cuts(result, gray.shape[1])
    if (
        USE_CROPPED_PAGE_A_FALLBACK and not result.used_cropped_page_a
        and cuts.bottom_boundary is None
    ):
        if _retry_surface_from_page_a(image_a, result, gray.shape):
            cuts = _analyze_surface_cuts(result, gray.shape[1])
    result_image = _create_result_image(gray, result, cuts)

    # 표면 기준 볼 ROI 생성 및 시각화
    ball_seconds = _create_ball_rois(gray.shape, result, ball_slot_count)

    # 상면 기준 기둥 탐색 및 시각화(기둥 계산 기반)
    pillar = _analyze_pillar_surface(image_a, gray.shape, result, cuts=cuts)
    _fit_slot_ranges_to_page_a(result, pillar)
    draw_surface_slot_guides(
        result_image, result.ball_slot_ranges,
        (cuts.bottom_boundary or {}).get("extended_line"),
    )
    draw_surface_slot_guides(
        result.source_visualization, result.ball_slot_ranges, pillar.bottom_cut_line
    )

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
    offset_lines = offset_surface_slot_lines(
        brightness_lines, pillar.top_cut_line, pillar.bottom_cut_line
    )
    draw_surface_slot_offset_lines(result.source_visualization, offset_lines)
    draw_surface_slot_offset_lines(result_image, offset_lines)
    # 2개 슬롯 모드는 가운데 슬롯을 빼고 Slot 1, Slot 3만 검사한다.
    slot_indices = [
        index for index in range(len(get_surface_slot_boundaries(result.ball_slot_ranges)) - 1)
        if not (result.ball_slot_count == 2 and index == 1)
    ]
    result.ball_slot_measurements = find_surface_slot_ball_rois(
        image_a, brightness_lines, offset_lines, result.ball_slot_ranges, slot_indices
    )
    result.ball_roi_polygons = [
        np.asarray(((left, top), (right, top), (right, bottom), (left, bottom)), dtype=np.int32)
        for _, _, _, box, _ in result.ball_slot_measurements if box is not None
        for left, top, right, bottom in (pad_ball_roi(box, image_a.shape),)
    ]
    draw_surface_slot_ball_boxes(result.source_visualization, result.ball_slot_measurements)
    draw_surface_slot_ball_boxes(result_image, result.ball_slot_measurements)
    result.ball_roi_preview = create_slot_ball_preview(
        image_a, result.ball_slot_measurements,
        len(get_surface_slot_boundaries(result.ball_slot_ranges)) - 1,
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
