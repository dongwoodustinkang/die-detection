from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ball import (
    create_ball_square_crop_preview,
    draw_ball_squares,
    get_bottom_contour_x_range,
    get_bottom_contour_y_by_x,
    find_downward_contour_white_points,
    get_bottommost_point_count,
    get_bottommost_points_by_thirds,
    select_detected_bottommost_points,
)

# 검출할 컨투어의 최소/최대 크기
MIN_CONTOUR_AREA = 3500
MAX_CONTOUR_AREA = 10000

# 기존 균등 샘플 수 (CONTACT_POINT_MODE = "sampled"일 때만 사용)
TOP_BOTTOM_SAMPLE_COUNT = 15
LEFT_RIGHT_SAMPLE_COUNT = 5
CONTACT_POINT_MODE = "full"  # "full"이면 전체 접점 분포, "sampled"면 기존 15/5 샘플
TOP_DENSE_BAND_TOLERANCE = 1
MAX_COUNT_RATIO = 0.5  # 최빈 접점 수를 기준으로 삼을 비율
FIRST_CONTACT_BAND_PIXELS = 2

# 점(원)의 크기 조절
REFERENCE_POINT_RADIUS = 2
REFERENCE_POINT_MIN_DISTANCE = 30  # 실제 보조 접점과의 최소 거리(px)
EXTREME_SENCODARY_DIFF = 2
MIN_DISTANCE = 30
APPROX_POLYGON_EPSILON_RATIO = 0.02
ANALYSIS_PREVIEW_MASK_MODE = "contour"
SOURCE_PREVIEW_MASK_MODE = "line_quadrilateral"
SOURCE_PREVIEW_OUTER_MARGIN = 0
SOURCE_PREVIEW_TOP_SCAN_HEIGHT = 3
SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD = 40
SOURCE_PREVIEW_EDGE_CHANGE_WINDOW = 3
SOURCE_PREVIEW_HORIZONTAL_CHANGE_MIN_RUN = 3
SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN = 3
SOURCE_PREVIEW_INVALID_Y_DIFFERENCE = 10
# "contour", "approx_polygon", 또는 "line_quadrilateral"
LINE_QUADRILATERAL_METHOD = "primary_axis"
# "primary_axis", "robust_contacts" 또는 이전 방식인 "extreme_pairs"

CONTOUR_COLOR = (247, 85, 168)  # 보라색 (BGR)
TOP_COLOR = (248, 189, 56)  # 하늘색 (BGR)
BOTTOM_COLOR = (0, 0, 255)
PILLAR_REFERENCE_POINT_COLOR = (128, 128, 128)
PILLAR_DOWNWARD_COLOR = TOP_COLOR
PILLAR_DOWNWARD_BRIGHT_POINT_COLOR = (255, 230, 180)
PILLAR_POINT_RADIUS = 1
PILLAR_REFERENCE_POINT_OUTER_OFFSET = 5
PILLAR_REFERENCE_BRIGHTNESS_MIN = 250
PILLAR_REFERENCE_BRIGHTNESS_MAX = 255
PILLAR_CONTOUR_CONTACT_SCAN_OFFSET = 20
PILLAR_CONTOUR_COLOR_CHANGE_OFFSET = 5
PILLAR_BOTTOM_CONTACT_MAX_Y_DIFFERENCE = 5
HISTOGRAM_PEAK_POINT_COLOR = (82, 68, 240)  # 빨간색 (#F04452, BGR)
HISTOGRAM_MERGE_POINT_COLOR = (0, 146, 255)  # 주황색 (#FF9200, BGR)
HISTOGRAM_REMAINDER_POINT_COLOR = (246, 130, 49)  # 파란색 (#3182F6, BGR)
CENTER_SPLIT_LINE_COLOR = (180, 180, 180)
CUT_LINE_EXTENSION_COLOR = (112, 112, 112)

@dataclass
class ContourMeasurement:
    """컨투어 하나의 4면 첫 접점, 절대 극단점 및 보조 극단점 정보."""

    bounding_rect: Tuple[int, int, int, int]
    top_points: List[Tuple[int, int]]
    bottom_points: List[Tuple[int, int]]
    left_points: List[Tuple[int, int]]
    right_points: List[Tuple[int, int]]

    # 1. 절대 극단점 (4개)
    top_extreme: Tuple[int, int] = None
    bottom_extreme: Tuple[int, int] = None
    left_extreme: Tuple[int, int] = None
    right_extreme: Tuple[int, int] = None

    # 2. 보조 극단점 (2px 이내, 30px 이상 떨어진 점, 4개)
    top_secondary: Tuple[int, int] = None
    bottom_secondary: Tuple[int, int] = None
    left_secondary: Tuple[int, int] = None
    right_secondary: Tuple[int, int] = None


@dataclass
class DetectionResult:
    """B 페이지 컨투어 분석 결과."""

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


def to_grayscale(image):
    """이미지를 그레이스케일로 변환한다."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image):
    """미리보기에 사용할 이미지를 BGR 3채널로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def to_bgra(image):
    """투명 Preview에 사용할 이미지를 BGRA 4채널로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image.copy()


def find_b_contours(image_b):
    """그레이 스케일, 이진화, 컨투어링 작업."""
    gray = to_grayscale(image_b) # 그레이스케일
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU,) # 이진화
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,) # 컨투어링 작업

    # 범위내 컨투어 검출
    filtered_contours = [
        con for con in contours
        if MIN_CONTOUR_AREA <= cv2.contourArea(con) <= MAX_CONTOUR_AREA
    ]

    return DetectionResult(contours=filtered_contours)


def get_contour_box_center_x(contours, image_width):
    """검출된 컨투어 박스 전체의 가로 중앙 x 좌표를 구한다."""
    if not contours:
        return image_width // 2

    boxes = [cv2.boundingRect(contour) for contour in contours]
    left_x = min(x for x, _, _, _ in boxes)
    right_x = max(x + width - 1 for x, _, width, _ in boxes)
    return (left_x + right_x) // 2


def find_first_contact_points(contour, image_shape):
    """
    ## 컨투어의 상하좌우 접점 찾기
    1. 컨투어를 그린다.
    2. 전체 또는 균등 샘플 위치에서 상하좌우 첫 접점을 수집한다.
    3. 각 포인트에 대해 가장 가까운 절대 극단점(4개)을 찾는다.  
    
    """

    image_height, image_width = image_shape
    x, y, width, height = cv2.boundingRect(contour) # 컨투어의 x,y, width, height를 구함

    # 채워진 내부 면적이 아니라 실제 외곽선만 사용한다. 히스토그램과 기준선은
    # 내부 픽셀이 아닌 각 방향에서 외곽선을 처음 만나는 지점만 사용한다.
    boundary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(
        boundary_mask,
        [contour],
        -1,
        255,
        thickness=0,
        lineType=cv2.LINE_8,
    )

    # 1. 접점 수집: 전체 분포 또는 기존 균등 샘플
    top_points = []
    bottom_points = []
    if CONTACT_POINT_MODE == "full":
        x_positions = range(x, x + width)
        y_positions = range(y, y + height)
    elif CONTACT_POINT_MODE == "sampled":
        x_positions = np.linspace(
            x, x + width - 1, TOP_BOTTOM_SAMPLE_COUNT, dtype=int
        )
        y_positions = np.linspace(
            y, y + height - 1, LEFT_RIGHT_SAMPLE_COUNT, dtype=int
        )
    else:
        raise ValueError(f"지원하지 않는 접점 수집 방식: {CONTACT_POINT_MODE}")


    # 상면 하면 접점 중에
    for point_x in x_positions:
        contact_y_positions = np.where(boundary_mask[:, point_x] == 255)[0] # 고정된 x 좌표에 색상이 255인 y좌표의 배열
        if len(contact_y_positions) == 0: # 만약 y좌표가 존재하지 않으면 pass
            continue

        top_points.append((int(point_x), int(contact_y_positions[0]))) # 가장 상위 좌표
        bottom_points.append((int(point_x), int(contact_y_positions[-1]))) # 가장 하위 좌표

    left_points = []
    right_points = []
    for point_y in y_positions:
        contact_x_positions = np.where(boundary_mask[point_y, :] == 255)[0]
        if len(contact_x_positions) == 0:
            continue

        left_points.append((int(contact_x_positions[0]), int(point_y)))
        right_points.append((int(contact_x_positions[-1]), int(point_y)))

    # 2. 절대 극단점 추출 로직 (기존 4개)
    pts = contour[:, 0, :]  # (N, 2) 형태의 좌표 배열

    top_idx = pts[:, 1].argmin()
    bottom_idx = pts[:, 1].argmax()
    left_idx = pts[:, 0].argmin()
    right_idx = pts[:, 0].argmax()

    top_extreme = tuple(pts[top_idx])
    bottom_extreme = tuple(pts[bottom_idx])
    left_extreme = tuple(pts[left_idx])
    right_extreme = tuple(pts[right_idx])

    # 3. 보조 극단점 추출 로직 (정도 차이 3px 이내, 50px 이상 떨어진 점)
    def find_secondary_point(candidates_mask, extreme_pt, min_distance=MIN_DISTANCE):
        candidate_indices = np.where(candidates_mask)[0]
        valid_pts = []
        for idx in candidate_indices:
            pt = pts[idx]
            dist = np.linalg.norm(pt - np.array(extreme_pt))
            if dist >= min_distance:
                valid_pts.append((dist, tuple(pt)))

        if not valid_pts:
            if len(candidate_indices) > 0:
                dists = [np.linalg.norm(pts[i] - np.array(extreme_pt)) for i in candidate_indices]
                max_i = np.argmax(dists)
                return tuple(pts[candidate_indices[max_i]])
            return None

        valid_pts.sort(key=lambda x: x[0], reverse=True)
        return valid_pts[0][1]


    # PIXEL 정도 차이
    top_secondary = find_secondary_point(pts[:, 1] <= pts[top_idx, 1] + EXTREME_SENCODARY_DIFF, top_extreme, min_distance=MIN_DISTANCE)
    bottom_secondary = find_secondary_point(pts[:, 1] >= pts[bottom_idx, 1] - EXTREME_SENCODARY_DIFF, bottom_extreme, min_distance=MIN_DISTANCE)
    left_secondary = find_secondary_point(pts[:, 0] <= pts[left_idx, 0] + EXTREME_SENCODARY_DIFF, left_extreme, min_distance=MIN_DISTANCE)
    right_secondary = find_secondary_point(pts[:, 0] >= pts[right_idx, 0] - EXTREME_SENCODARY_DIFF, right_extreme, min_distance=MIN_DISTANCE)

    return ContourMeasurement(
        bounding_rect=(x, y, width, height),
        top_points=top_points,
        bottom_points=bottom_points,
        left_points=left_points,
        right_points=right_points,
        top_extreme=top_extreme,
        bottom_extreme=bottom_extreme,
        left_extreme=left_extreme,
        right_extreme=right_extreme,
        top_secondary=top_secondary,
        bottom_secondary=bottom_secondary,
        left_secondary=left_secondary,
        right_secondary=right_secondary,
    )


def stack_preview_tiles(tiles):
    """여러 BGRA Preview 타일을 투명한 세로 Preview로 합친다."""

    if not tiles:
        return None
    if len(tiles) == 1:
        return tiles[0]

    gap = 8
    preview_width = max(tile.shape[1] for tile in tiles)
    preview_height = sum(tile.shape[0] for tile in tiles) + gap * (len(tiles) - 1)
    preview = np.zeros((preview_height, preview_width, 4), dtype=np.uint8)

    offset_y = 0
    for tile in tiles:
        tile_height, tile_width = tile.shape[:2]
        offset_x = (preview_width - tile_width) // 2
        preview[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width] = tile
        offset_y += tile_height + gap
    return preview


def approximate_contour_polygon(contour):
    """컨투어를 둘레 길이 비율 기반의 다각형으로 근사한다."""

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    polygon = cv2.approxPolyDP(
        contour, APPROX_POLYGON_EPSILON_RATIO * perimeter, True
    )
    return polygon if len(polygon) >= 3 else None


def get_line_intersection(first_start, first_end, second_start, second_end):
    """두 기준선의 교점을 반환하고, 평행하면 ``None``을 반환한다."""

    first_start = np.asarray(first_start, dtype=float)
    first_direction = np.asarray(first_end, dtype=float) - first_start
    second_start = np.asarray(second_start, dtype=float)
    second_direction = np.asarray(second_end, dtype=float) - second_start
    denominator = np.cross(first_direction, second_direction)
    if np.isclose(denominator, 0):
        return None

    distance = np.cross(second_start - first_start, second_direction) / denominator
    return tuple(np.rint(first_start + distance * first_direction).astype(int))


def get_densest_band_points(points, coordinate_index, tolerance):
    """최빈 좌표 주변의 가장 밀집한 점 띠를 반환한다."""

    samples = np.asarray(points, dtype=np.float32)
    if len(samples) < 2:
        return samples

    coordinates = np.rint(samples[:, coordinate_index]).astype(int)
    values, counts = np.unique(coordinates, return_counts=True)
    most_frequent_values = values[counts == counts.max()]
    median = np.median(coordinates)
    center = most_frequent_values[
        np.argmin(np.abs(most_frequent_values - median))
    ]
    return samples[np.abs(coordinates - center) <= tolerance]


def get_primary_contact_reference_point(points, coordinate_index, use_minimum):
    """첫 접점 중 가장 먼저 닿는 최솟값 또는 최댓값의 실제 대표점을 고른다."""

    if not points:
        return None

    coordinates = [point[coordinate_index] for point in points]
    coordinate = min(coordinates) if use_minimum else max(coordinates)
    candidates = [
        point for point in points if point[coordinate_index] == coordinate
    ]
    other_coordinate_index = 1 - coordinate_index
    median = np.median([point[other_coordinate_index] for point in candidates])
    return min(
        candidates,
        key=lambda point: (
            abs(point[other_coordinate_index] - median),
            point[other_coordinate_index],
        ),
    )


def draw_axis_aligned_reference_line(img, point, coordinate_index, color, thickness=1):
    """상·하는 수평선, 좌·우는 수직선을 대표점에서 이미지 끝까지 그린다."""

    if point is None:
        return

    image_height, image_width = img.shape[:2]
    x, y = point
    if coordinate_index == 1:
        cv2.line(img, (0, y), (image_width - 1, y), color, thickness)
    else:
        cv2.line(img, (x, 0), (x, image_height - 1), color, thickness)


def get_spaced_reference_point(points, reference_point, coordinate_index, min_distance):
    """같은 기준 좌표에서 충분히 떨어진 실제 첫 접점 하나를 반환한다."""

    if reference_point is None:
        return None

    other_coordinate_index = 1 - coordinate_index
    candidates = [
        point
        for point in points
        if point[coordinate_index] == reference_point[coordinate_index]
        and abs(point[other_coordinate_index] - reference_point[other_coordinate_index])
        >= min_distance
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda point: abs(
            point[other_coordinate_index] - reference_point[other_coordinate_index]
        ),
    )


def fit_line_from_contact_points(points, orientation, use_densest_band=False):
    """방향별 중앙값 이상점을 제외한 접점으로 기준선을 피팅한다."""

    if len(points) < 2:
        return None

    samples = np.asarray(points, dtype=np.float32)
    coordinate_index = 1 if orientation == "horizontal" else 0
    if use_densest_band:
        samples = get_densest_band_points(
            samples, coordinate_index, TOP_DENSE_BAND_TOLERANCE
        )
        if len(samples) < 2:
            return None

    coordinate_values = samples[:, coordinate_index]
    median = np.median(coordinate_values)
    median_absolute_deviation = np.median(np.abs(coordinate_values - median))
    tolerance = max(2.0, 3.0 * median_absolute_deviation)
    inliers = samples[np.abs(coordinate_values - median) <= tolerance]
    if len(inliers) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        inliers.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
    )
    direction = np.array([float(vx), float(vy)])
    if np.linalg.norm(direction) == 0:
        return None
    origin = np.array([float(x0), float(y0)])
    return tuple(origin - direction), tuple(origin + direction)


def get_extreme_pair_lines(measurement):
    """기존 극점·보조점 쌍으로 만든 네 기준선을 반환한다."""

    return (
        (measurement.top_extreme, measurement.top_secondary),
        (measurement.bottom_extreme, measurement.bottom_secondary),
        (measurement.left_extreme, measurement.left_secondary),
        (measurement.right_extreme, measurement.right_secondary),
    )


def get_robust_contact_lines(measurement):
    """상·하·좌·우 접점 집합으로 피팅한 네 기준선을 반환한다."""

    return (
        fit_line_from_contact_points(
            measurement.top_points, "horizontal", use_densest_band=True
        ),
        fit_line_from_contact_points(measurement.bottom_points, "horizontal"),
        fit_line_from_contact_points(measurement.left_points, "vertical"),
        fit_line_from_contact_points(measurement.right_points, "vertical"),
    )


def get_primary_axis_quadrilateral(measurement):
    """B 이미지의 대표 수평·수직 기준선 교점으로 만든 축 정렬 사각형을 반환한다."""

    top_point = get_primary_contact_reference_point(
        measurement.top_points, 1, use_minimum=True
    )
    bottom_point = get_primary_contact_reference_point(
        measurement.bottom_points, 1, use_minimum=False
    )
    left_point = get_primary_contact_reference_point(
        measurement.left_points, 0, use_minimum=True
    )
    right_point = get_primary_contact_reference_point(
        measurement.right_points, 0, use_minimum=False
    )
    if any(point is None for point in (top_point, bottom_point, left_point, right_point)):
        return None

    top_y = top_point[1]
    bottom_y = bottom_point[1]
    left_x = left_point[0]
    right_x = right_point[0]
    if top_y >= bottom_y or left_x >= right_x:
        return None

    return np.asarray(
        (
            (left_x, top_y),
            (right_x, top_y),
            (right_x, bottom_y),
            (left_x, bottom_y),
        ),
        dtype=np.int32,
    ).reshape(-1, 1, 2)


def is_valid_line_quadrilateral(corners, measurement):
    """교점 사각형이 컨투어 근방의 볼록한 도형인지 확인한다."""

    if any(corner is None for corner in corners):
        return False

    polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 0:
        return False

    x, y, width, height = measurement.bounding_rect
    margin_x = max(12, round(width * 0.2))
    margin_y = max(12, round(height * 0.2))
    points = polygon[:, 0, :]
    return (
        points[:, 0].min() >= x - margin_x
        and points[:, 0].max() <= x + width - 1 + margin_x
        and points[:, 1].min() >= y - margin_y
        and points[:, 1].max() <= y + height - 1 + margin_y
    )


def get_line_quadrilateral(measurement):
    """상·하·좌·우 기준선의 교점 네 개를 시계 방향으로 반환한다."""

    if LINE_QUADRILATERAL_METHOD == "primary_axis":
        return get_primary_axis_quadrilateral(measurement)
    if LINE_QUADRILATERAL_METHOD == "robust_contacts":
        lines = get_robust_contact_lines(measurement)
    elif LINE_QUADRILATERAL_METHOD == "extreme_pairs":
        lines = get_extreme_pair_lines(measurement)
    else:
        raise ValueError(
            f"지원하지 않는 기준선 사각형 방식: {LINE_QUADRILATERAL_METHOD}"
        )

    if any(line is None or any(point is None for point in line) for line in lines):
        return None
    top, bottom, left, right = lines

    corners = (
        get_line_intersection(*top, *left),
        get_line_intersection(*top, *right),
        get_line_intersection(*bottom, *right),
        get_line_intersection(*bottom, *left),
    )
    if any(corner is None for corner in corners):
        return None
    if (
        LINE_QUADRILATERAL_METHOD == "robust_contacts"
        and not is_valid_line_quadrilateral(corners, measurement)
    ):
        return None
    return np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)


def expand_quadrilateral_by_side(polygon, margin):
    """사각형의 상·하·좌·우 변을 좌표축 방향으로 바깥쪽 확장한다."""

    if margin <= 0:
        return polygon

    points = polygon[:, 0, :].astype(float)
    if len(points) != 4:
        return polygon

    top_left, top_right, bottom_right, bottom_left = points
    top = (top_left + (0, -margin), top_right + (0, -margin))
    bottom = (bottom_right + (0, margin), bottom_left + (0, margin))
    left = (bottom_left + (-margin, 0), top_left + (-margin, 0))
    right = (top_right + (margin, 0), bottom_right + (margin, 0))

    corners = (
        get_line_intersection(*top, *left),
        get_line_intersection(*top, *right),
        get_line_intersection(*bottom, *right),
        get_line_intersection(*bottom, *left),
    )
    if any(corner is None for corner in corners):
        return polygon
    return np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)


def get_preview_mask_shape(contour, measurement, mask_mode):
    """지정한 방식에 맞는 Preview 마스크 도형을 반환한다."""
    if mask_mode == "contour":
        return contour
    if mask_mode == "approx_polygon":
        return approximate_contour_polygon(contour)
    if mask_mode == "line_quadrilateral":
        quadrilateral = get_line_quadrilateral(measurement)
        if quadrilateral is None:
            return contour
        return expand_quadrilateral_by_side(
            quadrilateral, SOURCE_PREVIEW_OUTER_MARGIN
        )
    raise ValueError(f"지원하지 않는 Preview 마스크 방식: {mask_mode}")


def expand_polygon_to_side_reference_lines(
    polygon, left_reference_line, right_reference_line
):
    """다각형의 좌·우 변을 컨투어 좌·우 기준선까지 확장한다."""

    if len(left_reference_line) != 2 or len(right_reference_line) != 2:
        return polygon
    points = polygon[:, 0, :].astype(float)
    if len(points) != 4:
        return polygon

    top_left, top_right, bottom_right, bottom_left = points
    top_line = (top_left, top_right)
    bottom_line = (bottom_right, bottom_left)
    corners = (
        get_line_intersection(*top_line, *left_reference_line),
        get_line_intersection(*top_line, *right_reference_line),
        get_line_intersection(*bottom_line, *right_reference_line),
        get_line_intersection(*bottom_line, *left_reference_line),
    )
    if any(corner is None for corner in corners):
        return polygon
    expanded = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    return expanded if cv2.contourArea(expanded) > 0 else polygon


def get_polygon_crop_bounds(polygon, image_shape):
    """근사 다각형을 포함하는 최소 Crop 범위를 반환한다."""

    image_height, image_width = image_shape
    points = polygon[:, 0, :]
    left = max(0, int(points[:, 0].min()))
    top = max(0, int(points[:, 1].min()))
    right = min(image_width, int(points[:, 0].max()) + 1)
    bottom = min(image_height, int(points[:, 1].max()) + 1)
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def find_top_pillar_reference_points(image):
    """상단 양 끝에서 중앙으로 처음 만나는 큰 밝기 변화점을 찾는다.

    좌측은 ``(0, 0)``에서 오른쪽으로, 우측은 실제 마지막 픽셀
    ``(width - 1, 0)``에서 왼쪽으로 이동한다. 상단 행에서 두 변화점을
    찾지 못하면 y를 증가시켜 아래 행에서도 같은 탐색을 반복한다. 각 방향에서
    큰 변화가 연속으로 확인되는 첫 좌표만 반환해 가장자리 노이즈를 제외한다.
    """

    grayscale = to_grayscale(image)
    image_width = image.shape[1]
    image_height = image.shape[0]
    if image_width < 2 or image_height < SOURCE_PREVIEW_TOP_SCAN_HEIGHT:
        return None

    edge_window = SOURCE_PREVIEW_EDGE_CHANGE_WINDOW
    min_run = SOURCE_PREVIEW_HORIZONTAL_CHANGE_MIN_RUN
    if image_width <= edge_window * 2 or image_width < min_run:
        return None
    midpoint_x = image_width // 2
    if midpoint_x - edge_window < min_run:
        return None

    def first_persistent_change(profile, scan_x, reference_offset):
        """스캔 순서의 x 좌표 중 임계값 초과가 연속되는 첫 x를 반환한다."""

        candidates = list(scan_x)
        changes = np.asarray(
            [
                abs(
                    int(profile[x])
                    - int(profile[x + reference_offset])
                )
                for x in candidates
            ],
            dtype=np.int16,
        )
        changed = changes >= SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD
        persistent = np.convolve(
            changed.astype(np.int16),
            np.ones(min_run, dtype=np.int16),
            mode="valid",
        ) >= min_run
        if not np.any(persistent):
            return None
        return candidates[int(np.flatnonzero(persistent)[0])]

    # y=0에서 먼저 확인하고, 변화점이 없으면 아래 행으로 한 줄씩 이동한다.
    # 각 행은 연속된 3줄의 중앙값으로 만들어 센서 노이즈를 줄인다.
    for scan_y in range(image_height - SOURCE_PREVIEW_TOP_SCAN_HEIGHT + 1):
        profile = np.median(
            grayscale[
                scan_y : scan_y + SOURCE_PREVIEW_TOP_SCAN_HEIGHT, :
            ],
            axis=0,
        ).astype(np.int16)
        # 왼쪽은 x=0에서 증가하는 방향, 오른쪽은 x=width-1에서 감소하는 방향으로
        # 중앙선 바로 전까지만 탐색한다. 비교 대상은 각각 시작점 쪽 픽셀이다.
        left_x = first_persistent_change(
            profile, range(edge_window, midpoint_x), -edge_window
        )
        right_x = first_persistent_change(
            profile,
            range(image_width - 1 - edge_window, midpoint_x - 1, -1),
            edge_window,
        )
        if left_x is not None and right_x is not None and left_x < right_x:
            return ((left_x, scan_y), (right_x, scan_y))

    return None


def find_downward_color_change_points(image, reference_points):
    """두 기준점의 y 차이가 허용 범위인 아래 방향 변화점 쌍을 찾는다."""

    if reference_points is None:
        return ()

    grayscale = to_grayscale(image)
    candidate_points = []
    for point_x, point_y in reference_points:
        vertical_profile = grayscale[point_y:, point_x].astype(np.int16)
        reference_brightness = vertical_profile[0]
        color_changed = np.abs(
            vertical_profile - reference_brightness
        ) >= SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD
        persistent_change = np.convolve(
            color_changed.astype(np.int16),
            np.ones(SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN, dtype=np.int16),
            mode="valid",
        ) >= SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN
        transition_y = np.flatnonzero(persistent_change)
        if len(transition_y) == 0:
            candidate_points.append([])
            continue

        # 같은 색 변화 구간에서 연속으로 나온 y는 첫 지점 하나만 남긴다.
        transition_starts = transition_y[
            np.r_[True, np.diff(transition_y) > 1]
        ]
        candidate_points.append(
            [(point_x, int(point_y + candidate_y)) for candidate_y in transition_starts]
        )

    if len(candidate_points) != 2:
        return ()

    detected_point_groups = [points for points in candidate_points if points]
    if len(detected_point_groups) == 1:
        # 한쪽 점만 검출되면 그 첫 변화점을 수평선 기준으로 사용한다.
        return (detected_point_groups[0][0],)
    if len(detected_point_groups) != 2:
        return ()

    valid_pairs = [
        (left_point, right_point)
        for left_point in candidate_points[0]
        for right_point in candidate_points[1]
        if abs(left_point[1] - right_point[1])
        < SOURCE_PREVIEW_INVALID_Y_DIFFERENCE
    ]
    if not valid_pairs:
        return ()

    # 가장 위에서 처음 만나는, y 차이가 가장 작은 쌍을 선택한다.
    return tuple(
        min(
            valid_pairs,
            key=lambda pair: (
                max(pair[0][1], pair[1][1]),
                abs(pair[0][1] - pair[1][1]),
            ),
        )
    )


def get_pillar_outer_reference_points(image, reference_points):
    """바깥쪽 x 위치에서 밝기 250~255인 첫 기준점을 반환한다."""

    if reference_points is None or len(reference_points) != 2:
        return ()

    grayscale = to_grayscale(image)
    image_height, image_width = grayscale.shape[:2]
    left_point, right_point = sorted(reference_points, key=lambda point: point[0])
    x_coordinates = (
        max(0, left_point[0] - PILLAR_REFERENCE_POINT_OUTER_OFFSET),
        min(image_width - 1, right_point[0] + PILLAR_REFERENCE_POINT_OUTER_OFFSET),
    )

    bright_points = []
    for point, point_x in zip((left_point, right_point), x_coordinates):
        start_y = min(max(0, point[1]), image_height - 1)
        vertical_profile = grayscale[start_y:, point_x]
        bright_y = np.flatnonzero(
            (vertical_profile >= PILLAR_REFERENCE_BRIGHTNESS_MIN)
            & (vertical_profile <= PILLAR_REFERENCE_BRIGHTNESS_MAX)
        )
        if len(bright_y) == 0:
            return ()
        bright_points.append((point_x, int(start_y + bright_y[0])))

    return tuple(bright_points)


def find_contour_contact_color_change_points(
    image, reference_points, contour_outline
):
    """하늘색 점 아래의 컨투어 접점에서 A 페이지 색 변화를 찾는다."""

    if not reference_points or contour_outline is None:
        return ()

    grayscale = to_grayscale(image)
    image_height, image_width = grayscale.shape[:2]
    if contour_outline.shape[:2] != grayscale.shape[:2]:
        raise ValueError("컨투어 윤곽선과 A 페이지의 크기가 일치하지 않습니다.")

    change_points = []
    for point_x, point_y in reference_points:
        if not 0 <= point_x < image_width:
            continue
        start_y = min(
            max(0, point_y + PILLAR_CONTOUR_CONTACT_SCAN_OFFSET), image_height
        )
        contact_y = np.flatnonzero(contour_outline[start_y:, point_x] > 0)
        if len(contact_y) == 0:
            continue
        contact_y = int(start_y + contact_y[0])
        upper_y = max(0, contact_y - PILLAR_CONTOUR_COLOR_CHANGE_OFFSET)
        lower_y = min(
            image_height - 1, contact_y + PILLAR_CONTOUR_COLOR_CHANGE_OFFSET
        )
        if (
            abs(int(grayscale[lower_y, point_x]) - int(grayscale[upper_y, point_x]))
            >= SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD
        ):
            change_points.append((point_x, contact_y))

    # 양쪽 하늘색 점이 있지만 연한 하늘색 점이 한쪽만 검출되면, 반대편에는
    # 검출된 점과 같은 y 좌표를 사용해 표시·세로선 기준을 맞춘다.
    if len(reference_points) == 2 and len(change_points) == 1:
        detected_x, detected_y = change_points[0]
        missing_points = [
            point for point in reference_points if point[0] != detected_x
        ]
        if len(missing_points) == 1:
            change_points.append((missing_points[0][0], detected_y))

    return tuple(sorted(change_points, key=lambda point: point[0]))


def find_contour_contact_points(reference_points, contour_outline):
    """하늘색 점 아래에서 처음 만나는 컨투어 윤곽 접점을 반환한다."""

    if not reference_points or contour_outline is None:
        return ()

    image_height, image_width = contour_outline.shape[:2]
    contact_points = []
    for point_x, point_y in reference_points:
        if not 0 <= point_x < image_width:
            continue
        start_y = min(
            max(0, point_y + PILLAR_CONTOUR_CONTACT_SCAN_OFFSET), image_height
        )
        contact_y = np.flatnonzero(contour_outline[start_y:, point_x] > 0)
        if len(contact_y) > 0:
            contact_points.append((point_x, int(start_y + contact_y[0])))

    if len(reference_points) == 2 and len(contact_points) == 1:
        detected_x, detected_y = contact_points[0]
        missing_points = [
            point for point in reference_points if point[0] != detected_x
        ]
        if len(missing_points) == 1:
            contact_points.append((missing_points[0][0], detected_y))
    return tuple(sorted(contact_points, key=lambda point: point[0]))


def get_first_contact_side_lines(measurements, image_height):
    """컨투어 첫 접점의 최소·최대 x로 좌·우 세로 크롭선을 만든다."""

    left_points = [
        point for measurement in measurements for point in measurement.left_points
    ]
    right_points = [
        point for measurement in measurements for point in measurement.right_points
    ]
    if not left_points or not right_points:
        return (), ()

    left_x = min(point[0] for point in left_points)
    right_x = max(point[0] for point in right_points)
    return (
        ((left_x, 0), (left_x, image_height - 1)),
        ((right_x, 0), (right_x, image_height - 1)),
    )


def get_pillar_horizontal_cut_line(points, image_width):
    """기둥 기준 점을 잇는 컷선을 반환한다."""

    if not points:
        return None
    if len(points) == 1:
        line_y = points[0][1]
        return ((0, line_y), (image_width - 1, line_y))
    return tuple(sorted(points, key=lambda point: point[0]))


def get_pillar_bottom_cut_line(points, image_width):
    """좌·우 하단 접점 차이에 따라 연결선 또는 수평 컷선을 반환한다."""

    line = get_pillar_horizontal_cut_line(points, image_width)
    if line is None or len(points) != 2:
        return line
    left_point, right_point = line
    if abs(left_point[1] - right_point[1]) <= PILLAR_BOTTOM_CONTACT_MAX_Y_DIFFERENCE:
        return line
    bottom_y = max(left_point[1], right_point[1])
    return ((0, bottom_y), (image_width - 1, bottom_y))


def find_left_right_contour_reference_points(contour_outline):
    """상·하 반에서 윤곽의 가장 바깥 좌·우 접점을 찾는다."""

    image_height, _ = contour_outline.shape[:2]
    midpoint_y = image_height // 2
    side_points = {"left": [], "right": []}
    for start_y, end_y in ((0, midpoint_y), (midpoint_y, image_height)):
        y_coordinates, x_coordinates = np.where(
            contour_outline[start_y:end_y] > 0
        )
        if len(x_coordinates) == 0:
            continue
        y_coordinates = y_coordinates + start_y
        center_y = (start_y + end_y - 1) / 2
        for side_name, contact_x in (
            ("left", int(x_coordinates.min())),
            ("right", int(x_coordinates.max())),
        ):
            candidates_y = y_coordinates[x_coordinates == contact_x]
            contact_y = int(
                min(candidates_y, key=lambda point_y: abs(point_y - center_y))
            )
            side_points[side_name].append((contact_x, contact_y))
    return tuple(side_points["left"]), tuple(side_points["right"])


def draw_left_right_contour_reference_lines(image, left_points, right_points):
    """A 페이지에 컨투어 좌·우 접점과 세로 방향 기준선을 표시한다."""

    preview = to_bgr(image)
    for points in (left_points, right_points):
        if len(points) == 2:
            cv2.line(
                preview, points[0], points[1], PILLAR_DOWNWARD_COLOR, 1, cv2.LINE_AA
            )
        for point in points:
            cv2.circle(
                preview,
                point,
                PILLAR_POINT_RADIUS,
                PILLAR_DOWNWARD_COLOR,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )
    return preview


def draw_top_pillar_reference_points(
    image, reference_points, downward_points=(), downward_bright_points=()
):
    """원본 A 페이지에 좌우 색 변화 기준점과 필요한 수평선을 표시한다."""

    preview = to_bgr(image)
    if reference_points is None:
        return preview

    # 변화 좌표보다 바깥쪽 x 위치에서 밝기 250~255인 지점에 회색 점을 표시한다.
    # 좌측: x₁ - 5px, 우측: x₂ + 5px (이미지 범위를 벗어나지 않게 제한)
    display_points = get_pillar_outer_reference_points(
        preview, reference_points
    )
    for point in display_points:
        cv2.circle(
            preview,
            point,
            PILLAR_POINT_RADIUS,
            PILLAR_REFERENCE_POINT_COLOR,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_AA,
        )

    if downward_points:
        if len(downward_points) == 1:
            point = downward_points[0]
            left_point = (0, point[1])
            right_point = (preview.shape[1] - 1, point[1])
        else:
            left_point, right_point = sorted(downward_points, key=lambda point: point[0])
        line_y = round((left_point[1] + right_point[1]) / 2)
        cv2.line(
            preview,
            (left_point[0], line_y),
            (right_point[0], line_y),
            PILLAR_DOWNWARD_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # 선을 먼저 그린 뒤 검출된 하늘색 점을 덮어 그린다.
        for point in downward_points:
            cv2.circle(
                preview,
                point,
                PILLAR_POINT_RADIUS,
                PILLAR_DOWNWARD_COLOR,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )

    # 하늘색 점 아래의 컨투어 접점에서 색이 변하는 지점은 더 연한 하늘색으로 표시한다.
    downward_points_by_x = {point[0]: point for point in downward_points}
    bottom_cut_line = get_pillar_horizontal_cut_line(
        downward_bright_points, preview.shape[1]
    )
    if bottom_cut_line is not None:
        cv2.line(
            preview,
            *bottom_cut_line,
            PILLAR_DOWNWARD_BRIGHT_POINT_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    for point in downward_bright_points:
        source_point = downward_points_by_x.get(point[0])
        if source_point is not None:
            cv2.line(
                preview,
                source_point,
                point,
                PILLAR_DOWNWARD_BRIGHT_POINT_COLOR,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        cv2.circle(
            preview,
            point,
            PILLAR_POINT_RADIUS,
            PILLAR_DOWNWARD_BRIGHT_POINT_COLOR,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_AA,
        )
    # 세로선이 시작 하늘색 점을 덮지 않도록 마지막에 다시 표시한다.
    for point in downward_points:
        cv2.circle(
            preview,
            point,
            PILLAR_POINT_RADIUS,
            PILLAR_DOWNWARD_COLOR,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_AA,
        )
    return preview


def get_cut_line_y_coordinates(cut_line, x_coordinates):
    """좌·우 컷 좌표를 잇는 선의 각 x 위치 y 좌표를 반환한다."""
    if cut_line is None:
        return None

    (left_x, left_y), (right_x, right_y) = cut_line
    if left_x == right_x:
        return np.full_like(x_coordinates, left_y, dtype=np.float64)
    return np.interp(x_coordinates, (left_x, right_x), (left_y, right_y))


def apply_side_cutting_mask(mask, left, top, top_cut_line, bottom_cut_line):
    """좌·우 Merge 컷 좌표를 이은 선을 기준으로 마스크의 상·하를 제거한다."""
    x_coordinates = np.arange(left, left + mask.shape[1])
    y_coordinates = np.arange(top, top + mask.shape[0])[:, None]

    top_cut_y = get_cut_line_y_coordinates(top_cut_line, x_coordinates)
    if top_cut_y is not None:
        mask[y_coordinates < top_cut_y[None, :]] = 0

    bottom_cut_y = get_cut_line_y_coordinates(bottom_cut_line, x_coordinates)
    if bottom_cut_y is not None:
        mask[y_coordinates > bottom_cut_y[None, :]] = 0


def create_polygon_preview(
    image,
    contours,
    measurements,
    mask_mode,
    top_cut_line=None,
    bottom_cut_line=None,
    left_reference_line=(),
    right_reference_line=(),
):
    """선택한 마스크 도형 내부의 원본 픽셀만 남긴 투명 Preview를 만든다."""

    tiles = []
    for contour, measurement in zip(contours, measurements):
        polygon = get_preview_mask_shape(contour, measurement, mask_mode)
        if polygon is None:
            continue
        polygon = expand_polygon_to_side_reference_lines(
            polygon, left_reference_line, right_reference_line
        )

        bounds = get_polygon_crop_bounds(polygon, image.shape[:2])
        if bounds is None:
            continue

        left, top, right, bottom = bounds
        tile = to_bgra(image)[top:bottom, left:right].copy()
        mask = np.zeros(tile.shape[:2], dtype=np.uint8)
        local_polygon = polygon.copy()
        local_polygon[:, 0, 0] -= left
        local_polygon[:, 0, 1] -= top
        cv2.fillPoly(mask, [local_polygon], 255, lineType=cv2.LINE_AA)
        apply_side_cutting_mask(
            mask, left, top, top_cut_line, bottom_cut_line
        )
        if not np.any(mask):
            continue
        tile[:, :, 3] = cv2.bitwise_and(tile[:, :, 3], mask)
        tiles.append(tile)
    return stack_preview_tiles(tiles)


def add_preview_caption(preview, caption):
    """A/B 비교 미리보기 위에 페이지 구분 제목을 붙인다."""

    if preview is None or preview.size == 0:
        return None

    caption_height = 30
    height, width = preview.shape[:2]
    labeled = np.zeros((height + caption_height, width, 4), dtype=np.uint8)
    labeled[:caption_height, :, :3] = (245, 245, 245)
    labeled[:caption_height, :, 3] = 255
    labeled[caption_height:] = preview
    cv2.putText(
        labeled,
        caption,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def create_preview_comparison(left_preview, left_caption, right_preview, right_caption):
    """두 페이지의 같은 기준선 크롭 결과를 나란히 합친다."""

    left = add_preview_caption(left_preview, left_caption)
    right = add_preview_caption(right_preview, right_caption)
    if left is None:
        return right
    if right is None:
        return left

    gap = 12
    preview_height = max(left.shape[0], right.shape[0])
    preview_width = left.shape[1] + gap + right.shape[1]
    comparison = np.zeros((preview_height, preview_width, 4), dtype=np.uint8)
    comparison[: left.shape[0], : left.shape[1]] = left
    right_x = left.shape[1] + gap
    comparison[: right.shape[0], right_x : right_x + right.shape[1]] = right
    return comparison


def get_histogram_coordinate_groups(points):
    """지정 면의 히스토그램 빨간색·주황색 y 좌표를 반환한다."""
    coordinates = [point[1] for point in points]
    if not coordinates:
        return set(), set()

    values, counts = np.unique(coordinates, return_counts=True)
    max_count = int(max(counts))
    ratio_count = max_count * MAX_COUNT_RATIO
    peak_coordinates = values[counts == max_count]
    count_by_coordinate = dict(zip(values, counts))
    merge_coordinates = set()

    for peak_coordinate in peak_coordinates:
        peak_coordinate = int(peak_coordinate)
        for direction in (-1, 1):
            adjacent_coordinate = peak_coordinate + direction
            if int(count_by_coordinate.get(adjacent_coordinate, 0)) <= ratio_count:
                continue

            merge_coordinates.add(adjacent_coordinate)
            next_coordinate = adjacent_coordinate + direction
            if int(count_by_coordinate.get(next_coordinate, 0)) >= ratio_count:
                merge_coordinates.add(next_coordinate)

    peak_coordinates = {int(value) for value in peak_coordinates}
    return peak_coordinates, merge_coordinates - peak_coordinates


def get_side_points(measurements, point_attribute, midpoint_x):
    """상·하면 외곽 접점을 이미지 중앙 x 좌표를 기준으로 좌·우로 나눈다."""
    left_points, right_points = [], []
    for measurement in measurements:
        for point in getattr(measurement, point_attribute):
            (left_points if point[0] < midpoint_x else right_points).append(point)
    return left_points, right_points


def get_first_contact_band_points(points, use_maximum):
    """첫 접점에서 진행 방향으로 2px 범위 안의 접점만 반환한다."""
    if not points:
        return []
    first_contact_y = min(point[1] for point in points) if use_maximum else max(
        point[1] for point in points
    )
    if use_maximum:
        return [
            point
            for point in points
            if first_contact_y <= point[1] <= first_contact_y + FIRST_CONTACT_BAND_PIXELS
        ]
    return [
        point
        for point in points
        if first_contact_y - FIRST_CONTACT_BAND_PIXELS <= point[1] <= first_contact_y
    ]


def get_first_contact_candidate_groups(points, use_maximum):
    """첫 접점 범위의 후보에 더 빈번한 첫 접점을 주황 후보로 추가한다."""
    band_points = get_first_contact_band_points(points, use_maximum)
    peak_coordinates, merge_coordinates = get_histogram_coordinate_groups(band_points)
    if not band_points or not merge_coordinates:
        return band_points, peak_coordinates, merge_coordinates

    first_contact_y = min(point[1] for point in points) if use_maximum else max(
        point[1] for point in points
    )
    if first_contact_y in peak_coordinates or first_contact_y in merge_coordinates:
        return band_points, peak_coordinates, merge_coordinates

    count_by_coordinate = {}
    for _, coordinate in band_points:
        count_by_coordinate[coordinate] = count_by_coordinate.get(coordinate, 0) + 1
    first_contact_count = count_by_coordinate.get(first_contact_y, 0)
    largest_merge_count = max(
        count_by_coordinate.get(coordinate, 0) for coordinate in merge_coordinates
    )
    if first_contact_count > largest_merge_count:
        merge_coordinates = merge_coordinates | {first_contact_y}
    return band_points, peak_coordinates, merge_coordinates


def get_first_contact_density_log(
    measurements, point_attribute, midpoint_x, use_maximum, side_name
):
    """빨강·주황 후보 좌표가 좌·우에 고르게 있는지 기록한다."""
    points = [
        point
        for measurement in measurements
        for point in getattr(measurement, point_attribute)
    ]
    if not points:
        return None

    band_points, peak_coordinates, merge_coordinates = (
        get_first_contact_candidate_groups(points, use_maximum)
    )
    candidate_coordinates = peak_coordinates | merge_coordinates
    candidate_points = [
        point for point in band_points if point[1] in candidate_coordinates
    ]
    if not candidate_points:
        return None

    left_count = sum(point[0] < midpoint_x for point in candidate_points)
    right_count = len(candidate_points) - left_count
    minimum_x = min(point[0] for point in candidate_points)
    maximum_x = max(point[0] for point in candidate_points)

    if left_count == 0 or right_count == 0:
        distribution = "한쪽 집중"
    else:
        balance_ratio = min(left_count, right_count) / max(left_count, right_count)
        distribution = "좌우 균형" if balance_ratio >= 0.5 else "한쪽 치우침"

    return (
        f"[밀집도] {side_name} 빨강·주황 후보 y={sorted(candidate_coordinates)}: "
        f"{len(candidate_points)}개 · 좌 {left_count} / 우 {right_count} · "
        f"x={minimum_x}~{maximum_x} · {distribution}"
    )


def append_density_log_when_merge_exists(
    result, point_attribute, midpoint_x, use_maximum, side_name
):
    """주황 Merge 후보가 있을 때만 첫 접점 x 분포를 로그로 남긴다."""
    points = [
        point
        for measurement in result.measurements
        for point in getattr(measurement, point_attribute)
    ]
    _, _, merge_coordinates = get_first_contact_candidate_groups(
        points, use_maximum
    )
    if not merge_coordinates:
        return

    log_line = get_first_contact_density_log(
        result.measurements,
        point_attribute,
        midpoint_x,
        use_maximum,
        side_name,
    )
    if log_line:
        result.density_log_lines.append(log_line)
        print(log_line)


def get_concentrated_cut_line(
    measurements, point_attribute, image_width, use_maximum, side_name, midpoint_x=None
):
    """상·하면 첫 접점의 좌·우 분포에 맞는 특수 컷선을 만든다."""
    midpoint_x = image_width // 2 if midpoint_x is None else midpoint_x
    contact_points = [
        point
        for measurement in measurements
        for point in getattr(measurement, point_attribute)
    ]
    if not contact_points:
        return None

    band_points, peak_coordinates, merge_coordinates = (
        get_first_contact_candidate_groups(
            contact_points, use_maximum=use_maximum
        )
    )
    if not merge_coordinates:
        return None

    candidate_coordinates = peak_coordinates | merge_coordinates
    candidate_points = [
        point for point in band_points if point[1] in candidate_coordinates
    ]

    # 좌·우 밀집도 판단보다 먼저, 실제 첫 접점 방향의 끝 y가 전체 접점에서
    # 가장 많이 나타나면 해당 y를 바로 수평 컷으로 사용한다.
    # 후보 범위의 끝 y가 아니라 전체 접점의 끝 y를 기준으로 해야 한다.
    count_by_coordinate = {}
    for _, coordinate in contact_points:
        count_by_coordinate[coordinate] = count_by_coordinate.get(coordinate, 0) + 1
    extreme_y = (
        min(point[1] for point in contact_points)
        if use_maximum
        else max(point[1] for point in contact_points)
    )
    extreme_y_count = count_by_coordinate[extreme_y]
    extreme_name = "최상단" if use_maximum else "최하단"
    if extreme_y_count == max(count_by_coordinate.values()):
        marker = get_cut_marker_point(candidate_points, extreme_y)
        horizontal_line = ((0, extreme_y), (image_width - 1, extreme_y))
        return {
            "line": horizontal_line,
            "extended_line": horizontal_line,
            "markers": ((marker, extreme_y in peak_coordinates),),
            "concentration_side": f"{extreme_name} 최빈",
            "density_log_line": (
                f"[밀집도] {side_name} {extreme_name} y={extreme_y}가 전체 접점 중 최빈 "
                f"({extreme_y_count}개): 수평 컷선을 적용했습니다."
            ),
        }

    left_points = [
        point for point in candidate_points if point[0] < midpoint_x
    ]
    right_points = [
        point for point in candidate_points if point[0] >= midpoint_x
    ]
    left_count = len(left_points)
    right_count = len(right_points)
    balance_ratio = min(left_count, right_count) / max(left_count, right_count)

    if left_count == 0 or right_count == 0:
        if right_count:
            endpoint = max(right_points, key=lambda point: point[0])
            concentration_side = "우측"
        else:
            endpoint = min(left_points, key=lambda point: point[0])
            concentration_side = "좌측"
        horizontal_line = ((0, endpoint[1]), (image_width - 1, endpoint[1]))
        return {
            "line": horizontal_line,
            "extended_line": horizontal_line,
            "markers": ((endpoint, endpoint[1] in peak_coordinates),),
            "concentration_side": f"{concentration_side} 집중",
            "density_log_line": (
                f"[밀집도] {side_name} {concentration_side} 집중: 끝점 {endpoint}에 "
                "연장 수평 컷선을 적용했습니다."
            ),
        }
    if balance_ratio >= 0.5:
        # 좌·우가 균형이면 첫 접점 방향의 끝 y가 더 많은 측에서 그 점을,
        # 반대측에서 반대 끝 y 점을 골라 두 점을 연결한다.
        primary_y = (
            min(point[1] for point in candidate_points)
            if use_maximum
            else max(point[1] for point in candidate_points)
        )
        primary_coordinate_points = [
            point for point in candidate_points if point[1] == primary_y
        ]
        primary_left_count = sum(
            point[0] < midpoint_x for point in primary_coordinate_points
        )
        primary_right_count = len(primary_coordinate_points) - primary_left_count
        primary_side = "좌측" if primary_left_count >= primary_right_count else "우측"
        secondary_side = "우측" if primary_side == "좌측" else "좌측"

        primary_side_points = [
            point
            for point in candidate_points
            if (point[0] < midpoint_x) == (primary_side == "좌측")
        ]
        secondary_side_points = [
            point
            for point in candidate_points
            if (point[0] < midpoint_x) == (secondary_side == "좌측")
        ]
        primary_side_y = (
            min(point[1] for point in primary_side_points)
            if use_maximum
            else max(point[1] for point in primary_side_points)
        )
        secondary_side_y = (
            max(point[1] for point in secondary_side_points)
            if use_maximum
            else min(point[1] for point in secondary_side_points)
        )
        primary_candidates = [
            point for point in primary_side_points if point[1] == primary_side_y
        ]
        secondary_candidates = [
            point for point in secondary_side_points if point[1] == secondary_side_y
        ]
        primary_point = (
            min(primary_candidates, key=lambda point: point[0])
            if primary_side == "좌측"
            else max(primary_candidates, key=lambda point: point[0])
        )
        secondary_point = (
            min(secondary_candidates, key=lambda point: point[0])
            if secondary_side == "좌측"
            else max(secondary_candidates, key=lambda point: point[0])
        )
        line = (primary_point, secondary_point)
        return {
            "line": line,
            "extended_line": extend_line_to_image_edges(line, image_width),
            "markers": (
                (primary_point, primary_side_y in peak_coordinates),
                (secondary_point, secondary_side_y in peak_coordinates),
            ),
            "concentration_side": "좌우 균형",
            "density_log_line": (
                f"[밀집도] {side_name} 좌우 균형: {primary_side} {extreme_name} "
                f"y={primary_side_y} 끝점 {primary_point} → {secondary_side} "
                f"반대 끝 y={secondary_side_y} 끝점 {secondary_point} 연결"
            ),
        }

    if right_count > left_count:
        majority_points = right_points
        minority_points = left_points
        majority_side, minority_side = "우측", "좌측"
    else:
        majority_points = left_points
        minority_points = right_points
        majority_side, minority_side = "좌측", "우측"

    majority_y = (
        min(point[1] for point in majority_points)
        if use_maximum
        else max(point[1] for point in majority_points)
    )
    minority_y = (
        max(point[1] for point in minority_points)
        if use_maximum
        else min(point[1] for point in minority_points)
    )
    majority_point = get_cut_marker_point(majority_points, majority_y)
    minority_point = get_cut_marker_point(minority_points, minority_y)

    line = (majority_point, minority_point)
    return {
        "line": line,
        "extended_line": extend_line_to_image_edges(line, image_width),
        "markers": (
            (majority_point, majority_y in peak_coordinates),
            (minority_point, minority_y in peak_coordinates),
        ),
        "concentration_side": "한쪽 치우침",
        "density_log_line": (
            f"[밀집도] {side_name} 한쪽 치우침: {majority_side} {extreme_name} 점 "
            f"{majority_point} → {minority_side} 반대 끝 점 {minority_point} 연결"
        ),
    }


def get_side_cut_coordinate(points, use_maximum):
    """첫 접점 2px 범위 안의 빨강·주황 후보에서 Merge 컷을 선택한다."""
    band_points, peak_coordinates, merge_coordinates = (
        get_first_contact_candidate_groups(points, use_maximum)
    )
    candidates = peak_coordinates | merge_coordinates
    if not candidates:
        return None, peak_coordinates
    return (
        max(candidates) if use_maximum else min(candidates),
        peak_coordinates,
    )


def get_cut_marker_point(points, cut_y):
    """선으로 연결할 Merge 컷 y 좌표의 실제 외곽 접점 하나를 고른다."""
    candidates = [point for point in points if point[1] == cut_y]
    if not candidates:
        return None
    median_x = np.median([point[0] for point in candidates])
    return min(candidates, key=lambda point: (abs(point[0] - median_x), point[0]))


def extend_line_to_image_edges(line, image_width):
    """두 접점을 잇는 선을 이미지의 좌·우 끝까지 연장한다."""
    (left_x, left_y), (right_x, right_y) = line
    if left_x == right_x:
        return ((0, left_y), (image_width - 1, right_y))

    slope = (right_y - left_y) / (right_x - left_x)
    return (
        (0, int(round(left_y - left_x * slope))),
        (image_width - 1, int(round(left_y + (image_width - 1 - left_x) * slope))),
    )


def get_side_cut_line(
    measurements, point_attribute, image_width, use_maximum, midpoint_x=None
):
    """컨투어 박스 중앙으로 나눈 좌·우 Merge 컷을 하나의 선으로 연결한다."""
    midpoint_x = image_width // 2 if midpoint_x is None else midpoint_x
    all_points = [
        point
        for measurement in measurements
        for point in getattr(measurement, point_attribute)
    ]
    band_points, peak_coordinates, merge_coordinates = (
        get_first_contact_candidate_groups(all_points, use_maximum)
    )

    # 최빈 좌표 근처에 주황 Merge 후보가 없으면, 최빈 좌표 자체에 수평선을
    # 만들고 그 선을 측면 커팅 기준으로 사용한다.
    if peak_coordinates and not merge_coordinates:
        cut_y = max(peak_coordinates) if use_maximum else min(peak_coordinates)
        marker = get_cut_marker_point(all_points, cut_y)
        if marker is None:
            return None
        horizontal_line = ((0, cut_y), (image_width - 1, cut_y))
        return {
            "line": horizontal_line,
            "extended_line": horizontal_line,
            "markers": ((marker, True),),
        }

    left_points, right_points = get_side_points(
        measurements, point_attribute, midpoint_x
    )
    left_y, left_peaks = get_side_cut_coordinate(left_points, use_maximum)
    right_y, right_peaks = get_side_cut_coordinate(right_points, use_maximum)
    if left_y is None or right_y is None:
        return None

    left_marker = get_cut_marker_point(left_points, left_y)
    right_marker = get_cut_marker_point(right_points, right_y)
    if left_marker is None or right_marker is None:
        return None

    return {
        "line": (left_marker, right_marker),
        "extended_line": extend_line_to_image_edges(
            (left_marker, right_marker), image_width
        ),
        "markers": (
            (left_marker, left_y in left_peaks),
            (right_marker, right_y in right_peaks),
        ),
    }


def draw_frequency_contact_points(image, measurements, point_attribute, use_maximum):
    """선택 면 전체의 빈도에 따라 첫 접점을 빨강·주황·파랑으로 표시한다."""
    points = [
        point
        for measurement in measurements
        for point in getattr(measurement, point_attribute)
    ]
    band_points, peak_coordinates, merge_coordinates = (
        get_first_contact_candidate_groups(points, use_maximum)
    )
    for point_x, point_y in points:
        color = (
            HISTOGRAM_PEAK_POINT_COLOR
            if point_y in peak_coordinates
            else HISTOGRAM_MERGE_POINT_COLOR
            if point_y in merge_coordinates
            else HISTOGRAM_REMAINDER_POINT_COLOR
        )
        cv2.circle(
            image,
            (point_x, point_y),
            radius=1,
            color=color,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_AA,
        )


def draw_side_cutting_guides(image, midpoint_x):
    """좌·우 Merge 집계를 나누는 컨투어 박스 중앙 기준선을 표시한다."""
    cv2.line(
        image,
        (midpoint_x, 0),
        (midpoint_x, image.shape[0] - 1),
        CENTER_SPLIT_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_side_cutting_boundary(image, cut_boundary):
    """접점, 접점 연결선, 그리고 그 양쪽 연장선을 표시한다."""
    if cut_boundary is None:
        return

    start, end = cut_boundary["line"]
    extension_start, extension_end = cut_boundary["extended_line"]
    cv2.line(
        image,
        extension_start,
        extension_end,
        CUT_LINE_EXTENSION_COLOR,
        1,
        cv2.LINE_AA,
    )
    # 수평 컷선처럼 실제 선과 연장선이 같으면 전체를 회색 연장선으로 둔다.
    # 두 실제 접점을 잇는 구간만 별도로 존재할 때만 주황색으로 강조한다.
    if (start, end) != (extension_start, extension_end):
        cv2.line(image, start, end, HISTOGRAM_MERGE_POINT_COLOR, 1, cv2.LINE_AA)
    for point, is_peak in cut_boundary["markers"]:
        color = HISTOGRAM_PEAK_POINT_COLOR if is_peak else HISTOGRAM_MERGE_POINT_COLOR
        cv2.circle(image, point, radius=3, color=color, thickness=cv2.FILLED)


def load_ab_tiff_pages(image_path):
    """A/B 다중 페이지 TIFF의 앞 두 페이지를 플랫폼과 무관하게 읽는다."""
    try:
        with Image.open(image_path) as tiff:
            if getattr(tiff, "n_frames", 1) < 2:
                raise ValueError("A/B 두 페이지 TIFF가 아닙니다.")

            pages = []
            for page_index in (0, 1):
                tiff.seek(page_index)
                page = np.array(tiff.copy())
                if page.ndim not in (2, 3):
                    raise ValueError(
                        f"{page_index + 1}번째 페이지의 차원이 올바르지 않습니다: "
                        f"{page.ndim}D"
                    )
                pages.append(page)
            return pages
    except (OSError, ValueError) as error:
        raise ValueError(f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}") from error


def create_detection_visualization(image_path, show_side_cutting=True):
    """B 분석선과 동일 좌표의 A 페이지 Crop 미리보기를 만든다."""

    image_a, image_b = load_ab_tiff_pages(image_path)
    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(f"A/B 페이지 크기가 일치하지 않습니다: {image_path}")

    gray = to_grayscale(image_b) # 이미지 파일 그레이 스케일 
    result = find_b_contours(image_b) # 컨투어링 작업
    center_split_x = get_contour_box_center_x(result.contours, gray.shape[1])
    result.center_split_x = center_split_x

    for contour in result.contours:
        # 컨투어에 가장 먼저 닿는 접점 찾기
        measurement = find_first_contact_points(contour, gray.shape)
        result.measurements.append(measurement)

    append_density_log_when_merge_exists(
        result, "top_points", center_split_x, use_maximum=True, side_name="상면"
    )
    append_density_log_when_merge_exists(
        result,
        "bottom_points",
        center_split_x,
        use_maximum=False,
        side_name="하면",
    )

    # 비교용 최상단/빈도(회색) 기준선은 측면 커팅 스위치와 무관하게 계산한다.
    # 스위치는 원본·B 페이지에 실제 컷선을 그릴지 여부만 제어한다.
    image_width = gray.shape[1]
    gray_top_cut_boundary = get_concentrated_cut_line(
        result.measurements,
        "top_points",
        image_width,
        use_maximum=True,
        side_name="상면",
        midpoint_x=center_split_x,
    )
    if gray_top_cut_boundary is None:
        gray_top_cut_boundary = get_side_cut_line(
            result.measurements,
            "top_points",
            image_width,
            use_maximum=True,
            midpoint_x=center_split_x,
        )

    top_cut_boundary = None
    bottom_cut_boundary = None
    if show_side_cutting:
        top_cut_boundary = gray_top_cut_boundary
        if top_cut_boundary is not None:
            log_line = top_cut_boundary.get("density_log_line")
            if log_line:
                result.density_log_lines.append(log_line)
                print(log_line)
        bottom_cut_boundary = get_concentrated_cut_line(
            result.measurements,
            "bottom_points",
            image_width,
            use_maximum=False,
            side_name="하면",
            midpoint_x=center_split_x,
        )
        if bottom_cut_boundary is None:
            bottom_cut_boundary = get_side_cut_line(
                result.measurements,
                "bottom_points",
                image_width,
                use_maximum=False,
                midpoint_x=center_split_x,
            )
        else:
            log_line = bottom_cut_boundary["density_log_line"]
            result.density_log_lines.append(log_line)
            print(log_line)

    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 그레이 스케일를 컬로로 변환하여 컨투어 색상이 보이게
    draw_frequency_contact_points(
        result_image, result.measurements, "top_points", use_maximum=True
    )
    draw_frequency_contact_points(
        result_image, result.measurements, "bottom_points", use_maximum=False
    )
    draw_side_cutting_guides(result_image, center_split_x)
    if show_side_cutting:
        draw_side_cutting_boundary(result_image, top_cut_boundary)
        draw_side_cutting_boundary(result_image, bottom_cut_boundary)

    # 컨투어의 하면 기준선 전체에서 아래(y+)로 내려가며 최초로 값 255(흰색)이
    # 되는 지점을 찾는다. B 페이지는 컨투어 검출에만 사용하고 A에만 표시한다.
    a_ball_white_points = find_downward_contour_white_points(
        image_a, result.contours
    )
    result.a_ball_white_points = a_ball_white_points
    result.a_ball_bottommost_y, result.a_ball_bottommost_count = (
        get_bottommost_point_count(a_ball_white_points)
    )
    bottom_contour_x_range = get_bottom_contour_x_range(
        gray.shape, result.contours
    )
    result.a_ball_bottommost_points_by_third = get_bottommost_points_by_thirds(
        a_ball_white_points, bottom_contour_x_range
    )
    result.selected_ball_bottommost_points = select_detected_bottommost_points(
        result.a_ball_bottommost_points_by_third
    )
    result.ball_square_surface_bottom_y_by_x = get_bottom_contour_y_by_x(
        gray.shape, result.contours
    )
    result.ball_square_crop_preview = create_ball_square_crop_preview(
        image_a,
        result.selected_ball_bottommost_points,
        result.ball_square_surface_bottom_y_by_x,
    )

    # A 페이지 상단의 양 끝에서 중앙으로 스캔한 색 변화점을 회색 점으로 표시한다.
    pillar_reference_points = find_top_pillar_reference_points(image_a)
    pillar_outer_reference_points = get_pillar_outer_reference_points(
        image_a, pillar_reference_points
    )
    pillar_downward_points = find_downward_color_change_points(
        image_a, pillar_outer_reference_points
    )
    contour_outline = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(contour_outline, result.contours, -1, 255, thickness=1)
    left_contour_points, right_contour_points = (
        find_left_right_contour_reference_points(contour_outline)
    )
    blue_left_reference_line, blue_right_reference_line = (
        get_first_contact_side_lines(result.measurements, gray.shape[0])
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
    source_visualization = draw_ball_squares(
        source_visualization,
        result.selected_ball_bottommost_points,
        result.ball_square_surface_bottom_y_by_x,
    )
    source_preview_image = to_bgr(image_a)
    result.source_visualization = source_visualization
    if len(pillar_downward_points) == 2:
        pillar_top_cut_line = pillar_downward_points
    elif len(pillar_downward_points) == 1:
        point_y = pillar_downward_points[0][1]
        pillar_top_cut_line = ((0, point_y), (gray.shape[1] - 1, point_y))
    else:
        pillar_top_cut_line = None
    pillar_bottom_cut_line = get_pillar_bottom_cut_line(
        pillar_contour_contact_points, gray.shape[1]
    )
    existing_top_cut_line = (gray_top_cut_boundary or {}).get("extended_line")
    blue_line_preview = create_polygon_preview(
        source_preview_image,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=pillar_top_cut_line,
        bottom_cut_line=pillar_bottom_cut_line,
        left_reference_line=blue_left_reference_line,
        right_reference_line=blue_right_reference_line,
    )
    blue_line_b_preview = create_polygon_preview(
        image_b,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=pillar_top_cut_line,
        bottom_cut_line=pillar_bottom_cut_line,
        left_reference_line=blue_left_reference_line,
        right_reference_line=blue_right_reference_line,
    )
    gray_line_preview = create_polygon_preview(
        source_preview_image,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=existing_top_cut_line,
        bottom_cut_line=(bottom_cut_boundary or {}).get("extended_line"),
    )
    gray_line_b_preview = create_polygon_preview(
        image_b,
        result.contours,
        result.measurements,
        SOURCE_PREVIEW_MASK_MODE,
        top_cut_line=existing_top_cut_line,
        bottom_cut_line=(bottom_cut_boundary or {}).get("extended_line"),
    )
    result.analysis_preview = create_preview_comparison(
        blue_line_preview,
        "A Page crop",
        blue_line_b_preview,
        "B Page crop",
    )
    result.source_preview = create_preview_comparison(
        gray_line_preview,
        "A Page crop",
        gray_line_b_preview,
        "B Page crop",
    )
    return result_image, result
