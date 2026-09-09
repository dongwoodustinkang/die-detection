"""Side, Bottom, Top 검사에서 공유하는 컨투어 추출과 기하 계산."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from general import to_grayscale


MIN_CONTOUR_AREA = 3500
MAX_CONTOUR_AREA = 10000
TOP_BOTTOM_SAMPLE_COUNT = 15
LEFT_RIGHT_SAMPLE_COUNT = 5
CONTACT_POINT_MODE = "full"
TOP_DENSE_BAND_TOLERANCE = 1
MAX_COUNT_RATIO = 0.5
FIRST_CONTACT_BAND_PIXELS = 2
REFERENCE_POINT_RADIUS = 2
REFERENCE_POINT_MIN_DISTANCE = 30
EXTREME_SENCODARY_DIFF = 2
MIN_DISTANCE = 30
APPROX_POLYGON_EPSILON_RATIO = 0.02
LINE_QUADRILATERAL_METHOD = "primary_axis"


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


# 컨투어 추출과 첫 접점 측정
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

    return filtered_contours


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


# 공용 다각형·직선 기하 계산
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


# 접점 기반 기준선과 사각형 생성
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
