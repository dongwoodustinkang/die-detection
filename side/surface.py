"""Side 검사에서 사용하는 표면 기준선, 커팅, 크롭 생성 기능."""

import cv2
import numpy as np

from contour import (
    FIRST_CONTACT_BAND_PIXELS,
    MAX_COUNT_RATIO,
    approximate_contour_polygon,
    expand_quadrilateral_by_side,
    get_line_intersection,
    get_line_quadrilateral,
)
from general import to_bgr, to_bgra, to_grayscale


ANALYSIS_PREVIEW_MASK_MODE = "contour"
SOURCE_PREVIEW_MASK_MODE = "line_quadrilateral"
SOURCE_PREVIEW_OUTER_MARGIN = 0
SOURCE_PREVIEW_TOP_SCAN_HEIGHT = 3
SOURCE_PREVIEW_EDGE_CHANGE_THRESHOLD = 40
SOURCE_PREVIEW_EDGE_CHANGE_WINDOW = 3
SOURCE_PREVIEW_HORIZONTAL_CHANGE_MIN_RUN = 3
SOURCE_PREVIEW_COLOR_CHANGE_MIN_RUN = 3
SOURCE_PREVIEW_INVALID_Y_DIFFERENCE = 10
CONTOUR_COLOR = (247, 85, 168)
TOP_COLOR = (248, 189, 56)
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
HISTOGRAM_PEAK_POINT_COLOR = (82, 68, 240)
HISTOGRAM_MERGE_POINT_COLOR = (0, 146, 255)
HISTOGRAM_REMAINDER_POINT_COLOR = (246, 130, 49)
CENTER_SPLIT_LINE_COLOR = (180, 180, 180)
CUT_LINE_EXTENSION_COLOR = (112, 112, 112)


# Preview 조립과 마스크 도형
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


# 기둥 기준점과 컨투어 접점
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


# Crop 영역 계산과 A/B 비교 이미지
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


# 최상단/빈도 기반 컷선 분석
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


# 최종 기준선 렌더링
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
