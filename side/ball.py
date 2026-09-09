"""컨투어 아래 방향의 255(흰색) 지점을 찾고 표시하는 기능.

B 페이지에서 얻은 컨투어의 하면 선만 시작점으로 사용한다. 각 x 좌표에서
가장 아래에 있는 컨투어 픽셀을 선택하고, 여기서 y를 증가시키며 아래로
이동했을 때 처음 만나는 값 255(흰색) 위치를 반환한다.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# BGR. 화면에서 잘 보이는 핑크색이다.
PINK_COLOR = (203, 53, 255)
PINK_POINT_RADIUS = 2
BALL_SQUARE_HALF_SIZE = 20
# 검사 이미지의 흰색 픽셀 값이다.
WHITE_VALUE = 255
SCAN_START_OFFSET = 1


def _as_color_image(image: np.ndarray) -> np.ndarray:
    """표시용 8-bit 3채널 이미지를 만든다."""

    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        color = image

    if color.dtype == np.uint8:
        return color
    return cv2.normalize(color, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _white_mask(image: np.ndarray) -> np.ndarray:
    """정규화하지 않은 원본 값으로 색상값 255 픽셀만 선택한다."""

    if image.ndim == 2:
        return image == WHITE_VALUE
    # RGB/BGR 순서는 255 판정에 영향을 주지 않는다. 알파 채널은 제외한다.
    return np.all(image[:, :, :3] == WHITE_VALUE, axis=2)


def _contour_line_mask(
    image_shape: Tuple[int, int], contours: Iterable[np.ndarray]
) -> np.ndarray:
    """근사 꼭짓점이 아닌 실제 컨투어 선 전체를 1픽셀 마스크로 만든다."""

    mask = np.zeros(image_shape, dtype=np.uint8)
    contour_list = list(contours)
    if contour_list:
        cv2.drawContours(mask, contour_list, -1, 255, thickness=1, lineType=cv2.LINE_8)
    return mask


def _bottom_contour_line_points(contour_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """각 x 열에서 가장 아래에 있는 컨투어 선 좌표만 반환한다.

    이 좌표들은 상면·측면이 아닌 하면 기준선이다. 컨투어가 존재하는 모든
    x 열을 사용하므로 하면의 전체 폭을 빠뜨리지 않는다.
    """

    _, width = contour_mask.shape
    x_coordinates = []
    y_coordinates = []
    for x in range(width):
        ys = np.flatnonzero(contour_mask[:, x])
        if len(ys):
            x_coordinates.append(x)
            y_coordinates.append(int(ys[-1]))
    return np.asarray(y_coordinates, dtype=np.int32), np.asarray(x_coordinates, dtype=np.int32)


def get_bottom_contour_x_range(
    image_shape: Tuple[int, int], contours: Iterable[np.ndarray]
) -> Optional[Tuple[int, int]]:
    """하면 컨투어 선이 차지하는 x 좌표의 시작·끝 범위를 반환한다."""

    contour_mask = _contour_line_mask(image_shape, contours)
    _, x_coordinates = _bottom_contour_line_points(contour_mask)
    if len(x_coordinates) == 0:
        return None
    return int(x_coordinates.min()), int(x_coordinates.max())


def get_bottom_contour_y_by_x(
    image_shape: Tuple[int, int], contours: Iterable[np.ndarray]
) -> Dict[int, int]:
    """하면 컨투어의 각 x 좌표에 대응하는 가장 아래 y 좌표를 반환한다."""

    contour_mask = _contour_line_mask(image_shape, contours)
    y_coordinates, x_coordinates = _bottom_contour_line_points(contour_mask)
    return {
        int(point_x): int(point_y)
        for point_x, point_y in zip(x_coordinates, y_coordinates)
    }


def find_downward_contour_white_points(
    image: np.ndarray,
    contours: Iterable[np.ndarray],
    scan_start_offset: int = SCAN_START_OFFSET,
) -> List[Tuple[int, int]]:
    """컨투어 하면 선의 모든 좌표 아래에서 최초의 값 255 지점을 찾는다.

    각 x 열마다 다음 255 값 행을 미리 계산해, 컨투어 하면의 모든 시작점에서
    y를 증가시키는 결과를 효율적으로 얻는다. 컬러 이미지는 세 채널 모두가
    255인 픽셀만 선택한다.
    """

    height, width = image.shape[:2]
    if height < 2 or width == 0:
        return []

    contour_mask = _contour_line_mask((height, width), contours)
    line_y, line_x = _bottom_contour_line_points(contour_mask)
    if len(line_x) == 0:
        return []

    is_white = _white_mask(image)

    points = set()
    for x in np.unique(line_x):
        # next_white[y]는 y 이상에서 처음 값 255가 되는 행이다.
        next_white = np.full(height, -1, dtype=np.int32)
        next_y = -1
        for y in range(height - 1, -1, -1):
            if is_white[y, x]:
                next_y = y
            next_white[y] = next_y

        starts = line_y[line_x == x] + scan_start_offset
        for start_y in starts:
            if start_y >= height:
                continue
            white_y = next_white[start_y]
            if white_y >= 0:
                points.add((int(x), int(white_y)))

    return sorted(points, key=lambda point: (point[1], point[0]))


def get_bottommost_point_count(
    points: Iterable[Tuple[int, int]],
) -> Tuple[Optional[int], int]:
    """최하단 y좌표와 정확히 일치하는 점의 개수를 반환한다."""

    point_list = list(points)
    if not point_list:
        return None, 0
    bottom_y = max(point[1] for point in point_list)
    return bottom_y, sum(point[1] == bottom_y for point in point_list)


def get_bottommost_points_by_thirds(
    points: Iterable[Tuple[int, int]],
    x_range: Optional[Tuple[int, int]],
) -> List[Optional[Tuple[int, int]]]:
    """하면 x 범위의 3등분마다 가장 아래 핑크 점의 대표 좌표를 반환한다."""

    point_list = list(points)
    if x_range is None:
        return [None, None, None]

    left_x, right_x = x_range
    width = right_x - left_x + 1
    bottommost_points = []
    for section_index in range(3):
        start_x = left_x + width * section_index / 3
        end_x = left_x + width * (section_index + 1) / 3
        candidates = [
            point
            for point in point_list
            if start_x <= point[0] < end_x
            or (
                section_index == 2
                and start_x <= point[0] <= right_x
            )
        ]
        if not candidates:
            bottommost_points.append(None)
            continue

        bottom_y = max(point[1] for point in candidates)
        bottom_candidates = [point for point in candidates if point[1] == bottom_y]
        section_center_x = (start_x + end_x) / 2
        bottommost_points.append(
            min(bottom_candidates, key=lambda point: abs(point[0] - section_center_x))
        )
    return bottommost_points


def select_detected_bottommost_points(
    points_by_third: List[Optional[Tuple[int, int]]],
    maximum_y_difference: int = 10,
) -> List[Tuple[int, int]]:
    """3등분 최하단 좌표에서 2개 또는 3개 볼 조건을 자동으로 판정한다.

    1·2·3등분 모두에 좌표가 있고 y 범위가 ``maximum_y_difference``px 이내면
    3개 볼로 판정한다. 이 조건이 아니면 1·3등분 y 차이는 기준 이내이면서
    2등분의 y만 그 범위를 벗어날 때 2개 볼로 판정한다.
    """

    if len(points_by_third) != 3:
        return []

    first, middle, third = points_by_third
    if first is None or middle is None or third is None:
        return []

    first_y, middle_y, third_y = first[1], middle[1], third[1]
    if max(first_y, middle_y, third_y) - min(first_y, middle_y, third_y) <= maximum_y_difference:
        return [first, middle, third]

    outer_points_are_aligned = abs(first_y - third_y) <= maximum_y_difference
    middle_point_is_offset = (
        abs(middle_y - first_y) > maximum_y_difference
        or abs(middle_y - third_y) > maximum_y_difference
    )
    if outer_points_are_aligned and middle_point_is_offset:
        return [first, third]
    return []


def draw_ball_squares(
    image: np.ndarray,
    points: Iterable[Tuple[int, int]],
    surface_bottom_y_by_x: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """표면 하면부터 볼 최하단까지를 한 변으로 하는 정사각형을 표시한다."""

    preview = _as_color_image(image).copy()
    image_height, image_width = preview.shape[:2]
    for point_x, point_y in points:
        left, top, right, bottom = get_ball_square_bounds(
            (image_height, image_width),
            (point_x, point_y),
            (surface_bottom_y_by_x or {}).get(point_x),
        )
        cv2.rectangle(preview, (left, top), (right, bottom), PINK_COLOR, 1, cv2.LINE_AA)
        for corner in ((left, top), (right, top), (left, bottom), (right, bottom)):
            cv2.circle(
                preview,
                corner,
                PINK_POINT_RADIUS,
                PINK_COLOR,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )
    return preview


def get_ball_square_bounds(
    image_shape: Tuple[int, int],
    point: Tuple[int, int],
    surface_bottom_y: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """표면 하면과 볼 최하단을 기준으로 한 정사각형의 경계를 반환한다."""

    image_height, image_width = image_shape
    point_x, point_y = point
    top = (
        surface_bottom_y
        if surface_bottom_y is not None and 0 <= surface_bottom_y < point_y
        else point_y - BALL_SQUARE_HALF_SIZE * 2
    )
    top = max(0, top)
    bottom = min(image_height - 1, point_y)
    side_length = max(1, bottom - top + 1)
    left = point_x - side_length // 2
    right = left + side_length - 1
    if left < 0:
        right = min(image_width - 1, right - left)
        left = 0
    if right >= image_width:
        left = max(0, left - (right - image_width + 1))
        right = image_width - 1
    return left, top, right, bottom


def create_ball_square_crop_preview(
    image: np.ndarray,
    points: Iterable[Tuple[int, int]],
    surface_bottom_y_by_x: Optional[Dict[int, int]] = None,
) -> Optional[np.ndarray]:
    """검출된 볼 정사각형 내부를 가로 한 줄 미리보기로 합친다."""

    preview = _as_color_image(image)
    image_height, image_width = preview.shape[:2]
    crops = []
    for point in points:
        left, top, right, bottom = get_ball_square_bounds(
            (image_height, image_width),
            point,
            (surface_bottom_y_by_x or {}).get(point[0]),
        )
        if right >= left and bottom >= top:
            crops.append(preview[top : bottom + 1, left : right + 1].copy())
    if not crops:
        return None

    gap = 8
    border = 2
    tile_height = max(crop.shape[0] for crop in crops) + border * 2
    output_width = sum(crop.shape[1] + border * 2 for crop in crops) + gap * (len(crops) - 1)
    output = np.full((tile_height, output_width, 3), 255, dtype=np.uint8)
    offset_x = 0
    for crop in crops:
        crop_height, crop_width = crop.shape[:2]
        offset_y = (tile_height - crop_height) // 2
        output[
            offset_y : offset_y + crop_height,
            offset_x + border : offset_x + border + crop_width,
        ] = crop
        offset_x += crop_width + border * 2 + gap
    return output
