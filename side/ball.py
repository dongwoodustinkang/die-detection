"""컨투어 아래 방향의 255(흰색) 지점을 찾고 볼 위치를 검출하는 기능."""

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# BGR. 화면에서 잘 보이는 핑크색이다.
PINK_COLOR = (203, 53, 255)
PINK_POINT_RADIUS = 2
# 검사 이미지의 흰색 픽셀 값이다.
WHITE_VALUE = 255
SCAN_START_OFFSET = 5


# 표면 경계 및 흰색(255) 픽셀(배경이 되는 부분) 스캔
def _contour_line_mask(
    image_shape: Tuple[int, int], contours: Iterable[np.ndarray]
) -> np.ndarray:
    """1픽셀 컨투어 마스크 이미지 생성"""

    mask = np.zeros(image_shape, dtype=np.uint8)
    contour_list = list(contours)
    if contour_list:
        cv2.drawContours(mask, contour_list, -1, 255, thickness=1, lineType=cv2.LINE_8)
    return mask

def _bottom_contour_line_points(contour_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """표면의 하면 컨투어 좌표 추출"""

    _, width = contour_mask.shape
    x_coordinates = []
    y_coordinates = []
    for x in range(width):
        ys = np.flatnonzero(contour_mask[:, x])
        if len(ys):
            x_coordinates.append(x)
            y_coordinates.append(int(ys[-1]))
    return np.asarray(y_coordinates, dtype=np.int32), np.asarray(x_coordinates, dtype=np.int32)

def _white_mask(image: np.ndarray) -> np.ndarray:
    """정규화하지 않은 원본 값으로 색상값 255 픽셀만 선택한다."""

    if image.ndim == 2:
        return image == WHITE_VALUE
    return np.all(image[:, :, :3] == WHITE_VALUE, axis=2)

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
    """컨투어 하면 선의 모든 좌표 아래에서 최초의 값 255 지점을 찾는다."""

    height, width = image.shape[:2]
    if height < 2 or width == 0:
        return []

    contour_mask = _contour_line_mask((height, width), contours)
    line_y, line_x = _bottom_contour_line_points(contour_mask)
    if len(line_x) == 0:
        return []

    is_white = _white_mask(image)
    points = set()

    # x축 방향 순회 (0 -> width)
    for x in np.unique(line_x): 
        # y축 방향 순회 (height -> 0)
        next_white = np.full(height, -1, dtype=np.int32)
        next_y = -1
        for y in range(height - 1, -1, -1):
            if is_white[y, x]:
                next_y = y
            next_white[y] = next_y

        # 각 x에 해당하는 모든 y 좌표에 대해 255 픽셀 찾기
        starts = line_y[line_x == x] + scan_start_offset
        for start_y in starts:
            if start_y >= height:
                continue
            white_y = next_white[start_y]
            if white_y >= 0:
                points.add((int(x), int(white_y)))

    return sorted(points, key=lambda point: (point[1], point[0]))

# 볼 대표 최하단점 선정
def get_distant_bottommost_points(
    points: Iterable[Tuple[int, int]],
    min_x_distance: int = 40,
    max_y_difference: int = 10,
) -> List[Tuple[int, int]]:
    # 각 볼의 최하단 대표점 좌표 목록 생성

    pts = list(points)
    if not pts:
        return []

    global_max_y = max(p[1] for p in pts)
    min_allowed_y = global_max_y - max_y_difference

    candidates = sorted(
        [p for p in pts if p[1] >= min_allowed_y],
        key=lambda p: (-p[1], p[0]),
    )

    selected_points = []
    for cand in candidates:
        if all(abs(cand[0] - sel[0]) >= min_x_distance for sel in selected_points):
            selected_points.append(cand)

    return sorted(selected_points, key=lambda p: p[0])

# 바운딩 박스 계산 및 UI 시각화
def get_ball_square_bounds(
    image_shape: Tuple[int, int],
    point: Tuple[int, int],
    surface_bottom_y: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    # 버운딩 박스 좌표 계산

    image_height, image_width = image_shape
    point_x, point_y = point

    left = max(0, point_x - 25)
    right = min(image_width - 1, point_x + 25)
    top = max(0, point_y - 25)
    bottom = min(image_height - 1, point_y)

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

def draw_ball_squares(
    image: np.ndarray,
    points: Iterable[Tuple[int, int]],
    surface_bottom_y_by_x: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """하단 지점을 기준으로 25px 크기의 핑크색 바운딩 박스를 표시한다."""

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

def _as_color_image(image: np.ndarray):
    """UI 표시용 3채널 이미지 변환 코드"""

    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        color = image

    if color.dtype == np.uint8:
        return color
    return cv2.normalize(color, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
