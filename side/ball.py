"""표면 하면 컨투어를 기준으로 Side 볼 검사 ROI를 만드는 기능."""

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from general import to_grayscale


# Bottom ROI와 같은 BGR 색상이다.
ROI_COLOR = (230, 180, 20)
ROI_FILL_ALPHA = 0.18
ROI_SIDE_INSET_RATIO = 0.05
TWO_SLOT_INNER_INSET_RATIO = 0.02
BALL_ROI_WIDTH = 40
BALL_ROI_HEIGHT = 40
SLOT_LINE_GRAY_MIN = 220
SLOT_LINE_GRAY_MAX = 225
SLOT_LINE_SEARCH_HEIGHT = 80
WHITE_VALUE = 255
# 임시 컨투어 실험: 이 플래그로 기존 선·점 표시만 사용하는 상태로 돌아간다.
ENABLE_SLOT_CONTOUR_EXPERIMENT = True
SLOT_CONTOUR_THRESHOLD = 249
SLOT_CONTOUR_MIN_AREA = 160
SLOT_CONTOUR_MAX_AREA = 3200
SLOT_CONTOUR_SHARPEN_AMOUNT = 2.5


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

def get_surface_slot_ranges(
    surface_bottom_y_by_x: Dict[int, int], slot_count: int
) -> List[Tuple[int, int]]:
    """표면 하면의 가로 범위를 균등한 검사 슬롯으로 분할한다."""

    if slot_count <= 0 or not surface_bottom_y_by_x:
        return []

    left = min(surface_bottom_y_by_x)
    right = max(surface_bottom_y_by_x)
    width = right - left + 1
    return [
        (
            left + (index * width) // slot_count,
            left + ((index + 1) * width) // slot_count - 1,
        )
        for index in range(slot_count)
    ]


def create_surface_ball_roi_polygons(
    surface_bottom_y_by_x: Dict[int, int],
    slot_ranges: Iterable[Tuple[int, int]],
    side_inset_ratio: float = ROI_SIDE_INSET_RATIO,
    top_offset: int = 4,
    height: int = 40,
) -> List[np.ndarray]:
    """슬롯별 표면 하면 중앙값을 기준으로 수평 볼 ROI를 만든다."""

    polygons = []
    slot_range_list = list(slot_ranges)
    for index, (left, right) in enumerate(slot_range_list):
        slot_width = right - left + 1
        outer_inset = int(round(slot_width * side_inset_ratio))
        inner_inset = outer_inset
        if len(slot_range_list) == 2:
            inner_inset = int(round(slot_width * TWO_SLOT_INNER_INSET_RATIO))
        roi_left = left + (inner_inset if index == 1 else outer_inset)
        roi_right = right - (inner_inset if index == 0 else outer_inset)
        xs = [
            x
            for x in range(roi_left, roi_right + 1)
            if x in surface_bottom_y_by_x
        ]
        if len(xs) < 2:
            continue
        baseline_y = int(round(np.median([surface_bottom_y_by_x[x] for x in xs])))
        top_y = baseline_y + top_offset
        bottom_y = top_y + height
        polygons.append(
            np.asarray(
                ((xs[0], top_y), (xs[-1], top_y), (xs[-1], bottom_y), (xs[0], bottom_y)),
                dtype=np.int32,
            )
        )
    return polygons


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _create_centered_roi_polygon(
    center_x: int, center_y: int, image_shape: Tuple[int, int]
) -> np.ndarray:
    """앵커 점을 중심으로 이미지 범위 안의 고정 크기 사각 ROI를 만든다."""

    image_height, image_width = image_shape
    left = max(0, min(center_x - BALL_ROI_WIDTH // 2, image_width - BALL_ROI_WIDTH))
    top = max(0, min(center_y - BALL_ROI_HEIGHT // 2, image_height - BALL_ROI_HEIGHT))
    right = min(image_width - 1, left + BALL_ROI_WIDTH - 1)
    bottom = min(image_height - 1, top + BALL_ROI_HEIGHT - 1)
    return np.asarray(((left, top), (right, top), (right, bottom), (left, bottom)), dtype=np.int32)


def _find_slot_gray_line(
    gray: np.ndarray, slot_xs: List[int], baseline_y: int
) -> Optional[int]:
    """표면 하단부터 탐색해 슬롯 행 평균이 220~225인 첫 y를 찾는다."""

    if not slot_xs:
        return None
    start_y = max(0, baseline_y)
    end_y = min(gray.shape[0], start_y + SLOT_LINE_SEARCH_HEIGHT)
    left, right = slot_xs[0], slot_xs[-1] + 1
    for y in range(start_y, end_y):
        row_mean = float(np.mean(gray[y, left:right]))
        if SLOT_LINE_GRAY_MIN <= row_mean <= SLOT_LINE_GRAY_MAX:
            return y
    return None


def _find_deepest_white_anchor(
    gray: np.ndarray, slot_xs: List[int], line_y: int
) -> Optional[Tuple[int, int]]:
    """기준선 아래에서 첫 255 지점까지 거리가 가장 긴 x열의 점을 찾는다."""

    search_bottom = min(gray.shape[0], line_y + SLOT_LINE_SEARCH_HEIGHT)
    slot_center_x = (slot_xs[0] + slot_xs[-1]) / 2
    anchor = None
    anchor_key = None
    for x in slot_xs:
        white_ys = np.flatnonzero(gray[line_y:search_bottom, x] == WHITE_VALUE)
        if not len(white_ys):
            continue
        white_y = line_y + int(white_ys[0])
        key = (white_y - line_y, -abs(x - slot_center_x))
        if anchor_key is None or key > anchor_key:
            anchor = (x, white_y)
            anchor_key = key
    return anchor


def create_surface_ball_anchor_roi_polygons(
    image: np.ndarray,
    surface_bottom_y_by_x: Dict[int, int],
    slot_ranges: Iterable[Tuple[int, int]],
) -> List[np.ndarray]:
    """슬롯별 220~225 기준선과 최심부 흰색 지점으로 볼 ROI를 만든다."""

    slot_range_list = list(slot_ranges)
    fallback_polygons = create_surface_ball_roi_polygons(
        surface_bottom_y_by_x, slot_range_list
    )
    if image is None or image.size == 0:
        return fallback_polygons

    gray = _to_gray(image)
    image_height, image_width = gray.shape[:2]
    polygons = []
    for index, (left, right) in enumerate(slot_range_list):
        slot_xs = [
            x for x in range(max(0, left), min(image_width - 1, right) + 1)
            if x in surface_bottom_y_by_x
        ]
        if not slot_xs:
            if index < len(fallback_polygons):
                polygons.append(fallback_polygons[index])
            continue

        baseline_y = int(round(np.median([surface_bottom_y_by_x[x] for x in slot_xs])))
        line_y = _find_slot_gray_line(gray, slot_xs, baseline_y)
        anchor = (
            _find_deepest_white_anchor(gray, slot_xs, line_y)
            if line_y is not None
            else None
        )
        if anchor is None:
            if index < len(fallback_polygons):
                polygons.append(fallback_polygons[index])
            continue
        polygons.append(_create_centered_roi_polygon(*anchor, (image_height, image_width)))
    return polygons


def draw_surface_ball_rois(
    image: np.ndarray, polygons: Iterable[np.ndarray]
) -> np.ndarray:
    """A 페이지에 표면 기준 볼 ROI만 반투명하게 표시한다."""

    preview = _as_color_image(image).copy()
    polygon_list = [polygon for polygon in polygons if len(polygon) >= 3]
    if not polygon_list:
        return preview

    overlay = preview.copy()
    cv2.fillPoly(overlay, polygon_list, ROI_COLOR, lineType=cv2.LINE_AA)
    preview = cv2.addWeighted(overlay, ROI_FILL_ALPHA, preview, 1 - ROI_FILL_ALPHA, 0)
    for polygon in polygon_list:
        cv2.polylines(preview, [polygon], True, ROI_COLOR, 1, cv2.LINE_AA)
    return preview


def create_surface_ball_roi_preview(
    image: np.ndarray, polygons: Iterable[np.ndarray]
) -> Optional[np.ndarray]:
    """슬롯별 ROI 내부 영상을 가로 타일 프리뷰로 만든다."""

    if image is None or image.size == 0:
        return None
    preview = _as_color_image(image)
    image_height, image_width = preview.shape[:2]
    tiles = []
    for polygon in polygons:
        x, y, width, height = cv2.boundingRect(np.asarray(polygon, dtype=np.int32))
        left, top = max(0, x), max(0, y)
        right, bottom = min(image_width, x + width), min(image_height, y + height)
        if right <= left or bottom <= top:
            continue
        crop = preview[top:bottom, left:right].copy()
        tile = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=ROI_COLOR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), ROI_COLOR, 1)
        tiles.append(tile)
    if not tiles:
        return None

    gap = 8
    output_height = max(tile.shape[0] for tile in tiles)
    output_width = sum(tile.shape[1] for tile in tiles) + gap * (len(tiles) - 1)
    output = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
    offset_x = 0
    for tile in tiles:
        offset_y = (output_height - tile.shape[0]) // 2
        output[offset_y : offset_y + tile.shape[0], offset_x : offset_x + tile.shape[1]] = tile
        offset_x += tile.shape[1] + gap
    return output


# 원호 실험 모듈에서 쓰는 기존 크롭 유틸리티
def get_ball_square_bounds(
    image_shape: Tuple[int, int],
    point: Tuple[int, int],
    surface_bottom_y: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    # 바운딩 박스 좌표 계산

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


def get_surface_slot_boundaries(slot_ranges):
    """슬롯 수와 무관하게 양끝을 고정한 분할 좌표를 반환한다."""
    ranges = list(slot_ranges)
    if not ranges:
        return []
    # 기존 3슬롯의 양끝 위치를 슬롯 개수와 무관하게 유지한다.
    surface_left = ranges[0][0]
    surface_width = ranges[-1][1] + 1 - surface_left
    first_third = surface_left + surface_width // 3
    second_third = surface_left + (2 * surface_width) // 3
    third_width = second_third - first_third
    outer_left = first_third - third_width
    outer_right = second_third + third_width
    slot_count = len(ranges)
    return [
        outer_left + (index * (outer_right - outer_left)) // slot_count
        for index in range(slot_count + 1)
    ]


def find_surface_slot_brightness_lines(image, slot_ranges, bottom_cut_line):
    """슬롯별 기준선 아래에서 240~255 픽셀이 20% 이상인 첫 행을 찾는다."""
    if bottom_cut_line is None:
        return []
    (x1, y1), (x2, y2) = bottom_cut_line
    if x1 == x2:
        return []
    gray = to_grayscale(image)
    height, width = gray.shape
    boundaries = get_surface_slot_boundaries(slot_ranges)
    lines = []
    for left, stop in zip(boundaries, boundaries[1:]):
        left, stop = max(0, left), min(width, stop)
        if left >= stop:
            continue
        xs = np.arange(left, stop)
        baseline_ys = y1 + (y2 - y1) * (xs - x1) / (x2 - x1)
        # 기울어진 기준선의 각 x에서 기준선 아래인 픽셀만 집계한다.
        start_y = max(0, int(np.floor(baseline_ys.min())) + 1)
        for y in range(start_y, height):
            row = gray[y, left:stop]
            bright = (row >= 240) & (row <= 255) & (y > baseline_ys)
            if np.count_nonzero(bright) * 5 >= stop - left:
                lines.append(((left, y), (stop - 1, y)))
                break
    return lines


def draw_surface_slot_brightness_lines(image, lines):
    """검출된 슬롯별 밝기 조건 가로선을 표시한다."""
    for start, end in lines:
        cv2.line(image, start, end, (0, 200, 255), 1, cv2.LINE_AA)


def find_surface_slot_deepest_bright_points(image, lines):
    """각 x의 첫 230~255 접점 중 가로선에서 가장 먼 점을 반환한다."""
    gray = to_grayscale(image)
    height, width = gray.shape
    points = []
    for (left, line_y), (right, _) in lines:
        start_y = max(0, line_y + 1)
        if start_y >= height:
            continue
        center_x = (left + right) / 2
        best_point = None
        best_key = None
        for x in range(max(0, left), min(width - 1, right) + 1):
            column = gray[start_y:, x]
            hits = np.flatnonzero((column >= 230) & (column <= 255))
            if not hits.size:
                continue
            y = start_y + int(hits[0])
            key = (y - line_y, -abs(x - center_x))
            if best_key is None or key > best_key:
                best_key = key
                best_point = (x, y)
        if best_point is not None:
            points.append(best_point)
    return points


def draw_surface_slot_bright_points(image, points):
    """슬롯별 최장거리 밝기 접점을 노란색 점으로 표시한다."""
    for point in points:
        cv2.circle(image, point, 3, (0, 200, 255), cv2.FILLED, cv2.LINE_AA)


def create_representative_point_rois(lines, points, slot_ranges):
    """거리 10px 이상인 대표점에 너비 40px, 노란선부터의 ROI를 만든다."""
    boundaries = get_surface_slot_boundaries(slot_ranges)
    rois = []
    for (left, top), (right, _) in lines:
        slot_index = next((i for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))
                           if a <= left < b), None)
        for x, bottom in points:
            if left <= x <= right and bottom - top >= 10:
                rois.append((slot_index, (x - 20, top, x + 19, bottom)))
                break
    return rois


def draw_representative_point_rois(image, rois):
    """노란선을 상단, 대표점을 하단으로 하는 ROI를 표시한다."""
    for _, (left, top, right, bottom) in rois:
        cv2.rectangle(image, (left, top), (right, bottom), ROI_COLOR, 1, cv2.LINE_AA)


def fit_representative_rois_to_contours(rois, contours):
    """대표점에 가장 가까운 겹친 컨투어 전체가 들어가도록 ROI를 확장한다."""
    fitted = []
    for slot_index, (left, top, right, bottom) in rois:
        anchor = (float(left + 20), float(bottom))
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x > right or x + w - 1 < left or y > bottom or y + h - 1 < top:
                continue
            distance = abs(cv2.pointPolygonTest(contour, anchor, True))
            candidates.append((distance, -cv2.contourArea(contour), (x, y, w, h)))
        if candidates:
            _, _, (x, y, w, h) = min(candidates, key=lambda item: item[:2])
            left, top = min(left, x), min(top, y)
            right, bottom = max(right, x + w - 1), max(bottom, y + h - 1)
        fitted.append((slot_index, (left, top, right, bottom)))
    return fitted


def create_measured_ball_roi_preview(image, rois, contours):
    """슬롯별 ROI와 내부 컨투어 후보의 너비·높이를 px 단위로 표시한다."""
    source = _as_color_image(image).copy()
    image_height, image_width = source.shape[:2]
    tiles = []
    measurements = []
    for slot_index, (left, top, right, bottom) in rois:
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(image_width, right + 1), min(image_height, bottom + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        # ROI 안에 컨투어 중심이 있는 후보 중 면적이 가장 큰 객체를 측정한다.
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x0 <= x + (w - 1) / 2 < x1 and y0 <= y + (h - 1) / 2 < y1:
                candidates.append(contour)
        if candidates:
            contour = max(candidates, key=cv2.contourArea)
            _, _, ball_width, ball_height = cv2.boundingRect(contour)
        else:
            ball_width = ball_height = None
        measurements.append((slot_index, ball_width, ball_height))
        crop = source[y0:y1, x0:x1]
        scale = 3
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
        tile_width = max(160, crop.shape[1] + 8)
        tile = np.full((crop.shape[0] + 62, tile_width, 3), 255, dtype=np.uint8)
        offset = (tile_width - crop.shape[1]) // 2
        tile[4:4 + crop.shape[0], offset:offset + crop.shape[1]] = crop
        label = f"Slot {slot_index + 1}" if slot_index is not None else "Slot"
        cv2.putText(tile, label, (6, crop.shape[0] + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (40, 40, 40), 1, cv2.LINE_AA)
        size_text = (f"W: {ball_width}px  H: {ball_height}px" if ball_width is not None
                     else "W: --  H: --")
        cv2.putText(tile, size_text, (6, crop.shape[0] + 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None, measurements
    output = np.full((max(t.shape[0] for t in tiles),
                      sum(t.shape[1] for t in tiles) + 8 * (len(tiles) - 1), 3),
                     255, dtype=np.uint8)
    offset = 0
    for tile in tiles:
        output[:tile.shape[0], offset:offset + tile.shape[1]] = tile
        offset += tile.shape[1] + 8
    return output, measurements


def enhance_ball_contour_edges(gray):
    """약하게 흐린 영상과의 차이를 더해 이진화 전에 윤곽을 강조한다."""
    denoised = cv2.medianBlur(gray, 3)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 1.0)
    return cv2.addWeighted(
        denoised, 1.0 + SLOT_CONTOUR_SHARPEN_AMOUNT,
        blurred, -SLOT_CONTOUR_SHARPEN_AMOUNT, 0,
    )


def find_surface_slot_ball_contours(image, lines):
    """임시: 윤곽 강조 후 반전 이진화하고 면적으로 후보를 거른다."""
    if not ENABLE_SLOT_CONTOUR_EXPERIMENT:
        return []
    gray = to_grayscale(image)
    enhanced = enhance_ball_contour_edges(gray)
    height, width = gray.shape
    selected = []
    for (left, line_y), (right, _) in lines:
        left, right = max(0, left), min(width - 1, right)
        top = max(0, line_y + 1)
        if left > right or top >= height:
            continue
        _, mask = cv2.threshold(
            enhanced[top:, left:right + 1], SLOT_CONTOUR_THRESHOLD, 255,
            cv2.THRESH_BINARY_INV,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not SLOT_CONTOUR_MIN_AREA <= area <= SLOT_CONTOUR_MAX_AREA:
                continue
            contour = contour + np.asarray([[[left, top]]], dtype=np.int32)
            selected.append(contour)
    return selected


def draw_surface_slot_ball_contours(image, contours):
    """면적 필터를 통과한 모든 볼 후보 컨투어를 초록색으로 표시한다."""
    if contours:
        cv2.drawContours(image, contours, -1, (0, 220, 0), 1, cv2.LINE_AA)


def draw_surface_slot_guides(image, slot_ranges, bottom_cut_line):
    """하단 기준선 아래에 슬롯의 내부·외부 세로 경계를 표시한다."""
    if bottom_cut_line is None:
        return
    (x1, y1), (x2, y2) = bottom_cut_line
    if x1 == x2:
        return
    height, width = image.shape[:2]
    boundaries = get_surface_slot_boundaries(slot_ranges)
    for left in boundaries:
        if not 0 <= left < width:
            continue
        baseline_y = int(round(y1 + (y2 - y1) * (left - x1) / (x2 - x1)))
        if baseline_y >= height:
            continue
        cv2.line(
            image, (left, max(0, baseline_y)), (left, height - 1),
            (230, 180, 20), 1, cv2.LINE_AA,
        )
