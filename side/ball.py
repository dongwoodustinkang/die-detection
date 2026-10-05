"""표면 하면 컨투어를 기준으로 Side 볼 검사 ROI를 만드는 기능."""

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
# 노란 기준선에서 표면 높이만큼 아래로 띄운 보조 가로선 색(BGR, 자홍색)이다.
SLOT_OFFSET_LINE_COLOR = (255, 0, 255)
# 노란 기준선은 하단 기준선 아래에서 이 값을 넘는 첫 행이다.
SLOT_LINE_BRIGHT_MIN = 240
# 노란 기준선 행에서 볼 색으로 보는 최대 그레이 값이다.
SLOT_BALL_DARK_MAX = 210
# 노란 기준선 아래에 남은 표면 줄을 건너뛰는 조건이다. 노란선부터 최대 행 수 안에서
# 어두운 폭이 이 값(px) 이상 줄어들고, 그 아래 2행도 줄어든 폭에서 이 값 이상 더
# 줄지 않는(안정된) 마지막 지점의 다음 행부터 ROI를 시작한다. 계속 좁아지는 경우는
# 표면 줄이 아니라 작은 볼의 둥근 아래쪽이므로 건너뛰지 않는다.
SLOT_BALL_RESIDUE_SHRINK = 3
SLOT_BALL_RESIDUE_MAX_ROWS = 5
# 표면 잔여가 볼 한쪽에만 붙어 서서히 줄어드는 경우의 보정 조건이다. ROI 위 변부터
# 이 행 수 안에서 한쪽 끝만 SHRINK(px) 이상 안으로 들어오고 반대쪽은 1px 이하로
# 움직이면, 들어온 쪽 끝을 2행 연속 같은 위치가 된 곳으로 옮긴다.
SLOT_BALL_SIDE_RESIDUE_ROWS = 5
# 자홍색 기준선에서 위로 올라가며 이 값 이하(0~230)를 처음 만나는 행을 볼 하단으로 본다.
SLOT_BALL_BOTTOM_MAX = 230
# 볼 판정 기준이다. ROI 하단 행의 중앙 x ±3px에 검정(0~230)이 없으면 기형,
# 면적(너비×높이) 868 초과, 가로 31px 초과, 세로 27px 초과 중 하나면 크기 이상(대),
# 면적이 20×10보다 작으면 크기 이상(소)이다.
SLOT_BALL_CENTER_HALF_WIDTH = 3
SLOT_BALL_MAX_AREA = 868
SLOT_BALL_MAX_WIDTH = 31
SLOT_BALL_MAX_HEIGHT = 27
SLOT_BALL_MIN_AREA = 20 * 10
# 화면 표시·프리뷰용 ROI는 볼 영역에서 좌·우·아래로 이만큼(px) 여유를 둔다.
# 너비·높이 측정과 판정은 여유 없는 볼 영역으로 한다.
SLOT_BALL_ROI_PADDING = 2
BALL_STATUS_OK = "검출"
BALL_STATUS_DEFORMED = "기형"
BALL_STATUS_LARGE = "크기 이상(대)"
BALL_STATUS_SMALL = "크기 이상(소)"
BALL_STATUS_MISSING = "미검출"
# 한글 폰트가 없을 때 프리뷰에 쓰는 영문 표기다.
BALL_STATUS_ENGLISH = {
    BALL_STATUS_OK: "OK",
    BALL_STATUS_DEFORMED: "Deformed",
    BALL_STATUS_LARGE: "Size large",
    BALL_STATUS_SMALL: "Size small",
    BALL_STATUS_MISSING: "Not detected",
}
# 프리뷰 라벨용 한글 폰트 후보(macOS, Windows, Linux 순)이다.
PREVIEW_FONT_CANDIDATES = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


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

    if not surface_bottom_y_by_x:
        return []
    return split_slot_ranges(
        min(surface_bottom_y_by_x), max(surface_bottom_y_by_x), slot_count
    )


def split_slot_ranges(left: int, right: int, slot_count: int) -> List[Tuple[int, int]]:
    """좌·우 x 범위를 균등한 검사 슬롯으로 분할한다."""

    if slot_count <= 0 or right < left:
        return []

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


def _find_first_bright_y_on_column(gray, x, bottom_cut_line):
    """세로선 x에서 기준선 아래 첫 240 초과 픽셀의 y를 찾는다."""
    (x1, y1), (x2, y2) = bottom_cut_line
    baseline_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    start_y = max(0, int(np.floor(baseline_y)) + 1)
    column = gray[start_y:, x]
    hits = np.flatnonzero(column > SLOT_LINE_BRIGHT_MIN)
    return start_y + int(hits[0]) if hits.size else None


def find_surface_slot_brightness_lines(image, slot_ranges, bottom_cut_line):
    """가운데 슬롯 세로선 2개에서만 기준선 아래 첫 240 초과 픽셀 행을 찾는다."""
    if bottom_cut_line is None:
        return []
    (x1, _), (x2, _) = bottom_cut_line
    if x1 == x2:
        return []
    gray = to_grayscale(image)
    height, width = gray.shape
    boundaries = get_surface_slot_boundaries(slot_ranges)
    # 슬롯 전체 x 대신 양끝을 뺀 가운데 세로선 좌표만 검사해 연산을 줄인다.
    inner_hits = {
        x: _find_first_bright_y_on_column(gray, x, bottom_cut_line)
        for x in boundaries[1:-1]
        if 0 <= x < width
    }
    lines = []
    for left, stop in zip(boundaries, boundaries[1:]):
        # 슬롯에 맞닿은 가운데 세로선 접점의 평균 y를 사용한다.
        ys = [y for x, y in inner_hits.items() if x in (left, stop) and y is not None]
        left, stop = max(0, left), min(width, stop)
        if left >= stop or not ys:
            continue
        y = int(round(sum(ys) / len(ys)))
        lines.append(((left, y), (stop - 1, y)))
    return lines


def draw_surface_slot_brightness_lines(image, lines):
    """검출된 슬롯별 밝기 조건 가로선을 표시한다."""
    for start, end in lines:
        cv2.line(image, start, end, (0, 200, 255), 1, cv2.LINE_AA)


def _line_y_at(line, x):
    (x1, y1), (x2, y2) = line
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def offset_surface_slot_lines(lines, top_cut_line, bottom_cut_line):
    """슬롯별 노란 기준선을 상단~하단 기준선 간격(표면 높이)만큼 아래로 옮긴다."""
    if top_cut_line is None or bottom_cut_line is None:
        return []
    if top_cut_line[0][0] == top_cut_line[1][0] or bottom_cut_line[0][0] == bottom_cut_line[1][0]:
        return []
    offset_lines = []
    for (x1, y1), (x2, y2) in lines:
        # 이미지 배율이 바뀌어도 따라가도록 슬롯 가운데 x의 표면 높이를 쓴다.
        center_x = (x1 + x2) / 2
        distance = int(round(
            _line_y_at(bottom_cut_line, center_x) - _line_y_at(top_cut_line, center_x)
        ))
        offset_lines.append(((x1, y1 + distance), (x2, y2 + distance)))
    return offset_lines


def draw_surface_slot_offset_lines(image, lines):
    """이미지 안에 들어오는 보조 가로선만 표시한다."""
    height = image.shape[0]
    for start, end in lines:
        if start[1] < height:
            cv2.line(image, start, end, SLOT_OFFSET_LINE_COLOR, 1, cv2.LINE_AA)


def find_surface_slot_ball_rois(image, lines, offset_lines, slot_ranges, slot_indices):
    """노란 기준선 접점에서 가로로 좌·우 끝을, 자홍색 기준선에서 위로 하단을 찾아 볼 ROI를 만든다.

    slot_indices 순서대로 (슬롯 번호, 너비, 높이, 박스, 판정)을 반환한다.
    ROI가 없는 슬롯은 너비·높이·박스를 None, 판정을 미검출로 둔다.
    """
    gray = to_grayscale(image)
    height, width = gray.shape
    boundaries = get_surface_slot_boundaries(slot_ranges)
    # 슬롯 왼쪽 x로 자홍색 기준선 y를 찾는다(없으면 이미지 맨 아래까지 본다).
    offset_y_by_left = {start[0]: start[1] for start, _ in offset_lines}
    found = {}
    for (left, y), (right, _) in lines:
        magenta_y = min(height - 1, offset_y_by_left.get(left, height - 1))
        slot_index = next(
            (i for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])) if a <= left < b),
            None,
        )
        x0, x1 = max(0, left), min(width - 1, right)
        if slot_index is None or not 0 <= y < height or x0 > x1:
            continue
        # 좌·우 접점에서 안쪽으로 처음 만나는 볼 색(0~210)이 ROI의 좌·우 끝이다.
        rows = gray[y:min(height, y + SLOT_BALL_RESIDUE_MAX_ROWS + 3), x0:x1 + 1] <= SLOT_BALL_DARK_MAX
        spans = [
            (int(cols[0]), int(cols[-1])) if cols.size else None
            for cols in (np.flatnonzero(row) for row in rows)
        ]
        # 표면 줄은 여러 행일 수 있어(예: 폭 31→54→54→54→23) 범위 안에서 폭이
        # 확 줄어드는 마지막 지점까지를 표면 줄로 보고 그 다음 행을 ROI 위 변으로 쓴다.
        span_widths = [span[1] - span[0] + 1 if span else 0 for span in spans]
        skip = 0
        shrink = SLOT_BALL_RESIDUE_SHRINK
        for row in range(min(SLOT_BALL_RESIDUE_MAX_ROWS, len(spans) - 3)):
            after = span_widths[row + 1]
            if (
                spans[row] is not None and spans[row + 1] is not None
                and span_widths[row] - after >= shrink
                and span_widths[row + 2] >= after - shrink
                and span_widths[row + 3] >= after - shrink
            ):
                skip = row + 1
        if spans[skip] is None:
            continue
        y += skip
        hits = _trim_one_side_residue(gray, y, x0, x1, spans[skip])
        # ROI 좌·우 폭 안에서 자홍색 기준선부터 위로 올라가며 처음 닿는 검정(0~230)을
        # ROI 하단으로 정한다. 옆 슬롯에서 넘어온 볼은 ROI 폭 밖이라 영향을 주지 않는다.
        roi_left, roi_right = x0 + hits[0], x0 + hits[1]
        roi_bottom = _find_ball_bottom(gray, y, magenta_y, roi_left, roi_right)
        # 볼은 위(표면에 붙은 목)보다 가운데가 더 넓을 수 있어, 볼 몸통 행에서
        # 가운데와 이어진 어두운 구간까지 좌·우를 넓히고 하단을 다시 찾는다.
        roi_left, roi_right = _widen_to_ball_body(
            gray, y, roi_bottom, x0, x1, roi_left, roi_right
        )
        roi_bottom = _find_ball_bottom(gray, y, magenta_y, roi_left, roi_right)
        box = (roi_left, y, roi_right, roi_bottom)
        roi_width, roi_height = box[2] - box[0] + 1, box[3] - box[1] + 1
        # 1×N, 2×N(N×1, N×2)처럼 선 두께인 영역은 볼이 아니므로 검출 대상에서 뺀다(미검출).
        if roi_width <= 2 or roi_height <= 2:
            continue
        found[slot_index] = (
            slot_index, roi_width, roi_height, box, _classify_slot_ball(gray, box),
        )
    return [
        found.get(index, (index, None, None, None, BALL_STATUS_MISSING))
        for index in slot_indices
    ]


def _find_ball_bottom(gray, top, magenta_y, left, right):
    """ROI 좌·우 폭 안에서 자홍색 기준선부터 위로 올라가며 처음 닿는 검정(0~230) 행."""
    dark_rows = np.flatnonzero(
        (gray[top:max(top, magenta_y) + 1, left:right + 1] <= SLOT_BALL_BOTTOM_MAX).any(axis=1)
    )
    return top + int(dark_rows[-1]) if dark_rows.size else top


def _widen_to_ball_body(gray, top, bottom, x0, x1, left, right):
    """볼 몸통 행에서 중앙과 이어진 어두운 구간이 더 넓으면 ROI 좌·우를 넓힌다.

    표면 잔여가 붙는 위쪽 행(SLOT_BALL_SIDE_RESIDUE_ROWS)은 제외하고, 슬롯 범위
    안에서 중앙 x와 끊김 없이 이어진 볼 색(0~210) 구간만 본다.
    """
    center = (left + right) // 2 - x0
    start = top + SLOT_BALL_SIDE_RESIDUE_ROWS if bottom - top > SLOT_BALL_SIDE_RESIDUE_ROWS else top
    for row in gray[start:bottom + 1, x0:x1 + 1] <= SLOT_BALL_DARK_MAX:
        if not row[center]:
            continue
        gaps = np.flatnonzero(~row)
        run_left = int(gaps[gaps < center].max()) + 1 if (gaps < center).any() else 0
        run_right = int(gaps[gaps > center].min()) - 1 if (gaps > center).any() else len(row) - 1
        left, right = min(left, x0 + run_left), max(right, x0 + run_right)
    return left, right


def _trim_one_side_residue(gray, y, x0, x1, span):
    """볼 한쪽에만 붙은 표면 잔여 때문에 넓게 잡힌 좌·우 끝을 보정한다.

    볼 자체가 둥글게 좁아질 때는 좌·우가 함께 들어오지만, 한쪽에 붙은 잔여 줄은
    그쪽 끝만 행마다 조금씩 들어온다(예: 오른쪽 끝 55→55→54→52→50→50).
    """
    rows = SLOT_BALL_SIDE_RESIDUE_ROWS
    lefts, rights = [], []
    for row in gray[y:y + rows + 2, x0:x1 + 1] <= SLOT_BALL_DARK_MAX:
        cols = np.flatnonzero(row)
        if not cols.size:
            return span
        lefts.append(int(cols[0]))
        rights.append(int(cols[-1]))
    if len(lefts) < rows + 2:
        return span
    left, right = span
    for edges, others, inward in ((rights, lefts, -1), (lefts, rights, 1)):
        for row in range(1, rows + 1):
            moved = (edges[row] - edges[0]) * inward
            if (
                moved >= SLOT_BALL_RESIDUE_SHRINK
                and abs(others[row] - others[0]) <= 1
                and edges[row] == edges[row + 1]
            ):
                if inward < 0:
                    right = edges[row]
                else:
                    left = edges[row]
                break
    return left, right


def _classify_slot_ball(gray, box):
    """볼 ROI를 기형 → 크기 이상(대) → 크기 이상(소) → 검출 순으로 판정한다.

    기형: ROI 하단(마지막 검정 행)의 중앙 x ±3px에 검정(0~230)이 없는 경우.
    정상 볼은 가장 아래 점이 가운데에 있지만, 한쪽으로 찌그러진 볼은 가운데가 비어 있다.
    """
    left, top, right, bottom = box
    center_x = (left + right) // 2
    half = SLOT_BALL_CENTER_HALF_WIDTH
    bottom_center = gray[bottom, max(0, center_x - half):center_x + half + 1]
    if not (bottom_center <= SLOT_BALL_BOTTOM_MAX).any():
        return BALL_STATUS_DEFORMED
    roi_width, roi_height = right - left + 1, bottom - top + 1
    if (
        roi_width * roi_height > SLOT_BALL_MAX_AREA
        or roi_width > SLOT_BALL_MAX_WIDTH
        or roi_height > SLOT_BALL_MAX_HEIGHT
    ):
        return BALL_STATUS_LARGE
    if roi_width * roi_height < SLOT_BALL_MIN_AREA:
        return BALL_STATUS_SMALL
    return BALL_STATUS_OK


def pad_ball_roi(box, image_shape):
    """볼 영역을 좌·우·아래로 여유 있게 넓힌 표시용 ROI를 반환한다."""
    left, top, right, bottom = box
    height, width = image_shape[:2]
    pad = SLOT_BALL_ROI_PADDING
    return (max(0, left - pad), top, min(width - 1, right + pad), min(height - 1, bottom + pad))


def draw_surface_slot_ball_boxes(image, measurements):
    """슬롯별 볼 ROI를 사각형으로 표시한다."""
    for _, _, _, box, _ in measurements:
        if box is not None:
            roi = pad_ball_roi(box, image.shape)
            cv2.rectangle(image, roi[:2], roi[2:], ROI_COLOR, 1, cv2.LINE_AA)


@lru_cache(maxsize=None)
def _load_preview_font(size):
    """폰트 파일 읽기가 느려서 크기별로 한 번만 읽는다."""
    for path in PREVIEW_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


PREVIEW_LABEL_WIDTH = 120
PREVIEW_LABEL_HEIGHT = 20


@lru_cache(maxsize=512)
def _render_preview_label(text, color):
    """라벨 한 줄을 BGR 이미지로 그린다. 같은 문구는 다시 그리지 않고 재사용한다."""
    font = _load_preview_font(14)
    if font is None:
        label = np.full((PREVIEW_LABEL_HEIGHT, PREVIEW_LABEL_WIDTH, 3), 255, dtype=np.uint8)
        cv2.putText(label, text.replace("·", "/"), (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color[::-1], 1, cv2.LINE_AA)
        return label
    # OpenCV 기본 글꼴은 한글을 못 그리므로 라벨은 PIL로 그린다.
    label = Image.new("RGB", (PREVIEW_LABEL_WIDTH, PREVIEW_LABEL_HEIGHT), (255, 255, 255))
    ImageDraw.Draw(label).text((6, 2), text, font=font, fill=color)
    return np.asarray(label)[:, :, ::-1].copy()


# 슬롯 수별 위치 이름(한글, 한글 폰트가 없을 때 쓰는 영문)이다.
SLOT_POSITION_NAMES = {
    3: (("좌측", "Left"), ("중앙", "Center"), ("우측", "Right")),
    4: (("좌측", "Left"), ("중앙 좌", "Center L"), ("중앙 우", "Center R"), ("우측", "Right")),
}


def get_slot_position_name(slot_index, slot_count, korean=True):
    """슬롯 번호를 좌측/중앙/우측 같은 위치 이름으로 바꾼다."""
    names = SLOT_POSITION_NAMES.get(slot_count)
    if names is None or not 0 <= slot_index < len(names):
        return f"Slot {slot_index + 1}"
    return names[slot_index][0 if korean else 1]


def create_slot_ball_preview(image, measurements, slot_count):
    """슬롯 순서대로 볼 ROI 크롭과 '위치 (W X H)', 판정을 표시한다."""
    if not measurements:
        return None
    source = _as_color_image(image)
    image_height, image_width = source.shape[:2]
    scale, box_size, gap = 3, PREVIEW_LABEL_WIDTH, 8
    label_top = box_size + 2
    output = np.full(
        (label_top + 2 * PREVIEW_LABEL_HEIGHT + 4,
         len(measurements) * box_size + gap * (len(measurements) - 1), 3),
        255, dtype=np.uint8,
    )
    korean = _load_preview_font(14) is not None
    for index, (slot_index, ball_width, ball_height, box, status) in enumerate(measurements):
        abnormal = status != BALL_STATUS_OK
        border = (65, 65, 220) if abnormal else ROI_COLOR
        tile = output[:, index * (box_size + gap):index * (box_size + gap) + box_size]
        if box is not None:
            left, top, right, bottom = pad_ball_roi(box, source.shape)
            crop = source[max(0, top):min(image_height, bottom + 1),
                          max(0, left):min(image_width, right + 1)]
            ratio = min(scale, box_size / max(crop.shape[:2]))
            crop = cv2.resize(
                crop, (max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio))),
                interpolation=cv2.INTER_NEAREST,
            )
            y0, x0 = (box_size - crop.shape[0]) // 2, (box_size - crop.shape[1]) // 2
            tile[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        else:
            tile[:box_size] = 235
        cv2.rectangle(tile, (0, 0), (box_size - 1, box_size - 1), border, 1)
        title = get_slot_position_name(slot_index, slot_count, korean)
        if box is not None:
            title = f"{title} ({ball_width} X {ball_height})"
        tile[label_top:label_top + PREVIEW_LABEL_HEIGHT] = _render_preview_label(
            title, (40, 40, 40)
        )
        tile[label_top + PREVIEW_LABEL_HEIGHT:label_top + 2 * PREVIEW_LABEL_HEIGHT] = (
            _render_preview_label(
                status if korean else BALL_STATUS_ENGLISH[status],
                (220, 65, 65) if abnormal else (40, 140, 70),
            )
        )
    return output


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
