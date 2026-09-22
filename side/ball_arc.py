"""숄더 볼 크롭의 하단 원호를 측정한다. 완전한 원의 원형도나 OK/NG 판정은 아니다.

모든 측정은 확대 전 A 영상 좌표계(px)에서 수행한다. 10–20 px 반경 제한과
오차/대칭성/관측 범위 기준은 비교 실험용이며 생산 판정 전에 검증해야 한다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from general import to_bgr, to_grayscale
from .ball import get_ball_square_bounds


MIN_RADIUS = 10.0
MAX_RADIUS = 20.0
MAX_RMSE = 1.5
MAX_SYMMETRY_ERROR = 2.0
MIN_ARC_DEGREES = 70.0
MIN_POINTS = 12


@dataclass
class BallArcMeasurement:
    bounds: Tuple[int, int, int, int]
    enhanced: np.ndarray
    threshold: float
    # 측정 좌표는 원본 A 영상 기준. enhanced만 크롭 좌표계다.
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    rmse: Optional[float] = None
    arc_degrees: Optional[float] = None
    symmetry_error: Optional[float] = None
    coverage: float = 0.0
    reasons: Tuple[str, ...] = ()

    @property
    def status(self):
        return "missing" if self.center is None else "review" if self.reasons else "fitted"


def fit_lower_circle(points):
    """여러 하단 접점의 방사 거리 오차를 최소화하는 제한 반경 원을 구한다.

    대수적 초기값과 세 반경 초기값에서 제한 Gauss–Newton을 실행한다.
    손상점을 숨기지 않도록 모든 입력 접점에 동일한 가중치를 적용한다.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < MIN_POINTS or np.ptp(points[:, 0]) < 12 or np.ptp(points[:, 1]) < 2:
        return None
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    tip_x = np.median(points[points[:, 1] >= ymax - 1, 0])
    local = points - points.mean(axis=0)
    coefficients = np.linalg.lstsq(
        np.column_stack((2 * local, np.ones(len(local)))),
        np.sum(local ** 2, axis=1), rcond=None,
    )[0]
    center = coefficients[:2] + points.mean(axis=0)
    radius = np.sqrt(max(0, coefficients[2] + np.sum(coefficients[:2] ** 2)))
    seeds = [(center[0], center[1], radius)]
    seeds.extend((tip_x, ymax - r, r) for r in (MIN_RADIUS, 15.0, MAX_RADIUS))
    lower = np.array((xmin, ymin - MAX_RADIUS, MIN_RADIUS))
    upper = np.array((xmax, ymax - 2, MAX_RADIUS))

    def errors(parameters):
        return np.linalg.norm(points - parameters[:2], axis=1) - parameters[2]

    best = None
    for seed in seeds:
        parameters = np.clip(seed, lower, upper)
        for _ in range(30):
            delta = parameters[:2] - points
            distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
            jacobian = np.column_stack((delta / distance[:, None], -np.ones(len(points))))
            residual = distance - parameters[2]
            step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            step = np.clip(step, -3, 3)
            loss = np.mean(residual ** 2)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                trial = np.clip(parameters + scale * step, lower, upper)
                if np.mean(errors(trial) ** 2) < loss - 1e-10:
                    parameters, accepted = trial, True
                    break
            if not accepted or np.linalg.norm(step) < 1e-5:
                break
        loss = float(np.mean(errors(parameters) ** 2))
        if best is None or loss < best[0]:
            best = (loss, parameters)
    loss, parameters = best
    return tuple(parameters[:2]), float(parameters[2]), float(np.sqrt(loss))


def measure_ball_crop(crop, origin=(0, 0), expected_x=None):
    """약한 선명화 → Otsu 어두운 성분 → 열별 하단 접점 → 원호 피팅."""
    gray = to_grayscale(crop)
    height, width = gray.shape
    if not height or not width:
        raise ValueError("숄더 볼 원호 측정에는 비어 있지 않은 크롭이 필요합니다.")
    left, top = origin
    bounds = (left, top, left + width - 1, top + height - 1)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.8)
    enhanced = cv2.addWeighted(gray, 1.4, smooth, -0.4, 0)
    threshold, dark = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = BallArcMeasurement(bounds, enhanced, float(threshold))
    if min(height, width) < 5 or np.ptp(gray) < 10:
        result.reasons = ("명암 대비 또는 크롭 크기 부족",)
        return result

    # 하단 중앙에 위치한 성분을 선택해 분리된 먼지/픽셀 잡음을 배제한다.
    expected_x = (width - 1) / 2 if expected_x is None else expected_x
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 10 or y + h - 1 < height / 2:
            continue
        distance = max(x - expected_x, expected_x - (x + w - 1), 0)
        candidates.append((distance, -(y + h), -area, label))
    if not candidates or min(candidates)[0] > MAX_RADIUS:
        result.reasons = ("하단의 유효한 검은 성분 없음",)
        return result
    mask = labels == min(candidates)[-1]
    xs = np.flatnonzero(mask.any(axis=0))
    ys = height - 1 - np.argmax(mask[::-1, xs], axis=0)
    # 표면에 붙은 가로 띠를 제외하고 볼 끝의 20 px 깊이만 관찰한다.
    selected = (ys >= max(height * 0.45, ys.max() - MAX_RADIUS))
    points = np.column_stack((xs[selected], ys[selected])).astype(float)
    result.points = points + origin
    fitted = fit_lower_circle(points)
    if fitted is None:
        result.reasons = ("하단 윤곽의 점 수·폭 또는 곡률 부족",)
        return result
    center, result.radius, result.rmse = fitted
    cx, cy = center
    result.center = (cx + left, cy + top)
    result.coverage = float(len(points) / (np.ptp(points[:, 0]) + 1))
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    result.arc_degrees = float(np.degrees(np.ptp(angles)))
    reach = min(cx - points[0, 0], points[-1, 0] - cx)
    if reach >= 3:
        offsets = np.arange(1, reach + 0.01)
        left_y = np.interp(cx - offsets, points[:, 0], points[:, 1])
        right_y = np.interp(cx + offsets, points[:, 0], points[:, 1])
        result.symmetry_error = float(np.mean(np.abs(left_y - right_y)))
    reasons = []
    if result.rmse > MAX_RMSE:
        reasons.append(f"원호 오차 > {MAX_RMSE:g} px")
    if result.symmetry_error is None or result.symmetry_error > MAX_SYMMETRY_ERROR:
        reasons.append("좌우 비대칭 또는 대칭 비교 범위 부족")
    if result.arc_degrees < MIN_ARC_DEGREES:
        reasons.append("하단 원호 관측 범위 부족")
    if np.any(points[:, 1] < cy - 1):
        reasons.append("하단 원호 밖의 측면 접점 포함")
    if result.coverage < 0.9:
        reasons.append("하단 윤곽 끊김")
    if np.any((points[:, 0] <= 0) | (points[:, 0] >= width - 1) | (points[:, 1] >= height - 1)):
        reasons.append("윤곽이 크롭 경계에 닿음")
    if result.radius <= MIN_RADIUS + 0.05 or result.radius >= MAX_RADIUS - 0.05:
        reasons.append("반경 탐색 한계에 도달")
    result.reasons = tuple(reasons)
    return result


def analyze_ball_crops(image, points, surface_bottom_y_by_x):
    """기존과 같은 크롭을 사용한다. 입력 영상과 기존 크롭을 변경하지 않는다."""
    results = []
    for point in points:
        left, top, right, bottom = get_ball_square_bounds(
            image.shape[:2], point, surface_bottom_y_by_x.get(point[0]),
        )
        results.append(measure_ball_crop(
            image[top:bottom + 1, left:right + 1],
            (left, top), expected_x=point[0] - left,
        ))
    return results


def create_ball_arc_preview(results: List[BallArcMeasurement]):
    """선명화 영상 + 실제 하단 접점 + 추정 원. 원/접점은 확대 전에 측정한다."""
    if not results:
        return None
    tile_size, cell_width, cell_height = 168, 184, 234
    output = np.full((cell_height, cell_width * len(results), 3), 250, np.uint8)
    for index, result in enumerate(results):
        left, top, _, _ = result.bounds
        height, width = result.enhanced.shape
        scale = min(tile_size / width, tile_size / height)
        tile = cv2.resize(to_bgr(result.enhanced), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        color = (45, 155, 70) if result.status == "fitted" else (25, 125, 225)
        for point in result.points:
            xy = tuple(np.rint((point - (left, top)) * scale).astype(int))
            cv2.circle(tile, xy, 2, (230, 165, 15), -1)
        if result.center is not None:
            center = tuple(np.rint((np.array(result.center) - (left, top)) * scale).astype(int))
            cv2.circle(tile, center, int(round(result.radius * scale)), color, 1, cv2.LINE_AA)
            cv2.drawMarker(tile, center, color, cv2.MARKER_CROSS, 8, 1)
        x = index * cell_width + 8
        output[4:4 + tile.shape[0], x:x + tile.shape[1]] = tile
        lines = (
            f"{index + 1}  {result.status.upper()}",
            f"R {result.radius:.1f}  E {result.rmse:.2f} px" if result.radius is not None else "No reliable arc",
            f"S {result.symmetry_error:.2f}  {result.arc_degrees:.0f} deg" if result.symmetry_error is not None else "S --",
        )
        for row, line in enumerate(lines):
            cv2.putText(output, line, (x, 188 + row * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return output
