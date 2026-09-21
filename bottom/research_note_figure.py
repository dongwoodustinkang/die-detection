"""Bottom 원형 검출의 코드 흐름과 실제 TIFF 판정 결과를 한 장 그림으로 만든다.

사용 예:
    conda run -n keti python -m bottom.research_note_figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from general import to_bgr
from .pipeline import _binarize_a, run_bottom_detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OK_PATH = PROJECT_ROOT / "dataset/테스트/Bottom/OK_bottom_20260618_034734_658.tiff"
DEFAULT_NG_PATH = PROJECT_ROOT / "dataset/테스트/Bottom/NG_bottom_20260624_150136_734.tiff"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/bottom_detection_research_note.png"

NAVY = (20, 54, 90)
BLUE = (51, 103, 166)
GRAY = (93, 104, 117)
LIGHT_GRAY = (239, 243, 247)
MID_GRAY = (207, 216, 225)
GREEN = (42, 137, 85)
RED = (194, 63, 63)
AMBER = (205, 133, 28)
WHITE = (255, 255, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """macOS 기본 한글 폰트를 우선 사용하고, 다른 환경에서는 fallback 한다."""
    candidates = (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=1 if bold and path.endswith(".ttc") else 0)
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, size: int,
               color=NAVY, bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, text, font=_font(size, bold), fill=color, anchor=anchor)


def _paste_contained(canvas: Image.Image, image: np.ndarray, box: Tuple[int, int, int, int]) -> None:
    """OpenCV BGR 영상을 종횡비를 보존해 box 안에 붙인다."""
    x, y, width, height = box
    source = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2RGB)
    source_image = Image.fromarray(source)
    source_image.thumbnail((width, height), Image.Resampling.LANCZOS)
    left = x + (width - source_image.width) // 2
    top = y + (height - source_image.height) // 2
    canvas.paste(source_image, (left, top))


def _panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str,
           title_color=NAVY) -> None:
    x, y, width, height = box
    draw.rounded_rectangle((x, y, x + width, y + height), radius=18, fill=WHITE, outline=MID_GRAY, width=2)
    _draw_text(draw, (x + 18, y + 16), title, 25, title_color, bold=True)


def _arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color=BLUE) -> None:
    draw.line((start, end), fill=color, width=5)
    x, y = end
    draw.polygon(((x, y), (x - 15, y - 9), (x - 15, y + 9)), fill=color)


def _chip_overlay(result) -> np.ndarray:
    image = to_bgr(result.raw_image_a)
    if result.chip is None:
        return image
    corners = np.rint(result.chip.corners).astype(np.int32)
    cv2.polylines(image, [corners], True, (255, 145, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(image, tuple(int(round(value)) for value in result.chip.center),
                   (255, 145, 0), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    return image


def _status_lines(result) -> Iterable[Tuple[str, Tuple[int, int, int]]]:
    if result.all_regions_accepted:
        yield "판정: 원 후보 검출 (6 / 6)", GREEN
    else:
        yield f"판정: 검토 필요 ({result.candidate_count} 통과 / {len(result.circle_regions)})", RED
    for region in result.circle_regions:
        if region.status != "candidate":
            primary = region.primary
            if primary is None:
                yield f"• {region.name}: 성분 미검출", RED
            else:
                yield f"• {region.name}: 면적 {primary.area:.1f} px² (허용 300–450)", RED


def create_research_note_figure(ok_path: Path = DEFAULT_OK_PATH, ng_path: Path = DEFAULT_NG_PATH,
                                output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Bottom 코드의 실제 처리 결과를 포함한 연구노트용 PNG를 저장한다."""
    ok_result = run_bottom_detection(ok_path)
    ng_result = run_bottom_detection(ng_path)
    ok_binary, _ = _binarize_a(ok_result.raw_image_a)

    canvas = Image.new("RGB", (2400, 1580), WHITE)
    draw = ImageDraw.Draw(canvas)
    _draw_text(draw, (1200, 66), "반도체 Die Bottom 원형 검출 로직", 52, NAVY, True, "ma")
    _draw_text(draw, (1200, 125), "구현 코드(bottom/pipeline.py · bottom/circles.py) 및 실제 TIFF 적용 결과", 24, GRAY, anchor="ma")

    # 코드 단계를 상단에 배치한다. 각 실제 영상은 아래 사례 패널에 제시한다.
    stages = (
        ("1. A TIFF 입력", "A/B 2-page TIFF\nA 페이지만 검출", "입력"),
        ("2. 칩 위치 검출", "Gaussian 3×3\n배경 + Otsu 기반 이진화", "칩"),
        ("3. 기준 ROI 생성", "칩 좌표계 추종\n6개 · 40×40 px", "ROI"),
        ("4. 검은 성분 측정", "Contour area\nCircularity · aspect", "윤곽"),
        ("5. 후보 판정·출력", "6/6 통과 또는 검토\n좌표 · 면적 · 사유", "결과"),
    )
    card_width, card_height, gap, start_x, y = 420, 220, 35, 95, 205
    for index, (title, detail, badge) in enumerate(stages):
        x = start_x + index * (card_width + gap)
        _panel(draw, (x, y, card_width, card_height), title)
        draw.rounded_rectangle((x + 20, y + 65, x + 95, y + 108), radius=12, fill=LIGHT_GRAY)
        _draw_text(draw, (x + 57, y + 87), badge, 19, BLUE, True, "mm")
        for line_index, line in enumerate(detail.splitlines()):
            _draw_text(draw, (x + 116, y + 70 + line_index * 38), line, 21, GRAY)
        if index < len(stages) - 1:
            _arrow(draw, (x + card_width + 7, y + card_height // 2),
                   (x + card_width + gap - 8, y + card_height // 2))

    _draw_text(draw, (120, 470), "판정 조건", 30, NAVY, True)
    rules = (
        "ROI 위치: 폭 10%·90% × 높이 10%·50%·90%",
        "검은 성분만 추출 (칩 외부 및 10 px² 미만 노이즈 제외)",
        "면적: 300–450 px²  |  원형도: ≥ 0.70",
        "가로/세로 비: ≥ 0.70  |  예상 중심 거리: ≤ 12 px",
        "경계 접촉 · 복수 후보 · 조건 미달은 검토로 보존",
    )
    for index, rule in enumerate(rules):
        draw.ellipse((125, 523 + index * 42, 137, 535 + index * 42), fill=BLUE)
        _draw_text(draw, (153, 515 + index * 42), rule, 22, GRAY)

    feedback_y = 515 + len(rules) * 42 + 18
    draw.rounded_rectangle((120, feedback_y, 1000, feedback_y + 52), radius=14, fill=(255, 246, 229))
    _draw_text(draw, (560, feedback_y + 26), "조건 미통과 → threshold / ROI / 필터 기준을 재검토", 21, AMBER, True, "mm")

    # 두 입력에 대한 실제 코드 출력: 왼쪽은 A 기반 중간 단계, 오른쪽은 최종 ROI 판정이다.
    sample_y, sample_height, sample_width = 810, 690, 1092
    for index, (label, path, result, accent) in enumerate((
        ("OK 사례", ok_path, ok_result, GREEN),
        ("NG 사례", ng_path, ng_result, RED),
    )):
        panel_x = 95 + index * 1213
        _panel(draw, (panel_x, sample_y, sample_width, sample_height), label, accent)
        _draw_text(draw, (panel_x + 20, sample_y + 58), path.name, 17, GRAY)

        image_y = sample_y + 94
        image_box_width, image_box_height = 322, 305
        columns = (
            ("A 원본", result.raw_image_a),
            ("칩 기준", _chip_overlay(result)),
            ("ROI·실제 윤곽", result.source_visualization),
        )
        for image_index, (caption, image) in enumerate(columns):
            image_x = panel_x + 22 + image_index * 356
            draw.rectangle((image_x, image_y, image_x + image_box_width, image_y + image_box_height), fill=(20, 20, 20))
            _paste_contained(canvas, image, (image_x, image_y, image_box_width, image_box_height))
            _draw_text(draw, (image_x + image_box_width // 2, image_y + image_box_height + 18), caption, 20, GRAY, anchor="ma")

        line_y = sample_y + 470
        for line_index, (line, color) in enumerate(_status_lines(result)):
            _draw_text(draw, (panel_x + 24, line_y + line_index * 40), line, 25 if line_index == 0 else 21,
                       color, line_index == 0)
        if index == 0:
            _draw_text(draw, (panel_x + 24, line_y + 92),
                       f"threshold {result.threshold:.1f} · chip center ({result.chip.center[0]:.1f}, {result.chip.center[1]:.1f})",
                       20, GRAY)
        else:
            _draw_text(draw, (panel_x + 24, line_y + 92),
                       "원형도는 0.851로 통과했으나 면적 조건에서 탈락", 20, GRAY)

    _draw_text(draw, (1200, 1540), "A 페이지만 판정에 사용하며, B 페이지는 원본 참조용이다.", 20, GRAY, anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    saved_path = create_research_note_figure()
    print(saved_path)
