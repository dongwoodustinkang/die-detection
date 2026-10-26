"""10/15 NG 마스크와 10/06 OK로 Side 표면 결함 검출(분할→박스) 데이터셋을 만든다."""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from general import load_ab_tiff_pages, to_grayscale # 이미지 불러오기
from side import SURFACE_CROP_VERSION, extract_surface_tiles # 표면 위치 찾기
from tools.make_data import (
    LABELS, SPLITS, check_output, display_path, parse_label_and_view,
    ratios_to_counts, write_csv, write_png,
)


@dataclass
class Sample:
    """고정 크기로 자른 A/B 타일과 결함 마스크."""
    sample_id: str
    label: str
    view: str
    split: str
    source: Path
    mask_source: str = ""
    gray_a: np.ndarray = None
    gray_b: np.ndarray = None
    mask: np.ndarray = None  # 0 / 255, OK는 전부 0
    crop: tuple = ()         # 원본 기준 (left, top, right, bottom)
    surface: tuple = ()      # 크롭 기준 표면 타일 (left, top, right, bottom)
    mask_outside: int = 0    # 크롭 밖으로 잘린 결함 픽셀 수
    boxes: list = field(default_factory=list)


# 입출력 경로
DEFAULT_NG_INPUT = ROOT / "dataset" / "2026_10_15"
DEFAULT_OK_SAMPLES = ROOT / "dataset" / "infer" / "side_surface_v1" / "samples.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "infer" / "side_surface_od_v1"

VIEWS = ("Front", "Right")  # 10/15 NG가 있는 방향만 사용
INPUT_KINDS = ("A", "B", "mask")

# 표면 타일(약 33×175) 중심 기준 고정 크롭. 위아래·좌우 여백으로 가장자리 결함이 잘리지 않게 한다.
CROP_HEIGHT, CROP_WIDTH = 48, 192
DEFAULT_MASK_THRESHOLD = 3  # JSON에 blurredMaskThresholdByte가 없을 때
MERGE_KERNEL = 5            # 가는 사선이 끊긴 조각을 한 박스로 합치는 팽창 크기
MIN_BOX_PIXELS = 4          # 이보다 작은 성분은 박스로 만들지 않는다 (마스크는 그대로 둠)

SAMPLE_FIELDS = (
    "sample_id", "label", "target", "view", "split", "source", "mask_source",
    "crop_left", "crop_top", "crop_right", "crop_bottom",
    "surface_left", "surface_top", "surface_right", "surface_bottom",
    "ng_pixels", "mask_outside", "box_count",
)
BOX_FIELDS = ("sample_id", "split", "view", "box_index", "x1", "y1", "x2", "y2", "pixels")
FAILED_FIELDS = ("source", "label", "reason")

# ───────────────────────────────────────────────

def fixed_crop(image, center_x, center_y):
    """중심 기준 CROP_HEIGHT×CROP_WIDTH로 자른다. 이미지 밖은 가장자리 픽셀로 채운다."""
    left, top = center_x - CROP_WIDTH // 2, center_y - CROP_HEIGHT // 2
    right, bottom = left + CROP_WIDTH, top + CROP_HEIGHT
    height, width = image.shape[:2]
    padded = cv2.copyMakeBorder(
        image,
        max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width),
        cv2.BORDER_REPLICATE,
    )
    shift_x, shift_y = max(0, -left), max(0, -top)
    tile = padded[top + shift_y: bottom + shift_y, left + shift_x: right + shift_x]
    return tile, (left, top, right, bottom)


def find_mask(tiff_path):
    """TIFF 옆 JSON이 가리키는 마스크와 이진화 기준을 찾는다.
    파일이 없거나 비어 있으면(0바이트) `{stem}_mask*.png` 중 비어 있지 않은 하나를 쓴다."""
    meta_path = tiff_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    threshold = int(meta.get("blurredMaskThresholdByte", DEFAULT_MASK_THRESHOLD))
    path = tiff_path.parent / meta.get("maskFile", f"{tiff_path.stem}_mask.png")
    if not path.exists() or path.stat().st_size == 0:
        candidates = [
            candidate for candidate in sorted(tiff_path.parent.glob(f"{tiff_path.stem}_mask*.png"))
            if candidate.stat().st_size > 0
        ]
        if len(candidates) != 1:
            return None, threshold
        path = candidates[0]
    return path, threshold


def mask_boxes(mask):
    """팽창으로 가까운 조각을 묶은 뒤, 묶음마다 원래 결함 픽셀을 감싸는 박스를 만든다."""
    binary = (mask > 0).astype(np.uint8)
    merged = cv2.dilate(binary, np.ones((MERGE_KERNEL, MERGE_KERNEL), np.uint8))
    count, labels = cv2.connectedComponents(merged, connectivity=8)
    boxes = []
    for index in range(1, count):
        ys, xs = np.nonzero((labels == index) & (binary > 0))
        if len(xs) < MIN_BOX_PIXELS:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, len(xs)))
    return boxes


def crop_sample(sample, mask_path=None, threshold=DEFAULT_MASK_THRESHOLD):
    """표면 위치를 찾아 A/B/마스크를 같은 좌표로 자른다. 실패하면 사유 문자열을 돌려준다."""
    image_a, image_b = load_ab_tiff_pages(sample.source)
    tiles = extract_surface_tiles(image_a, image_b)
    if tiles is None:
        return "A 컨투어 없음"
    left, top, right, bottom = tiles.bounds
    center_x, center_y = (left + right) // 2, (top + bottom) // 2

    sample.gray_a, sample.crop = fixed_crop(to_grayscale(image_a), center_x, center_y)
    sample.gray_b, _ = fixed_crop(to_grayscale(image_b), center_x, center_y)
    crop_left, crop_top = sample.crop[:2]
    sample.surface = (left - crop_left, top - crop_top, right - crop_left, bottom - crop_top)

    if mask_path is None:
        sample.mask = np.zeros((CROP_HEIGHT, CROP_WIDTH), np.uint8)
        return None
    full = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if full is None or full.shape != image_a.shape[:2]:
        return f"마스크를 읽을 수 없거나 크기가 다름: {mask_path.name}"
    binary = np.where(full >= threshold, 255, 0).astype(np.uint8)
    # 이미지 밖은 결함이 아니므로 0으로 채운다
    padded = cv2.copyMakeBorder(binary, CROP_HEIGHT, CROP_HEIGHT, CROP_WIDTH, CROP_WIDTH, cv2.BORDER_CONSTANT, value=0)
    crop_right, crop_bottom = sample.crop[2:]
    sample.mask = padded[crop_top + CROP_HEIGHT: crop_bottom + CROP_HEIGHT, crop_left + CROP_WIDTH: crop_right + CROP_WIDTH]
    sample.mask_outside = int((binary > 0).sum() - (sample.mask > 0).sum())
    sample.mask_source = display_path(mask_path)
    if not (sample.mask > 0).any():
        return "크롭 안에 결함 픽셀이 없음"
    sample.boxes = mask_boxes(sample.mask)
    return None

# ───────────────────────────────────────────────

def collect_ng(ng_dir, ratios, seed):
    """방향별로 섞어 비율대로 train/val/test를 정한 NG 샘플 목록."""
    rng = np.random.default_rng(seed)
    samples = []
    for view in VIEWS:
        paths = []
        for path in sorted(ng_dir.rglob("*.tif*")):
            label, file_view = parse_label_and_view(path)
            if label == "NG" and file_view == view and not path.name.startswith("."):
                paths.append(path)
        rng.shuffle(paths)
        counts = ratios_to_counts(len(paths), ratios)
        start = 0
        for split in SPLITS:
            for path in paths[start: start + counts[split]]:
                samples.append(Sample(sample_id=path.stem, label="NG", view=view, split=split, source=path))
            start += counts[split]
    return samples


def collect_ok(samples_csv):
    """기존 분류기 데이터셋의 split을 그대로 따르는 OK 샘플 목록 (VIEWS만)."""
    with open(samples_csv, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [
        Sample(sample_id=row["sample_id"], label="OK", view=row["view"], split=row["split"], source=ROOT / row["source"])
        for row in rows if row["label"] == "OK" and row["view"] in VIEWS
    ]


def crop_all(samples):
    done, failures, renamed_masks = [], [], []
    for index, sample in enumerate(samples, 1):
        mask_path, threshold = (None, DEFAULT_MASK_THRESHOLD)
        reason = None
        if sample.label == "NG":
            mask_path, threshold = find_mask(sample.source)
            if mask_path is None:
                reason = "마스크 파일 없음"
            elif mask_path.name != f"{sample.source.stem}_mask.png":
                renamed_masks.append(display_path(mask_path))
        if reason is None:
            try:
                reason = crop_sample(sample, mask_path, threshold)
            except Exception as error:
                reason = repr(error)
        if reason:
            failures.append(dict(source=display_path(sample.source), label=sample.label, reason=reason))
        else:
            done.append(sample)
        if index % 500 == 0 or index == len(samples):
            print(f"  Crop {index}/{len(samples)}  : 성공 {len(done)} · 실패 {len(failures)}")
    return done, failures, renamed_masks

# ───────────────────────────────────────────────

def reset_output(out_dir):
    """출력 폴더를 비우고 <종류>/<split>/<라벨> 폴더를 만든다."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for kind in INPUT_KINDS:
        for split in SPLITS:
            for label in LABELS:
                (out_dir / kind / split / label).mkdir(parents=True, exist_ok=True)


def save_images(samples, out_dir):
    for index, sample in enumerate(samples, 1):
        relative = Path(sample.split) / sample.label / f"{sample.sample_id}.png"
        write_png(out_dir / "A" / relative, sample.gray_a)
        write_png(out_dir / "B" / relative, sample.gray_b)
        write_png(out_dir / "mask" / relative, sample.mask)
        if index % 1000 == 0 or index == len(samples):
            print(f"  저장 {index}/{len(samples)}")


def sample_row(sample):
    return dict(
        sample_id=sample.sample_id, label=sample.label, target=int(sample.label == "NG"),
        view=sample.view, split=sample.split, source=display_path(sample.source), mask_source=sample.mask_source,
        **dict(zip(("crop_left", "crop_top", "crop_right", "crop_bottom"), sample.crop)),
        **dict(zip(("surface_left", "surface_top", "surface_right", "surface_bottom"), sample.surface)),
        ng_pixels=int((sample.mask > 0).sum()), mask_outside=sample.mask_outside, box_count=len(sample.boxes),
    )


def box_rows(samples):
    for sample in samples:
        for index, (x1, y1, x2, y2, pixels) in enumerate(sample.boxes):
            yield dict(sample_id=sample.sample_id, split=sample.split, view=sample.view,
                       box_index=index, x1=x1, y1=y1, x2=x2, y2=y2, pixels=pixels)


def summarize(args, samples, failures, renamed_masks):
    ng = [sample for sample in samples if sample.label == "NG"]
    boxes = [box for sample in ng for box in sample.boxes]
    return dict(
        created_at=datetime.now().isoformat(timespec="seconds"),
        crop_version=SURFACE_CROP_VERSION,
        crop_size=dict(height=CROP_HEIGHT, width=CROP_WIDTH),
        ng_input=display_path(args.ng_input),
        ok_samples=display_path(args.ok_samples),
        views=list(VIEWS),
        seed=args.seed,
        ng_ratios=dict(train=args.train, val=args.val, test=args.test),
        box_rule=dict(merge_kernel=MERGE_KERNEL, min_box_pixels=MIN_BOX_PIXELS),
        total=len(samples),
        failed=len(failures),
        renamed_masks=renamed_masks,
        by_split_view_label=dict(sorted(Counter(
            f"{sample.split}/{sample.view}/{sample.label}" for sample in samples
        ).items())),
        ng_mask_outside=dict(
            samples=sum(sample.mask_outside > 0 for sample in ng),
            pixels=int(sum(sample.mask_outside for sample in ng)),
        ),
        boxes_per_ng=dict(sorted(Counter(len(sample.boxes) for sample in ng).items())),
        box_total=len(boxes),
    )

# ───────────────────────────────────────────────

HELP_EPILOG = f"""\
사용 예:
  python tools/make_sg_data.py                   기본 설정으로 생성
  python tools/make_sg_data.py --overwrite       기존 결과를 지우고 다시 생성

출력 구조:
  <output>/A/<split>/<OK|NG>/<sample_id>.png     A 그레이스케일 {CROP_HEIGHT}×{CROP_WIDTH}
  <output>/B/<split>/<OK|NG>/<sample_id>.png     B 그레이스케일 {CROP_HEIGHT}×{CROP_WIDTH}
  <output>/mask/<split>/<OK|NG>/<sample_id>.png  결함 마스크 0/255 (OK는 전부 0)
  <output>/samples.csv        샘플별 split·크롭 좌표·표면 위치·결함 픽셀 수
  <output>/boxes.csv          마스크에서 만든 결함 박스 (크롭 좌표, x2·y2는 미포함)
  <output>/failed.csv         실패 목록과 사유
  <output>/dataset_info.json  개수·설정 요약
"""


def main():
    parser = argparse.ArgumentParser(
        prog="python tools/make_sg_data.py",
        description=__doc__.splitlines()[0],
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ng-input", type=Path, default=DEFAULT_NG_INPUT, metavar="DIR",
                        help=f"NG TIFF·마스크·JSON 폴더 (기본: {display_path(DEFAULT_NG_INPUT)})")
    parser.add_argument("--ok-samples", type=Path, default=DEFAULT_OK_SAMPLES, metavar="CSV",
                        help=f"OK split을 가져올 samples.csv (기본: {display_path(DEFAULT_OK_SAMPLES)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, metavar="DIR",
                        help=f"출력 폴더 (기본: {display_path(DEFAULT_OUTPUT)})")
    parser.add_argument("--train", type=float, default=0.75, metavar="R", help="NG train 비율 (기본: 0.75)")
    parser.add_argument("--val", type=float, default=0.125, metavar="R", help="NG val 비율 (기본: 0.125)")
    parser.add_argument("--test", type=float, default=0.125, metavar="R", help="NG test 비율 (기본: 0.125)")
    parser.add_argument("--seed", type=int, default=7, metavar="N", help="NG 분할 셔플 seed (기본: 7)")
    parser.add_argument("--overwrite", action="store_true", help="출력 폴더가 있으면 지우고 다시 만든다")
    args = parser.parse_args()

    check_output(args.output, args.overwrite)
    ratios = dict(train=args.train, val=args.val, test=args.test)

    samples = collect_ng(args.ng_input, ratios, args.seed) + collect_ok(args.ok_samples)
    print(f"⬜⬛️⬛️ : 샘플 {len(samples)}개 확보 " + str(dict(Counter(f"{s.label}/{s.view}" for s in samples))))
    samples, failures, renamed_masks = crop_all(samples)

    print(f"⬜⬜⬛️ : 저장 → {args.output}")
    reset_output(args.output)
    save_images(samples, args.output)
    write_csv(args.output / "samples.csv", SAMPLE_FIELDS, [sample_row(sample) for sample in samples])
    write_csv(args.output / "boxes.csv", BOX_FIELDS, box_rows(samples))
    write_csv(args.output / "failed.csv", FAILED_FIELDS, failures)

    print(f"⬜⬜⬜ : 요약")
    info = summarize(args, samples, failures, renamed_masks)
    (args.output / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
