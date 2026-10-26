"""10/15 NG 마스크와 10/06 OK로 Side 표면 결함 Object Detection(박스) 데이터셋을 만든다.

크롭은 검사 프로그램(app.py → side.extract_surface_tiles)과 같은 표면 타일을 그대로 쓰고,
NG 마스크도 같은 좌표로 잘라 결함 박스를 만든다. split은 make_sg_data.py와 같다 (seed 7, 3000/500/500).
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from general import load_ab_tiff_pages, to_grayscale # 이미지 불러오기
from side import SURFACE_CROP_VERSION, extract_surface_tiles # 표면 위치 찾기 (app.py와 같은 크롭)
from tools.make_data import check_output, display_path, write_csv
from tools.make_sg_data import (
    BOX_FIELDS, DEFAULT_MASK_THRESHOLD, DEFAULT_OK_SAMPLES, DEFAULT_NG_INPUT, FAILED_FIELDS,
    MERGE_KERNEL, MIN_BOX_PIXELS, VIEWS, Sample, box_rows, collect_ng, collect_ok, find_mask,
    mask_boxes, reset_output, save_images,
)

DEFAULT_OUTPUT = ROOT / "dataset" / "infer" / "side_surface_det_v1"
CROP_MODE = "surface"  # 고정 크기가 아니라 표면 타일(약 33×175) 그대로

SAMPLE_FIELDS = (
    "sample_id", "label", "target", "view", "split", "source", "mask_source",
    "height", "width", "left", "top", "right", "bottom",
    "ng_pixels", "mask_outside", "box_count",
)

# ───────────────────────────────────────────────

def crop_sample(sample, mask_path=None, threshold=DEFAULT_MASK_THRESHOLD):
    """표면 타일 좌표로 A/B/마스크를 자른다. 실패하면 사유 문자열을 돌려준다."""
    image_a, image_b = load_ab_tiff_pages(sample.source)
    tiles = extract_surface_tiles(image_a, image_b)
    if tiles is None:
        return "A 컨투어 없음"
    sample.gray_a, sample.gray_b = tiles.gray_a, tiles.gray_b
    sample.crop = tiles.bounds  # 원본 기준 (left, top, right, bottom)
    left, top, right, bottom = tiles.bounds

    if mask_path is None:
        sample.mask = np.zeros(sample.gray_a.shape[:2], np.uint8)
        return None
    full = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if full is None or full.shape != image_a.shape[:2]:
        return f"마스크를 읽을 수 없거나 크기가 다름: {mask_path.name}"
    binary = np.where(full >= threshold, 255, 0).astype(np.uint8)
    sample.mask = binary[top:bottom, left:right]
    sample.mask_outside = int((binary > 0).sum() - (sample.mask > 0).sum())
    sample.mask_source = display_path(mask_path)
    if not (sample.mask > 0).any():
        return "크롭 안에 결함 픽셀이 없음"
    sample.boxes = mask_boxes(sample.mask)
    return None


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

def sample_row(sample):
    height, width = sample.gray_a.shape[:2]
    return dict(
        sample_id=sample.sample_id, label=sample.label, target=int(sample.label == "NG"),
        view=sample.view, split=sample.split, source=display_path(sample.source), mask_source=sample.mask_source,
        height=height, width=width,
        **dict(zip(("left", "top", "right", "bottom"), sample.crop)),
        ng_pixels=int((sample.mask > 0).sum()), mask_outside=sample.mask_outside, box_count=len(sample.boxes),
    )


def summarize(args, samples, failures, renamed_masks):
    def size_range(values):
        return dict(min=int(np.min(values)), median=float(np.median(values)), max=int(np.max(values)))

    ng = [sample for sample in samples if sample.label == "NG"]
    boxes = [box for sample in ng for box in sample.boxes]
    return dict(
        created_at=datetime.now().isoformat(timespec="seconds"),
        crop_version=SURFACE_CROP_VERSION,
        crop_mode=CROP_MODE,
        tile_size=dict(
            height=size_range([sample.gray_a.shape[0] for sample in samples]),
            width=size_range([sample.gray_a.shape[1] for sample in samples]),
        ),
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
        box_size=dict(
            width=size_range([x2 - x1 for x1, _, x2, _, _ in boxes]),
            height=size_range([y2 - y1 for _, y1, _, y2, _ in boxes]),
        ),
    )

# ───────────────────────────────────────────────

HELP_EPILOG = """\
사용 예:
  python tools/make_od_data.py                   기본 설정으로 생성
  python tools/make_od_data.py --overwrite       기존 결과를 지우고 다시 생성

출력 구조:
  <output>/A/<split>/<OK|NG>/<sample_id>.png     A 그레이스케일 표면 타일 (app.py와 같은 크롭, 약 33×175)
  <output>/B/<split>/<OK|NG>/<sample_id>.png     B 그레이스케일 표면 타일
  <output>/mask/<split>/<OK|NG>/<sample_id>.png  결함 마스크 0/255 (OK는 전부 0)
  <output>/samples.csv        샘플별 split·크롭 좌표·타일 크기·결함 픽셀 수
  <output>/boxes.csv          마스크에서 만든 결함 박스 = 학습 정답 (타일 좌표, x2·y2는 미포함)
  <output>/failed.csv         실패 목록과 사유
  <output>/dataset_info.json  개수·설정 요약
"""


def main():
    parser = argparse.ArgumentParser(
        prog="python tools/make_od_data.py",
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
