"""Side A/B TIFF에서 표면 Crop 타일을 뽑아 train/val/test 학습 데이터셋을 만든다."""

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from general import load_ab_tiff_pages # 이미지 불러오기
from side import SURFACE_CROP_VERSION, extract_surface_tiles # 이미지 자르기 


@dataclass
class Sample:
    """크롭된 샘플을 csv에 담을 양식"""
    sample_id: str
    label: str
    view: str
    source: Path
    gray_a: np.ndarray
    gray_b: np.ndarray
    bounds: tuple  # (left, top, right, bottom)
    split: str = ""

# 입출력 이미지 폴더 경로
DEFAULT_INPUT = ROOT / "dataset" / "원본" / "Side_Demo_1006"
DEFAULT_OUTPUT = ROOT / "dataset" / "side_surface"

IMAGE_EXTS = {".tif", ".tiff"} # 이미지 파일 확장자
LABELS = ("OK", "NG")
SPLITS = ("train", "val", "test")
INPUT_KINDS = ("A", "B")

# 파일명 앞의 패턴
FILENAME_PATTERN = re.compile(r"^(OK|NG)_([A-Za-z]+)_", re.IGNORECASE)

# CSV 파일에 저장될 컬럼
SAMPLE_FIELDS = (
    "sample_id", "label", "target", "view", "split", "source",
    "height", "width", "left", "top", "right", "bottom",
)
FAILED_FIELDS = ("source", "label", "reason")

# ───────────────────────────────────────────────

def display_path(path):
    """저장소 안 경로는 상대 경로로 남겨 다른 PC에서도 CSV를 쓸 수 있게 한다."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


# ───────────────────────────────────────────────

def collect_tiffs(input_dirs):
    files = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise SystemExit(f"해당 디렉토리는 존재 하지 않습니다. : {input_dir}")
        files.extend(
            path for path in input_dir.rglob("*") # 하위 디렉토리 재귀 탐색
            if path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".") # 이미지 파일만 추출
        )
    if not files:
        raise SystemExit("TIFF나 TIF 파일을 찾지 못했습니다.")
    return sorted(files)

def parse_label_and_view(path):
    """상위 폴더(OK/NG) 또는 파일명 접두어에서 라벨과 촬영 방향을 읽는다."""
    match = FILENAME_PATTERN.match(path.name)
    view = match.group(2).capitalize() if match else ""
    for part in reversed(path.parts[:-1]):
        if part.upper() in LABELS:
            return part.upper(), view
    return (match.group(1).upper() if match else None), view

def crop_images(files):
    samples, failures = [], [] # 정상과 실패의 이미지 데이터셋를 담는 리스트
    used_ids = Counter() # 중복 여부, 사용 빈도 체크
    for index, path in enumerate(files, 1):
        label, view = parse_label_and_view(path)
        reason = None
        if label is None:
            reason = "OK/NG 라벨을 알 수 없습니다."
        else:
            try:
                tiles = extract_surface_tiles(*load_ab_tiff_pages(path))
                if tiles is None:
                    reason = "A 컨투어 없음"
            except Exception as error:
                reason = repr(error)

        if reason:
            failures.append(dict(source=display_path(path), label=label or "", reason=reason))
        else:
            # 다른 폴더에 같은 파일명이 있으면 번호를 붙여 덮어쓰지 않는다. 
            used_ids[path.stem] += 1
            count = used_ids[path.stem]
            samples.append(Sample(
                sample_id=path.stem if count == 1 else f"{path.stem}__{count}",
                label=label, view=view, source=path,
                gray_a=tiles.gray_a, gray_b=tiles.gray_b,
                bounds=tiles.bounds,
            ))
        if index % 500 == 0 or index == len(files):
            print(f"  Crop {index}/{len(files)}  : 성공 {len(samples)} · 실패 {len(failures)}")
    return samples, failures
# ───────────────────────────────────────────────

def share_by_size(total, group_sizes):
    """total개를 그룹 크기 비율대로 정수 배분한다(나머지는 소수점이 큰 그룹부터 1개씩)."""
    whole = sum(group_sizes.values())
    exact = {key: total * size / whole for key, size in group_sizes.items()}
    shares = {key: int(value) for key, value in exact.items()}
    leftover = total - sum(shares.values())
    for key in sorted(exact, key=lambda k: exact[k] - shares[k], reverse=True)[:leftover]:
        shares[key] += 1
    return shares


def check_ratios(ratios):
    """비율은 0 이상이고 합이 0보다 크고 1 이하여야 한다. 합이 1보다 작으면 남는 샘플은 쓰지 않는다."""
    total = sum(ratios.values())
    if any(ratio < 0 for ratio in ratios.values()) or not 0 < total <= 1 + 1e-9:
        given = " ".join(f"--{split} {ratio}" for split, ratio in ratios.items())
        raise SystemExit(
            f"비율이 잘못되었습니다: {given} (합 {total:g})\n"
            "각 비율은 0 이상, 합은 0보다 크고 1 이하여야 합니다. "
            "예: --train 0.7 --val 0.15 --test 0.15 / 테스트만: --train 0 --val 0 --test 1.0"
        )


def ratios_to_counts(available, ratios):
    """라벨 하나의 샘플 수를 비율대로 train/val/test 개수로 바꾼다. 비율 합이 1보다 작으면 나머지는 미사용."""
    weights = dict(ratios, unused=max(0.0, 1 - sum(ratios.values())))
    counts = share_by_size(available, weights)
    return {split: counts[split] for split in SPLITS}


def assign_splits(samples, ratios, seed):
    """라벨마다 비율만큼 훈련, 검증, 테스트 세트를 배정하고 라벨별 개수를 돌려준다."""
    rng = np.random.default_rng(seed) # seed 고정
    counts_by_label = {}
    for label in LABELS: # 각 OK/NG 라벨에 대해
        by_view = defaultdict(list) #촬영 방향별로 딕셔너리 묶기
        for sample in samples:
            if sample.label == label:
                by_view[sample.view].append(sample)

        available = sum(len(group) for group in by_view.values()) # 현재 라벨의 총 개수
        if available == 0:
            continue
        counts = ratios_to_counts(available, ratios) # 비율 → 개수 (예: 4000 × 0.7 = 2800)
        counts_by_label[label] = counts

        for view in sorted(by_view): # 촬영 방향별로
            rng.shuffle(by_view[view]) # 랜덤 섞기
        sizes = {view: len(group) for view, group in by_view.items()} # 촬영 방향별 개수
        start = dict.fromkeys(by_view, 0) # 시작점 초기화
        for split in SPLITS: # train, val, test
            for view, take in share_by_size(counts[split], sizes).items(): # 비율에 따라 나누기
                for sample in by_view[view][start[view]: start[view] + take]:
                    sample.split = split
                start[view] += take
    return counts_by_label


# ───────────────────────────────────────────────

def write_png(path, image):
    """한글 경로에서도 안전하게 무손실 PNG로 저장한다."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"PNG 인코딩 실패: {path.name}")
    path.write_bytes(encoded.tobytes())


def check_output(out_dir, overwrite):
    """이전 결과와 섞이지 않도록, 출력 폴더가 비어 있지 않으면 --overwrite 없이는 중단한다."""
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"출력 폴더가 비어 있지 않습니다: {out_dir}\n"
            "다시 만들려면 --overwrite를 붙이세요."
        )


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
        if index % 1000 == 0 or index == len(samples):
            print(f"  저장 {index}/{len(samples)}")


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_row(sample):
    left, top, right, bottom = sample.bounds
    height, width = sample.gray_a.shape[:2]
    return dict(
        sample_id=sample.sample_id, label=sample.label, target=int(sample.label == "NG"),
        view=sample.view, split=sample.split, source=display_path(sample.source),
        height=height, width=width,
        left=left, top=top, right=right, bottom=bottom,
    )


def summarize(args, files, cropped, used, failures, counts_by_label):
    def size_range(values):
        return dict(min=int(np.min(values)), median=float(np.median(values)), max=int(np.max(values)))

    by_split = Counter(sample.split for sample in used)
    return dict(
        created_at=datetime.now().isoformat(timespec="seconds"),
        crop_version=SURFACE_CROP_VERSION,
        inputs=[display_path(path) for path in args.inputs],
        seed=args.seed,
        tiff_total=len(files),
        crop_ok=len(cropped),
        crop_failed=len(failures),
        failed_by_label=dict(Counter(row["label"] for row in failures)),
        ratios=dict(train=args.train, val=args.val, test=args.test),
        counts_by_label=counts_by_label,
        used=len(used),
        unused=len(cropped) - len(used),
        by_split=dict(by_split),
        percent_of_cropped={split: round(100 * by_split[split] / len(cropped), 2) for split in SPLITS} if cropped else {},
        by_split_view_label=dict(sorted(Counter(
            f"{sample.split}/{sample.view}/{sample.label}" for sample in used
        ).items())),
        tile_size=dict(
            height=size_range([sample.gray_a.shape[0] for sample in cropped]),
            width=size_range([sample.gray_a.shape[1] for sample in cropped]),
        ),
    )


# ───────────────────────────────────────────────




HELP_EPILOG = """\
사용 예:
  python tools/data/make_data.py                      기본 설정으로 생성
  python tools/data/make_data.py --overwrite          기존 결과를 지우고 다시 생성
  python tools/data/make_data.py dataset/원본/A dataset/원본/B --output dataset/side_surface_v2
  python tools/data/make_data.py --train 0.6 --val 0.2 --test 0.2 --seed 7 --overwrite
  python tools/data/make_data.py dataset/원본/새데이터 --output dataset/새데이터_test --train 0 --val 0 --test 1.0
                                                 전부 test로 (테스트만 할 경우)

출력 구조:
  <output>/A/<train|val|test>/<OK|NG>/<sample_id>.png    A 그레이스케일 타일
  <output>/B/<train|val|test>/<OK|NG>/<sample_id>.png    B 그레이스케일 타일
  <output>/samples.csv        샘플별 split·label·view·크기
  <output>/failed.csv         Crop 실패 목록과 사유
  <output>/dataset_info.json  개수·비율·seed 요약
"""


def main():
    parser = argparse.ArgumentParser(
        prog="python tools/data/make_data.py",
        description=__doc__.splitlines()[0],
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="이 도움말을 보여주고 종료")
    parser.add_argument(
        "inputs", nargs="*", type=Path, default=[DEFAULT_INPUT], metavar="INPUT_DIR",
        help=f"A/B TIFF가 들어 있는 폴더, 여러 개 가능·하위 폴더 포함 (기본: {display_path(DEFAULT_INPUT)})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, metavar="DIR",
                        help=f"출력 폴더 (기본: {display_path(DEFAULT_OUTPUT)})")
    parser.add_argument("--train", type=float, default=0.7, metavar="R", help="OK/NG 각각에서 train 비율 (기본: 0.7)")
    parser.add_argument("--val", type=float, default=0.15, metavar="R", help="OK/NG 각각에서 val 비율 (기본: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, metavar="R", help="OK/NG 각각에서 test 비율 (기본: 0.15)")
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="분할 셔플 seed, 같으면 분할도 같음 (기본: 42)")
    parser.add_argument("--overwrite", action="store_true", help="출력 폴더가 있으면 지우고 다시 만든다")
    args = parser.parse_args()

    ratios = dict(train=args.train, val=args.val, test=args.test)
    check_ratios(ratios) # Crop 전에 비율 확인
    check_output(args.output, args.overwrite) # OVERWRITE 체크

    files = collect_tiffs(args.inputs) # 이미지 파일 탐색
    print(f"⬜⬛️⬛️ : 이미지 파일 개수 {len(files)}개 확보")
    cropped, failures = crop_images(files) # 모든 이미지 파일 표면 크롭 후 성공 및 실패 결과를 도출

    print(f"⬜⬜⬛️ : 분할")
    counts_by_label = assign_splits(cropped, ratios, args.seed)
    for label, counts in counts_by_label.items():
        print(f"  {label}: " + " · ".join(f"{split} {count}" for split, count in counts.items()))
    used = [sample for sample in cropped if sample.split]

    print(f"⬜⬜⬜ : 저장 → {args.output}")
    reset_output(args.output)
    save_images(used, args.output)
    write_csv(args.output / "samples.csv", SAMPLE_FIELDS, [sample_row(sample) for sample in used])
    write_csv(args.output / "failed.csv", FAILED_FIELDS, failures)
    info = summarize(args, files, cropped, used, failures, counts_by_label)
    (args.output / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
