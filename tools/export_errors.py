"""오분류 CSV(eval_classifier.py 출력)의 원본 tiff와 crop png를 dataset/errors/{CSV 이름}/{origin,crop}/{FN,FP}/ 에 모은다."""

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dataset" / "errors"


def resolve(path):
    """CSV 안 상대 경로는 저장소 기준으로 해석한다."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:  # utf-8-sig: BOM이 붙은 CSV도 첫 열 이름이 깨지지 않게
        return list(csv.DictReader(f))


def find_samples_csv(crop_path):
    """crop 이미지 위쪽 폴더를 거슬러 올라가며 make_data.py가 남긴 samples.csv를 찾는다."""
    for parent in crop_path.parents:
        candidate = parent / "samples.csv"
        if candidate.is_file():
            return candidate
        if parent == ROOT:
            break
    return None


def build_source_map(wrong_rows):
    """sample_id → 원본 tiff 경로. 데이터셋마다 samples.csv를 한 번만 읽는다."""
    sources, loaded = {}, set()
    for row in wrong_rows:
        samples_csv = find_samples_csv(resolve(row["path"]))
        if samples_csv is None or samples_csv in loaded:
            continue
        loaded.add(samples_csv)
        for sample in read_csv(samples_csv):
            sources.setdefault(sample["sample_id"], resolve(sample["source"]))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="오분류 CSV (예: models/side_surface_cnn_101310_wrong_test.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="상위 저장 폴더 (기본: dataset/errors)")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    wrong_rows = read_csv(csv_path)
    out_dir = args.out.resolve() / csv_path.stem
    sources = build_source_map(wrong_rows)

    copied, missing = {}, []
    for row in wrong_rows:
        files = {"origin": sources.get(row["sample_id"]), "crop": resolve(row["path"])}
        for kind, path in files.items():
            if path is None or not path.is_file():
                missing.append(f"{row['sample_id']} ({kind}): {path}")
                continue
            dest = out_dir / kind / row["error"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
            key = (kind, row["error"])
            copied[key] = copied.get(key, 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, out_dir / csv_path.name)

    print(f"[저장 위치] {out_dir}")
    for kind, error in sorted(copied):
        print(f"[{kind}/{error}] {copied[kind, error]}장")
    if missing:
        print(f"\n[찾지 못한 파일 {len(missing)}개]")
        for line in missing:
            print(f"  {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
