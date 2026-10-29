"""지금까지 학습한 표면 모델들의 평가 결과(오분류 CSV)를 모아 모델 × 평가셋 정확도 CSV를 만든다.

볼 검사(run_side_detection)에서 이미 NG인 샘플을 뺀 결과도 같은 줄에 붙인다.
(eval_detection.py의 {FN,FP}_notbal 과 같은 기준: 슬롯 하나라도 '검출'이 아니면 볼 NG)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.eval.eval_detection import ball_is_ng
from tools.data.make_data import collect_tiffs, display_path, parse_label_and_view, write_csv

DEFAULT_OUTPUT = ROOT / "models" / "eval_summary.csv"
BALL_CACHE = ROOT / "models" / "ball_status.csv"  # 볼 검사는 모델과 무관하므로 원본 TIFF별로 한 번만 계산
ORIGIN_1006 = ROOT / "dataset" / "원본" / "Side_Demo_1006"

# (그룹, 표시 이름, 가중치 파일). .json이 없는 예전 모델은 NO_CONFIG에서 설정을 읽는다.
MODELS = (
    ("Classifier", "CNN A", "models/Classifier/side_surface_cnnA_101910.pt"),
    ("Classifier", "CNN B", "models/Classifier/side_surface_cnnB_101910.pt"),
    ("Classifier", "CNN A", "models/Classifier/side_surface_cnnA_102211.pt"),
    ("Classifier", "CNN B", "models/Classifier/side_surface_cnnB_102211.pt"),
    ("Classifier", "CNN A+B", "models/Classifier/side_surface_cnnAB_102211.pt"),
    ("Segmenter", "U-Net A+B (48×192)", "models/Segmenter/side_surface_unet_101911.pt"),
    ("Segmenter", "U-Net A", "models/Segmenter/side_surface_unetA_102009.pt"),
    ("Segmenter", "U-Net B", "models/Segmenter/side_surface_unetB_102009.pt"),
    ("Segmenter", "U-Net A+B", "models/Segmenter/side_surface_unetAB_102211.pt"),
    ("Detector", "Detector A (이전 버전)", "models/Detector/side_surface_detA_102011.pt"),
    ("Detector", "Detector B (이전 버전)", "models/Detector/side_surface_detB_102011.pt"),
    ("Detector", "Faster R-CNN A", "models/Detector/side_surface_frcnnA_102113.pt"),
    ("Detector", "Faster R-CNN B", "models/Detector/side_surface_frcnnB_102114.pt"),
    ("Detector", "Faster R-CNN A+B", "models/Detector/side_surface_frcnnAB_*.pt"),
)
NO_CONFIG = {
    "side_surface_cnnA_101910": dict(kind="A", crop="surface", data="dataset/infer/side_surface_v1"),
    "side_surface_cnnB_101910": dict(kind="B", crop="surface", data="dataset/infer/side_surface_v1"),
    "side_surface_unet_101911": dict(kind="AB", crop="fixed", data="dataset/infer/side_surface_od_v1"),
}

# (평가셋 이름, 오분류 CSV 접미사 후보, TIFF 폴더). TIFF 폴더가 None이면 학습 데이터셋의 test split.
EVAL_SETS = (
    ("Test_split", ("test",), None),
    ("Test_1013", ("Test_1013", "real_v1"), [ROOT / "dataset" / "Test_1013"]),
    ("1006_Back_Left", ("back_261006_left_261006",), [ORIGIN_1006 / "back_261006", ORIGIN_1006 / "left_261006"]),
    ("1006_Front_Right_NG", ("front_261006-NG_right_261006-NG",),
     [ORIGIN_1006 / "front_261006" / "NG", ORIGIN_1006 / "right_261006" / "NG"]),
)

COUNT_FIELDS = ("total", "OK", "NG", "TP", "FN", "FP", "TN", "accuracy")
FIELDS = (
    "group", "model", "file", "kind", "crop", "train_data", "eval_set", *COUNT_FIELDS,
    "ball_ng_excluded", *(f"{field}_notbal" for field in COUNT_FIELDS),
)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:  # utf-8-sig: BOM이 붙은 CSV도 첫 열 이름이 깨지지 않게
        return list(csv.DictReader(file))


def model_config(path):
    config_path = path.with_suffix(".json")
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return NO_CONFIG[path.stem]


def eval_samples(tiff_dirs, train_data):
    """평가셋 전체 샘플 {sample_id: (label, view, 원본 TIFF)}."""
    if tiff_dirs is None:
        rows = read_csv(ROOT / train_data / "samples.csv")
        return {row["sample_id"]: (row["label"], row["view"], ROOT / row["source"]) for row in rows if row["split"] == "test"}
    samples = {}
    for path in collect_tiffs(tiff_dirs):
        label, view = parse_label_and_view(path)
        if label is not None:
            if path.stem in samples:
                raise SystemExit(f"평가셋 안에 같은 sample_id가 있습니다: {path.stem}")
            samples[path.stem] = (label, view, path)
    return samples


def load_ball_cache():
    if not BALL_CACHE.exists():
        return {}
    return {row["source"]: row["ball_ng"] == "1" for row in read_csv(BALL_CACHE)}


def ball_status(samples, cache):
    """sample_id → 볼 NG 여부. 캐시에 없는 원본만 볼 검사를 돌린다."""
    status = {}
    for sample_id, (_, view, source) in samples.items():
        key = display_path(source)
        if key not in cache:
            cache[key] = ball_is_ng(source, sample_id, view)
        status[sample_id] = cache[key]
    return status


def counts(samples, wrong, keep):
    """keep에 든 샘플만으로 TP/FN/FP/TN과 정확도를 센다."""
    ng = {sid for sid in keep if samples[sid][0] == "NG"}
    ok = keep - ng
    fn, fp = len(ng & wrong), len(ok & wrong)
    total = len(keep)
    return dict(total=total, OK=len(ok), NG=len(ng), TP=len(ng) - fn, FN=fn, FP=fp, TN=len(ok) - fp,
                accuracy=f"{(total - fn - fp) / total:.4f}" if total else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"출력 CSV (기본: {display_path(DEFAULT_OUTPUT)})")
    args = parser.parse_args()

    cache = load_ball_cache()
    rows, skipped = [], []
    for group, name, pattern in MODELS:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            skipped.append(f"{pattern} (가중치 없음)")
            continue
        for model_path in matches:
            config = model_config(model_path)
            for eval_name, suffixes, tiff_dirs in EVAL_SETS:
                wrong_csv = next((path for suffix in suffixes
                                  if (path := model_path.parent / f"{model_path.stem}_wrong_{suffix}.csv").exists()), None)
                if wrong_csv is None:
                    skipped.append(f"{model_path.stem} · {eval_name} (오분류 CSV 없음)")
                    continue
                samples = eval_samples(tiff_dirs, config["data"])
                wrong = {row["sample_id"] for row in read_csv(wrong_csv)}
                if wrong - samples.keys():
                    raise SystemExit(f"{wrong_csv.name}의 샘플이 평가셋에 없습니다: {sorted(wrong - samples.keys())[:3]}")
                ball = ball_status(samples, cache)
                notbal = {sid for sid in samples if not ball[sid]}
                rows.append(dict(
                    group=group, model=name, file=model_path.stem, kind=config["kind"], crop=config.get("crop", "surface"),
                    train_data=Path(config["data"]).name, eval_set=eval_name,
                    **counts(samples, wrong, set(samples)),
                    ball_ng_excluded=len(samples) - len(notbal),
                    **{f"{key}_notbal": value for key, value in counts(samples, wrong, notbal).items()},
                ))
            print(f"  {model_path.stem}")

    write_csv(BALL_CACHE, ("source", "ball_ng"), [dict(source=source, ball_ng=int(ng)) for source, ng in sorted(cache.items())])
    write_csv(args.output, FIELDS, rows)
    print(f"저장: {display_path(args.output)} ({len(rows)}줄)  볼 검사 캐시: {display_path(BALL_CACHE)}")
    for item in skipped:
        print(f"  건너뜀: {item}")


if __name__ == "__main__":
    main()
