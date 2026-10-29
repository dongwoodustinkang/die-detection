"""저장된 표면 결함 분할 U-Net으로 데이터셋 split 또는 원본 TIFF 폴더를 평가한다."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from side.segmenter import SurfaceUNet, pick_device, preprocess
from tools.data.make_data import collect_tiffs, display_path, parse_label_and_view
from tools.data import make_od_data, make_sg_data
from tools.data.make_sg_data import MERGE_KERNEL, MIN_BOX_PIXELS, Sample
from tools.train.train_segmenter import DEFAULT_DATA, MIN_DEFECT_PIXELS, PIXEL_THRESHOLD, load_split

VIEWS = ("Back", "Front", "Left", "Right")
WRONG_FIELDS = ("error", "sample_id", "label", "pred", "defect_pixels", "max_prob", "view", "path")


def load_tiffs(input_dirs, views, crop="fixed"):
    """원본 TIFF를 make_sg_data.py(48×192 고정 크롭) 또는 make_od_data.py(표면 타일)와 같은 방식으로 잘라 (rows, x)를 만든다."""
    crop_sample = make_od_data.crop_sample if crop == "surface" else make_sg_data.crop_sample
    rows, inputs, failed = [], [], 0
    for path in collect_tiffs(input_dirs):
        label, view = parse_label_and_view(path)
        if label is None or (views and view not in views):
            continue
        sample = Sample(sample_id=path.stem, label=label, view=view, split="test", source=path)
        try:
            reason = crop_sample(sample)
        except Exception as error:
            reason = repr(error)
        if reason:
            failed += 1
            continue
        rows.append({"sample_id": path.stem, "label": label, "view": view, "path": path})
        inputs.append(preprocess(sample.gray_a, sample.gray_b, crop))
    if failed:
        print(f"[크롭 실패] {failed}장은 평가에서 제외")
    if not rows:
        raise SystemExit("평가할 TIFF가 없습니다 (라벨 OK/NG·방향 필터 확인).")
    return rows, torch.from_numpy(np.stack(inputs))


def predict(model, x, device, batch_size=256):
    """(N, 2, H, W) → (N, H, W) 결함 확률."""
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            probs.append(model(x[start: start + batch_size].to(device)).sigmoid().cpu())
    return torch.cat(probs).squeeze(1).numpy()


def regions(binary):
    """make_sg_data.py의 박스 규칙과 같게, 팽창으로 묶은 영역 중 결함 픽셀이 MIN_BOX_PIXELS 이상인 것만."""
    merged = cv2.dilate(binary.astype(np.uint8), np.ones((MERGE_KERNEL, MERGE_KERNEL), np.uint8))
    count, labels = cv2.connectedComponents(merged, connectivity=8)
    return [labels == i for i in range(1, count) if (binary & (labels == i)).sum() >= MIN_BOX_PIXELS]


def defect_scores(pred_masks, true_masks):
    """정답 결함 영역 중 예측과 겹친 비율(재현율), 예측 영역 중 정답과 겹친 비율(정밀도)."""
    true_hit = true_total = pred_hit = pred_total = 0
    for pred, true in zip(pred_masks, true_masks):
        for region in regions(true):
            true_total += 1
            true_hit += bool((pred & region).any())
        for region in regions(pred):
            pred_total += 1
            pred_hit += bool((true & region).any())
    return true_hit / max(true_total, 1), pred_hit / max(pred_total, 1), true_total, pred_total


def save_wrong_csv(path, rows, defect_pixels, max_probs, pred, ng):
    """오분류 샘플을 FN(불량 놓침) → FP(오검출) 순, 각 그룹은 판정 기준에서 먼 것부터 저장한다."""
    wrong = []
    for i in np.flatnonzero(pred != ng):
        row = rows[i]
        wrong.append({
            "error": "FN" if ng[i] else "FP",
            "sample_id": row["sample_id"],
            "label": row["label"],
            "pred": "NG" if pred[i] else "OK",
            "defect_pixels": int(defect_pixels[i]),
            "max_prob": f"{max_probs[i]:.4f}",
            "view": row["view"],
            "path": display_path(row["path"]),
        })
    wrong.sort(key=lambda r: (r["error"] != "FN", -abs(r["defect_pixels"] - MIN_DEFECT_PIXELS)))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글 경로가 깨지지 않게
        writer = csv.DictWriter(f, fieldnames=WRONG_FIELDS)
        writer.writeheader()
        writer.writerows(wrong)
    return len(wrong)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="평가할 .pt 파일")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="side_surface_od_v1 형식 데이터셋 폴더")
    parser.add_argument("--split", default="test", help="--data에서 평가할 split (train | val | test)")
    parser.add_argument("--tiff", type=Path, nargs="+", default=None,
                        help="원본 TIFF 폴더 (지정하면 --data 대신 즉석 크롭해서 이미지 단위로만 평가)")
    parser.add_argument("--views", nargs="+", default=None, help="--tiff에서 평가할 방향만 (예: Front Right)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output", type=Path, default=None,
                        help="오분류 CSV 경로 (기본: 모델 옆 {모델명}_wrong_{평가 이름}.csv)")
    args = parser.parse_args()

    # 모델 불러오기
    model_path = args.model
    config_path = model_path.with_suffix(".json")
    crop = json.loads(config_path.read_text(encoding="utf-8"))["crop"] if config_path.exists() else "fixed"
    device = pick_device(args.device)
    model = SurfaceUNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    if args.tiff:
        rows, x = load_tiffs(args.tiff, args.views, crop)
        true_masks = None
        # 폴더 이름이 OK/NG뿐이면 상위 폴더 이름을 붙인다 (예: front_261006-NG)
        name = "_".join(
            f"{path.resolve().parent.name}-{path.name}" if path.name.upper() in ("OK", "NG") else path.resolve().name
            for path in args.tiff
        ) + ("_" + "".join(args.views) if args.views else "")
        source = ", ".join(display_path(path) for path in args.tiff)
    else:
        rows, x, y = load_split(args.data, args.split, crop)
        true_masks = y.squeeze(1).numpy() > 0
        name, source = args.split, display_path(args.data / args.split)

    probs = predict(model, x, device)
    pred_masks = probs >= PIXEL_THRESHOLD
    defect_pixels = pred_masks.sum(axis=(1, 2))
    max_probs = probs.max(axis=(1, 2))
    pred = defect_pixels >= MIN_DEFECT_PIXELS
    ng = np.array([r["label"] == "NG" for r in rows])

    tp, fn = int((pred & ng).sum()), int((~pred & ng).sum())
    fp, tn = int((pred & ~ng).sum()), int((~pred & ~ng).sum())
    print(f"\n[모델 이름] {model_path.name}  \n[데이터 위치] {source}  \n[판정 기준] 픽셀 확률 ≥ {PIXEL_THRESHOLD}, 결함 픽셀 ≥ {MIN_DEFECT_PIXELS}개 → NG")
    print(f"\n[전체 {len(rows)}장] \n[정확도] {(tp + tn) / len(rows):.4f}  \n[TP] {tp}  \n[FN] {fn}  \n[FP] {fp}  \n[TN] {tn}")

    views = np.array([r["view"] for r in rows])
    for view in VIEWS:
        mask = views == view
        if mask.any():
            print(f"  {view:5s} {mask.sum()}장  오류 {int((pred[mask] != ng[mask]).sum())}"
                  f"  (FN {int((~pred[mask] & ng[mask]).sum())} · FP {int((pred[mask] & ~ng[mask]).sum())})")

    if true_masks is not None: # 마스크가 있을 때만 픽셀·결함 단위 지표
        inter = (pred_masks & true_masks).sum()
        dice = 2 * inter / max(pred_masks.sum() + true_masks.sum(), 1)
        recall, precision, true_total, pred_total = defect_scores(pred_masks, true_masks)
        print(f"\n[픽셀 Dice] {dice:.4f}")
        print(f"[결함 재현율] {recall:.4f} (정답 결함 {true_total}개)  \n[결함 정밀도] {precision:.4f} (예측 영역 {pred_total}개)")

    csv_path = args.output or model_path.parent / f"{model_path.stem}_wrong_{name}.csv"
    count = save_wrong_csv(csv_path, rows, defect_pixels, max_probs, pred, ng)
    print(f"\n[오분류] {count}장 → {display_path(csv_path)}")


if __name__ == "__main__":
    main()