"""저장된 Side 표면 CNN을 OK/NG 폴더에 돌려 결과를 확인한다."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.classifier import SurfaceCNN, pick_device
from tools.train_classifier import DEFAULT_DATA, load_folder

VIEWS = ("Back", "Front", "Left", "Right")
WRONG_FIELDS = ("error", "sample_id", "label", "pred", "cnn_prob", "threshold", "view", "path")


def display_path(path):
    """저장소 안 경로는 상대 경로로 남겨 다른 PC에서도 CSV를 쓸 수 있게 한다."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def save_wrong_csv(path, rows, probs, pred, ng, threshold):
    """오분류 샘플을 FN(불량 놓침) → FP(오검출) 순, 각 그룹은 경계에서 먼 것부터 저장한다."""
    wrong = []
    for i in np.flatnonzero(pred != ng):
        row = rows[i]
        wrong.append({
            "error": "FN" if ng[i] else "FP",
            "sample_id": row["sample_id"],
            "label": row["label"],
            "pred": "NG" if pred[i] else "OK",
            "cnn_prob": f"{probs[i]:.4f}",
            "threshold": threshold,
            "view": row["view"],
            "path": display_path(row["path"]),
        })
    wrong.sort(key=lambda r: (r["error"] != "FN", -abs(float(r["cnn_prob"]) - threshold)))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글 경로가 깨지지 않게
        writer = csv.DictWriter(f, fieldnames=WRONG_FIELDS)
        writer.writeheader()
        writer.writerows(wrong)
    return len(wrong)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="평가할 .pt 파일") #
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA / "B" / "test", help="OK/, NG/ 하위 폴더가 포함된 평가 디렉토리") # 테스트를 위한 디렉토리
    parser.add_argument("--threshold", type=float, default=0.5) # NG 판정 기준으로 기본값은 0.5기준으로 판별
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output", type=Path, default=None,
                        help="오분류 CSV 경로 (기본: 모델 옆 {모델명}_wrong_{데이터 폴더명}.csv)")
    args = parser.parse_args()

    # 모델 불러오기
    model_path = args.model
    device = pick_device(args.device)
    model = SurfaceCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()
    data_dir = args.data.resolve()

    rows, x, y = load_folder(data_dir) 
    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
    targets = y.numpy()
    pred, ng = probs >= args.threshold, targets == 1

    tp, fn = int((pred & ng).sum()), int((~pred & ng).sum())
    fp, tn = int((pred & ~ng).sum()), int((~pred & ~ng).sum())
    print(f"\n[모델 이름] {model_path.name}  \n[데이터 위치] {data_dir}  \n[판정 기준(threshold)] {args.threshold}")
    print(f"\n[전체 {len(targets)}장] \n[정확도] {(tp + tn) / len(targets):.4f}  \n[TP] {tp}  \n[FN] {fn}  \n[FP] {fp}  \n[TN] {tn}")

    views = np.array([r["view"] for r in rows])
    for view in VIEWS:
        mask = views == view
        print(f"  {view:5s} {mask.sum()}장  오류 {int((pred[mask] != ng[mask]).sum())}")

    csv_path = args.output or model_path.parent / f"{model_path.stem}_wrong_{data_dir.name}.csv"
    count = save_wrong_csv(csv_path, rows, probs, pred, ng, args.threshold)
    print(f"\n[오분류] {count}장 → {display_path(csv_path)}")


if __name__ == "__main__":
    main()