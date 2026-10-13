"""저장된 Side 표면 CNN을 OK/NG 폴더에 돌려 결과를 확인한다."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.classifier import SurfaceCNN, pick_device
from tools.train_classifier import DEFAULT_DATA, MODEL_DIR, load_folder

VIEWS = ("Back", "Front", "Left", "Right")


def latest_model():
    """models/에서 가장 최근에 저장된 .pt (파일명이 아닌 수정 시각 기준)."""
    paths = list(MODEL_DIR.glob("side_surface_cnn_*.pt"))
    if not paths:
        raise SystemExit(f"모델이 없습니다: {MODEL_DIR}")
    return max(paths, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="평가할 .pt (기본: 가장 최근 모델)") #
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA / "B" / "test", help="OK/, NG/ 하위 폴더가 포함된 평가 디렉토리") # 테스트를 위한 디렉토리
    parser.add_argument("--threshold", type=float, default=0.5) # NG 판정 기준으로 기본값은 0.5기준으로 판별
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    # 모델 불러오기
    model_path = args.model or latest_model()
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
    print(f"모델 {model_path.name}  데이터 {data_dir}  threshold {args.threshold}")
    print(f"전체 {len(targets)}장  정확도 {(tp + tn) / len(targets):.4f}  TP {tp}  FN {fn}  FP {fp}  TN {tn}")

    views = np.array([r["view"] for r in rows])
    for view in VIEWS:
        mask = views == view
        print(f"  {view:5s} {mask.sum()}장  오류 {int((pred[mask] != ng[mask]).sum())}")

    for label, is_ng in (("OK", False), ("NG", True)):
        p = probs[ng == is_ng]
        print(f"{label} 확률: 최소 {p.min():.4f}  중앙 {np.median(p):.4f}  최대 {p.max():.4f}")

    order = np.argsort(probs)
    print("NG 중 확률 낮은 5장:", [(rows[i]["sample_id"], round(float(probs[i]), 3)) for i in order if ng[i]][:5])
    print("OK 중 확률 높은 5장:", [(rows[i]["sample_id"], round(float(probs[i]), 3)) for i in order[::-1] if not ng[i]][:5])

    wrong = [(rows[i]["sample_id"], round(float(probs[i]), 3)) for i in np.flatnonzero(pred != ng)]
    print(f"오분류 {len(wrong)}장:", wrong)


if __name__ == "__main__":
    main()