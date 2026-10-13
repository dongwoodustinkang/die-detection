"""저장된 Side 표면 CNN을 OK/NG 폴더에 돌려 결과를 확인한다."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.classifier import SurfaceCNN, pick_device
from tools.train_classifier import DEFAULT_DATA, load_folder

VIEWS = ("Back", "Front", "Left", "Right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="평가할 .pt 파일") #
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA / "B" / "test", help="OK/, NG/ 하위 폴더가 포함된 평가 디렉토리") # 테스트를 위한 디렉토리
    parser.add_argument("--threshold", type=float, default=0.5) # NG 판정 기준으로 기본값은 0.5기준으로 판별
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
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

    wrong = [(rows[i]["sample_id"], round(float(probs[i]), 3)) for i in np.flatnonzero(pred != ng)]
    print(f"오분류 {len(wrong)}장:", wrong)


if __name__ == "__main__":
    main()