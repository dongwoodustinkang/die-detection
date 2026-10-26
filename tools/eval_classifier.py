"""저장된 Side 표면 CNN을 OK/NG 폴더에 돌려 결과를 확인한다."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.classifier import SurfaceCNN, pick_device, preprocess
from tools.make_data import collect_tiffs, parse_label_and_view
from tools.make_od_data import crop_sample
from tools.make_sg_data import Sample
from tools.train_classifier import DEFAULT_DATA, load_folder, load_split

VIEWS = ("Back", "Front", "Left", "Right")
WRONG_FIELDS = ("error", "sample_id", "label", "pred", "cnn_prob", "threshold", "view", "path")


def display_path(path):
    """저장소 안 경로는 상대 경로로 남겨 다른 PC에서도 CSV를 쓸 수 있게 한다."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def model_kind(model_path):
    """모델 옆 .json의 kind. 없으면 파일명(side_surface_cnnA_...)에서, 그래도 없으면 B."""
    config_path = model_path.with_suffix(".json")
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))["kind"]
    for kind in ("AB", "A", "B"):
        if f"cnn{kind}" in model_path.stem:
            return kind
    return "B"


def load_tiffs(input_dirs, views, kind):
    """원본 TIFF를 make_od_data.py와 같은 표면 타일로 잘라 (rows, x, y)를 만든다."""
    rows, tiles, failed = [], [], 0
    for path in collect_tiffs(input_dirs):
        label, view = parse_label_and_view(path)
        if label is None or (views and view.capitalize() not in views):
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
        images = {"A": sample.gray_a, "B": sample.gray_b}
        tiles.append(np.concatenate([preprocess(images[k]) for k in kind]))  # AB는 A·B를 채널로 겹친다
    if failed:
        print(f"[크롭 실패] {failed}장은 평가에서 제외")
    if not rows:
        raise SystemExit("평가할 TIFF가 없습니다 (라벨 OK/NG·방향 필터 확인).")
    y = torch.tensor([float(r["label"] == "NG") for r in rows])
    return rows, torch.from_numpy(np.stack(tiles)), y


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
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA / "B" / "test",
                        help="OK/, NG/ 하위 폴더가 포함된 평가 디렉토리 (AB 모델은 {데이터셋}/A/{split}을 주면 옆 B도 읽음)") # 테스트를 위한 디렉토리
    parser.add_argument("--tiff", type=Path, nargs="+", default=None,
                        help="원본 TIFF 폴더 (지정하면 --data 대신 즉석 크롭해서 평가)")
    parser.add_argument("--views", nargs="+", default=None, help="--tiff에서 평가할 방향만 (예: Front Right)")
    parser.add_argument("--threshold", type=float, default=0.5) # NG 판정 기준으로 기본값은 0.5기준으로 판별
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output", type=Path, default=None,
                        help="오분류 CSV 경로 (기본: 모델 옆 {모델명}_wrong_{데이터 폴더명}.csv)")
    args = parser.parse_args()

    # 모델 불러오기
    model_path = args.model
    device = pick_device(args.device)
    kind = model_kind(model_path)
    model = SurfaceCNN(in_ch=len(kind))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    if args.tiff:
        rows, x, y = load_tiffs(args.tiff, args.views, kind)
        # 폴더 이름이 OK/NG뿐이면 상위 폴더 이름을 붙인다 (예: front_261006-NG)
        name = "_".join(
            f"{path.resolve().parent.name}-{path.name}" if path.name.upper() in ("OK", "NG") else path.resolve().name
            for path in args.tiff
        ) + ("_" + "".join(args.views) if args.views else "")
        source = f"{', '.join(display_path(path) for path in args.tiff)} ({kind} 타일)"
    else:
        data_dir = args.data.resolve()
        rows, x, y = load_split(data_dir.parents[1], data_dir.name, "AB") if kind == "AB" else load_folder(data_dir)
        name, source = data_dir.name, data_dir
    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
    targets = y.numpy()
    pred, ng = probs >= args.threshold, targets == 1

    tp, fn = int((pred & ng).sum()), int((~pred & ng).sum())
    fp, tn = int((pred & ~ng).sum()), int((~pred & ~ng).sum())
    print(f"\n[모델 이름] {model_path.name}  \n[데이터 위치] {source}  \n[판정 기준(threshold)] {args.threshold}")
    print(f"\n[전체 {len(targets)}장] \n[정확도] {(tp + tn) / len(targets):.4f}  \n[TP] {tp}  \n[FN] {fn}  \n[FP] {fp}  \n[TN] {tn}")

    views = np.array([r["view"].capitalize() for r in rows])
    for view in VIEWS:
        mask = views == view
        if mask.any():
            print(f"  {view:5s} {mask.sum()}장  오류 {int((pred[mask] != ng[mask]).sum())}"
                  f"  (FN {int((~pred[mask] & ng[mask]).sum())} · FP {int((pred[mask] & ~ng[mask]).sum())})")

    csv_path = args.output or model_path.parent / f"{model_path.stem}_wrong_{name}.csv"
    count = save_wrong_csv(csv_path, rows, probs, pred, ng, args.threshold)
    print(f"\n[오분류] {count}장 → {display_path(csv_path)}")


if __name__ == "__main__":
    main()