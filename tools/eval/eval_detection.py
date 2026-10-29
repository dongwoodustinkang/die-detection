"""저장된 표면 결함 Object Detection(Faster R-CNN)으로 데이터셋 split 또는 원본 TIFF 폴더를 평가한다."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from side import run_side_detection
from side.ball import BALL_STATUS_OK
from side.detection import SCORE_THRESHOLD, load_model, predict, preprocess, scale_boxes
from tools.data.make_data import collect_tiffs, display_path, parse_label_and_view, write_png
from tools.data.make_od_data import crop_sample
from tools.data.make_sg_data import Sample
from tools.train.train_detection import DEFAULT_DATA, load_split

VIEWS = ("Back", "Front", "Left", "Right")
MATCH_IOU = 0.3 # 박스 재현율·정밀도에서 정답과 맞았다고 보는 IoU (가는 사선은 IoU가 낮게 나와 0.5보다 느슨하게)
WRONG_FIELDS = ("error", "sample_id", "label", "pred", "max_score", "boxes", "view", "path", "source")
DEFAULT_ERRORS = ROOT / "dataset" / "errors" / "Detector"
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def load_tiffs(input_dirs, views, kind):
    """원본 TIFF를 make_od_data.py와 같은 표면 타일로 잘라 (rows, x)를 만든다."""
    rows, inputs, failed = [], [], 0
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
        tiles = {"A": sample.gray_a, "B": sample.gray_b}
        parts = [preprocess(tiles[k]) for k in kind]  # AB는 A·B를 채널로 겹친다
        x, size = np.concatenate([part[0] for part in parts]), parts[0][1]
        rows.append({"sample_id": path.stem, "label": label, "view": view, "path": path, "source": path, "size": size,
                     "gray_a": sample.gray_a, "gray_b": sample.gray_b})
        inputs.append(x)
    if failed:
        print(f"[크롭 실패] {failed}장은 평가에서 제외")
    if not rows:
        raise SystemExit("평가할 TIFF가 없습니다 (라벨 OK/NG·방향 필터 확인).")
    return rows, torch.from_numpy(np.stack(inputs))


def attach_split_images(rows, data_dir):
    """split 평가에서 오류 이미지용 A/B 타일과 원본 TIFF 경로를 붙인다."""
    with open(data_dir / "samples.csv", newline="", encoding="utf-8-sig") as file:
        sources = {row["sample_id"]: ROOT / row["source"] for row in csv.DictReader(file)}
    for row in rows:
        relative = row["path"].relative_to(row["path"].parents[2])  # {kind}/{split}/{label}/x.png → {split}/{label}/x.png
        row["gray_a"] = cv2.imread(str(data_dir / "A" / relative), cv2.IMREAD_GRAYSCALE)
        row["gray_b"] = cv2.imread(str(data_dir / "B" / relative), cv2.IMREAD_GRAYSCALE)
        row["source"] = sources.get(row["sample_id"])

# 지표 ───────────────────────────────────────────────

def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda box: (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def box_scores(outputs, targets, iou=MATCH_IOU):
    """점수 ≥ SCORE_THRESHOLD 박스 기준 재현율·정밀도와, 전체 점수로 계산한 AP(VOC 전 구간)."""
    true_hit = pred_hit = pred_total = 0
    true_total = sum(len(t) for t in targets)
    ranked = []  # (score, 정답과 맞음 여부) — 정답 하나에는 가장 높은 점수 박스 하나만 맞음
    for out, true in zip(outputs, targets):
        boxes, scores = out["boxes"], out["scores"]
        ious = iou_matrix(boxes, true)
        keep = scores >= SCORE_THRESHOLD
        pred_total += int(keep.sum())
        pred_hit += int((ious[keep] >= iou).any(axis=1).sum()) if len(true) else 0
        true_hit += int((ious[keep] >= iou).any(axis=0).sum()) if keep.any() and len(true) else 0
        used = set()
        for i in np.argsort(-scores):
            j = int(np.argmax(ious[i])) if len(true) else -1
            hit = j >= 0 and ious[i, j] >= iou and j not in used
            if hit:
                used.add(j)
            ranked.append((scores[i], hit))
    ranked.sort(key=lambda item: -item[0])
    hits = np.array([hit for _, hit in ranked], bool)
    tp, fp = np.cumsum(hits), np.cumsum(~hits)
    recall = tp / max(true_total, 1)
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    ap = float(np.sum(np.diff(np.concatenate([[0], recall])) * envelope)) if len(recall) else 0.0
    return true_hit / max(true_total, 1), pred_hit / max(pred_total, 1), ap, true_total, pred_total

# 저장 ───────────────────────────────────────────────

def wrong_rows(rows, outputs, pred, ng):
    """오분류 샘플을 FN(불량 놓침) → FP(오검출) 순, 각 그룹은 판정 기준에서 먼 것부터."""
    wrong = []
    for i in np.flatnonzero(pred != ng):
        scores = outputs[i]["scores"]
        max_score = float(scores.max()) if len(scores) else 0.0
        wrong.append({
            "error": "FN" if ng[i] else "FP",
            "sample_id": rows[i]["sample_id"],
            "label": rows[i]["label"],
            "pred": "NG" if pred[i] else "OK",
            "max_score": f"{max_score:.4f}",
            "boxes": int((scores >= SCORE_THRESHOLD).sum()),
            "view": rows[i]["view"],
            "path": display_path(rows[i]["path"]),
            "source": display_path(rows[i]["source"]) if rows[i].get("source") else "",
            "_index": int(i),
        })
    wrong.sort(key=lambda r: (r["error"] != "FN", -abs(float(r["max_score"]) - SCORE_THRESHOLD)))
    return wrong


def save_wrong_csv(path, wrong):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글 경로가 깨지지 않게
        writer = csv.DictWriter(f, fieldnames=WRONG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(wrong)


def draw_boxes(gray, boxes, color, scores=None, scale=4):
    """타일을 scale배 키우고 박스(타일 좌표)를 그린다."""
    image = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        cv2.rectangle(image, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale) - 1, int(y2 * scale) - 1), color, 2)
        if scores is not None:
            cv2.putText(image, f"{scores[k]:.2f}", (int(x1 * scale) + 2, max(int(y1 * scale) - 3, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return image


def error_image(row, out, true_boxes, info):
    """FN/FP 한 장: 머리글 + A 원본 / B 평탄화 / 예측 박스(빨강) / 정답 박스(초록, split만)."""
    size = row["size"]
    pred_keep = out["scores"] >= info["low_score"]
    pred_boxes = scale_boxes(out["boxes"][pred_keep], size, to_input=False)
    base = row["gray_a"] if info["kind"] == "A" else row["gray_b"]
    panels = [("A 원본", draw_boxes(row["gray_a"], [], None)), ("B 평탄화", draw_boxes(row["gray_b"], [], None)),
              (f"{info['model_label']} 예측", draw_boxes(base, pred_boxes, (220, 30, 30), out["scores"][pred_keep]))]
    if true_boxes is not None:
        panels.append(("정답 박스", draw_boxes(base, scale_boxes(true_boxes, size, to_input=False), (30, 160, 60))))

    font_big, font = ImageFont.truetype(FONT_PATH, 24), ImageFont.truetype(FONT_PATH, 18)
    label_w, pad = 200, 12
    width = label_w + max(p.shape[1] for _, p in panels) + pad
    header = 120
    height = header + sum(p.shape[0] + pad for _, p in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    scores = out["scores"]
    max_score = float(scores.max()) if len(scores) else 0.0
    date = next((part for part in row["sample_id"].split("_") if part.isdigit() and len(part) == 8), "")
    draw.text((pad, 8), f"{info['error']}  |  실제 {row['label']} → 판정 {info['pred']}  |  {info['model_label']}  |  {info['eval_set']}",
              fill=(200, 0, 0), font=font_big)
    draw.text((pad, 44), f"{row['sample_id']}  · {row['view']} · {date}", fill=(70, 70, 70), font=font)
    draw.text((pad, 70), f"박스 {int((scores >= SCORE_THRESHOLD).sum())}개 (점수 ≥ {SCORE_THRESHOLD}, NG 기준 1개) · 최대 점수 {max_score:.3f}"
              f" · 빨강은 점수 ≥ {info['low_score']} 박스", fill=(70, 70, 70), font=font)
    y = header
    for name, panel in panels:
        draw.text((pad, y + panel.shape[0] // 2 - 10), name, fill=(40, 120, 40), font=font)
        canvas.paste(Image.fromarray(panel), (label_w, y))
        y += panel.shape[0] + pad
    return canvas


def ball_slot_count(sample_id, view):
    """볼 슬롯 수: Back·Front 2개, Left·Right 3개. 07/14 Left 다이는 볼이 원래 2개라 2슬롯."""
    if view.capitalize() == "Left" and "_20260714_" in sample_id:
        return 2
    return 2 if view.capitalize() in ("Back", "Front") else 3


def ball_is_ng(source, sample_id, view):
    """볼 검사(run_side_detection)에서 슬롯 하나라도 '검출'이 아니면 볼 NG."""
    _, result = run_side_detection(source, ball_slot_count(sample_id, view))
    return any(status != BALL_STATUS_OK for *_, status in result.ball_slot_measurements)


def export_errors(out_dir, wrong, rows, outputs, targets, info):
    """dataset/errors와 같은 구조: {out_dir}/{FN,FP}/{sample_id}.png
    + 볼 검사에서 이미 NG인 샘플을 뺀 {FN,FP}_notbal/{origin,crop,result}/ (표면 모델이 놓치면 그대로 유출되는 샘플)"""
    excluded = 0
    for item in wrong:
        i = item["_index"]
        row = rows[i]
        dest = out_dir / item["error"]
        dest.mkdir(parents=True, exist_ok=True)
        image = error_image(row, outputs[i], targets[i] if targets is not None else None,
                            dict(info, error=item["error"], pred=item["pred"]))
        image.save(dest / f"{item['sample_id']}.png")

        source = row.get("source")
        if source is None or ball_is_ng(source, row["sample_id"], row["view"]):
            excluded += source is not None
            continue
        notbal = out_dir / f"{item['error']}_notbal"
        for folder in ("origin", "crop", "result"):
            (notbal / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, notbal / "origin" / Path(source).name)
        write_png(notbal / "crop" / f"{item['sample_id']}_A.png", row["gray_a"])
        write_png(notbal / "crop" / f"{item['sample_id']}_B.png", row["gray_b"])
        image.save(notbal / "result" / f"{item['sample_id']}.png")
    return excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="평가할 .pt 파일 (옆의 .json에서 A/B를 읽음)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="side_surface_det_v1 형식 데이터셋 폴더")
    parser.add_argument("--split", default="test", help="--data에서 평가할 split (train | val | test)")
    parser.add_argument("--tiff", type=Path, nargs="+", default=None,
                        help="원본 TIFF 폴더 (지정하면 --data 대신 즉석 크롭해서 이미지 단위로만 평가)")
    parser.add_argument("--views", nargs="+", default=None, help="--tiff에서 평가할 방향만 (예: Front Right)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output", type=Path, default=None,
                        help="오분류 CSV 경로 (기본: 모델 옆 {모델명}_wrong_{평가 이름}.csv)")
    parser.add_argument("--errors", type=Path, default=None, metavar="DIR",
                        help="오분류 비교 이미지를 {DIR}/{FN,FP}/에 저장 (예: dataset/errors/Detector/FRCNN_A/Test_1013)")
    args = parser.parse_args()

    # 모델 불러오기
    model_path = args.model
    config = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    kind = config["kind"]
    model, device = load_model(model_path, args.device, in_ch=len(kind))

    if args.tiff:
        rows, x = load_tiffs(args.tiff, args.views, kind)
        targets = None
        # 폴더 이름이 OK/NG뿐이면 상위 폴더 이름을 붙인다 (예: front_261006-NG)
        name = "_".join(
            f"{path.resolve().parent.name}-{path.name}" if path.name.upper() in ("OK", "NG") else path.resolve().name
            for path in args.tiff
        ) + ("_" + "".join(args.views) if args.views else "")
        source = ", ".join(display_path(path) for path in args.tiff)
    else:
        rows, x, targets = load_split(args.data, args.split, kind)
        attach_split_images(rows, args.data)
        name, source = args.split, display_path(args.data / kind / args.split)

    outputs = predict(model, x, device)
    pred = np.array([(out["scores"] >= SCORE_THRESHOLD).any() for out in outputs])
    ng = np.array([r["label"] == "NG" for r in rows])

    tp, fn = int((pred & ng).sum()), int((~pred & ng).sum())
    fp, tn = int((pred & ~ng).sum()), int((~pred & ~ng).sum())
    print(f"\n[모델 이름] {model_path.name} ({kind} 타일)  \n[데이터 위치] {source}  \n[판정 기준] 박스 점수 ≥ {SCORE_THRESHOLD}인 박스가 1개 이상 → NG")
    print(f"\n[전체 {len(rows)}장] \n[정확도] {(tp + tn) / len(rows):.4f}  \n[TP] {tp}  \n[FN] {fn}  \n[FP] {fp}  \n[TN] {tn}")

    views = np.array([r["view"].capitalize() for r in rows])
    for view in VIEWS:
        mask = views == view
        if mask.any():
            print(f"  {view:5s} {mask.sum()}장  오류 {int((pred[mask] != ng[mask]).sum())}"
                  f"  (FN {int((~pred[mask] & ng[mask]).sum())} · FP {int((pred[mask] & ~ng[mask]).sum())})")

    if targets is not None: # 정답 박스가 있을 때만 박스 단위 지표
        recall, precision, ap, true_total, pred_total = box_scores(outputs, targets)
        print(f"\n[박스 AP@{MATCH_IOU}] {ap:.4f}")
        print(f"[결함 재현율] {recall:.4f} (정답 박스 {true_total}개)  \n[결함 정밀도] {precision:.4f} (예측 박스 {pred_total}개)"
              f"  — IoU ≥ {MATCH_IOU}")

    wrong = wrong_rows(rows, outputs, pred, ng)
    csv_path = args.output or model_path.parent / f"{model_path.stem}_wrong_{name}.csv"
    save_wrong_csv(csv_path, wrong)
    print(f"\n[오분류] {len(wrong)}장 → {display_path(csv_path)}")

    if args.errors:
        info = dict(kind=kind, model_label=f"Faster R-CNN({kind})", eval_set=args.errors.name, low_score=0.1)
        excluded = export_errors(args.errors, wrong, rows, outputs, targets, info)
        print(f"[오류 이미지] {display_path(args.errors)}  (볼 NG로 _notbal에서 제외 {excluded}장)")


if __name__ == "__main__":
    main()
