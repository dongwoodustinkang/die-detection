"""표면 결함 Object Detection(Faster R-CNN, ResNet-18 FPN)을 A 또는 B 표면 타일로 학습한다."""

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.detection import (
    INPUT_W, NMS_IOU, SCORE_THRESHOLD, build_model, pick_device, predict, preprocess, scale_boxes,
)
from tools.make_data import display_path
from tools.train_classifier import next_model_stem
from tools.train_graphs import save_history_plot

DEFAULT_DATA = ROOT / "dataset" / "infer" / "side_surface_det_v1" # make_od_data.py 출력
LABELS = ("OK", "NG")
BATCH_SIZE = 32

# MODEL ───────────────────────────────────────────────

MODEL_DIR = ROOT / "models" / "Detector"
EPOCHS = 25
PATIENCE = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
TRAIN_ACC_SAMPLES = 500 # 학습세트 정확도는 매 epoch 이만큼만 뽑아 계산 (시간 절약)
EMPTY_CACHE_EVERY = 10  # MPS는 박스 수가 배치마다 달라 캐시가 계속 커지므로(수십 GB) 주기적으로 비운다


def read_boxes(data_dir):
    """boxes.csv → sample_id별 정답 박스 (타일 좌표)."""
    boxes = defaultdict(list)
    with open(data_dir / "boxes.csv", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            boxes[row["sample_id"]].append([float(row[key]) for key in ("x1", "y1", "x2", "y2")])
    return boxes


def load_split(data_dir, split, kind):
    """{kind}/{split}/{OK,NG}/*.png → rows, x (N, C, 32, 176), 입력 좌표 정답 박스 목록.
    kind="AB"는 같은 파일명의 A·B 타일을 채널로 겹친다 (C=2)."""
    all_boxes = read_boxes(data_dir)
    rows, inputs, targets = [], [], []
    for label in LABELS:
        for path in sorted((data_dir / kind[0] / split / label).glob("*.png")):
            parts = [preprocess(cv2.imread(str(data_dir / k / split / label / path.name), cv2.IMREAD_GRAYSCALE)) for k in kind]
            x, size = np.concatenate([part[0] for part in parts]), parts[0][1]
            rows.append({"sample_id": path.stem, "label": label, "view": path.stem.split("_")[1], "path": path, "size": size})
            inputs.append(x)
            targets.append(scale_boxes(all_boxes.get(path.stem, []), size))
    if not rows:
        raise SystemExit(f"이미지가 없습니다: {data_dir / kind[0] / split} (OK/, NG/ 하위 폴더 필요)")
    return rows, torch.from_numpy(np.stack(inputs)), targets


def to_target(boxes, device):
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        "labels": torch.ones(len(boxes), dtype=torch.int64, device=device), # 결함 = 1
    }


def random_hflip(image, boxes):
    """50% 확률로 좌우 반전 (박스도 같이). 위아래는 기둥·볼 위치가 달라 뒤집지 않는다."""
    if np.random.rand() < 0.5:
        return image, boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] = INPUT_W - boxes[:, [2, 0]]
    return image.flip(-1), boxes


def loss_mode(model):
    """손실만 계산하는 모드: torchvision 검출 모델은 train()에서만 손실을 주므로, BN 통계는 고정해서 val이 섞이지 않게 한다."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def image_accuracy(model, x, rows, device):
    """박스 점수 ≥ SCORE_THRESHOLD가 하나라도 있으면 NG로 본 이미지 단위 정확도."""
    model.eval()
    outputs = predict(model, x, device)
    pred = np.array([(out["scores"] >= SCORE_THRESHOLD).any() for out in outputs])
    true = np.array([row["label"] == "NG" for row in rows])
    return float((pred == true).mean())


def run_loss(model, x, targets, device, optimizer=None):
    """optimizer가 있으면 한 epoch 학습, 없으면 val 손실만. 평균 손실(4개 손실 합)을 돌려준다."""
    is_train = optimizer is not None
    model.train() if is_train else loss_mode(model)
    order = np.random.permutation(len(x)) if is_train else np.arange(len(x))
    total, count = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for start in range(0, len(order), BATCH_SIZE):
            images, batch_targets = [], []
            for i in order[start: start + BATCH_SIZE]:
                image, boxes = x[i], targets[i]
                if is_train:
                    image, boxes = random_hflip(image, boxes)
                images.append(image.to(device))
                batch_targets.append(to_target(boxes, device))
            loss = sum(model(images, batch_targets).values())
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
            if device.type == "mps" and (start // BATCH_SIZE) % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()
    return total / count


def train(model, data, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []} # acc = 이미지 OK/NG 정확도

    train_rows, train_x, train_targets = data["train"]
    val_rows, val_x, val_targets = data["val"]
    pick = np.random.default_rng(7).choice(len(train_x), min(TRAIN_ACC_SAMPLES, len(train_x)), replace=False)
    pick_rows = [train_rows[i] for i in pick]

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_loss(model, train_x, train_targets, device, optimizer) # 학습 에폭
        val_loss = run_loss(model, val_x, val_targets, device) # 검증 손실
        train_acc = image_accuracy(model, train_x[pick], pick_rows, device)
        val_acc = image_accuracy(model, val_x, val_rows, device)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(
            f"[{epoch:2d}] 1️⃣  학습세트 loss {train_loss:.4f} 정확도 {train_acc:.4f}"
            f" | 2️⃣  검증세트 loss {val_loss:.4f} 정확도 {val_acc:.4f}", flush=True,
        )
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"조기 종료: patience {PATIENCE} 도달")
                break

    print(f"[BEST👍] 최고 검증 epoch {best_epoch} : loss {best_loss:.4f}")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("A", "B", "AB"), required=True,
                        help="학습에 쓸 표면 타일 (A 원본 | B 평탄화 | AB 두 장을 채널로 겹침)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="side_surface_det_v1 형식 데이터셋 폴더")
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    device = pick_device(args.device)

    data = {split: load_split(args.data, split, args.kind) for split in ("train", "val", "test")}
    for split, (rows, x, targets) in data.items():
        ng_ratio = sum(r["label"] == "NG" for r in rows) / len(rows)
        print(f"[{split}] {len(rows)}장  x={tuple(x.shape)}  NG 비율 {ng_ratio:.2f}  정답 박스 {sum(len(t) for t in targets)}개")
    print(f"[device] {device}  [kind] {args.kind}")

    model = build_model(in_ch=len(args.kind)).to(device)
    model, history, best_epoch = train(model, data, device)

    test_rows, test_x, _ = data["test"]
    print(f"[test] 정확도 {image_accuracy(model, test_x, test_rows, device):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = next_model_stem(MODEL_DIR, prefix=f"side_surface_frcnn{args.kind}")
    weight_path = MODEL_DIR / f"{stem}.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, weight_path)
    save_history_plot(history, best_epoch, MODEL_DIR / f"{stem}.png")
    config = dict(
        kind=args.kind, crop="surface", data=display_path(args.data), model="Faster R-CNN (ResNet-18 FPN, scratch)",
        best_epoch=best_epoch, score_threshold=SCORE_THRESHOLD, nms_iou=NMS_IOU,
    )
    (MODEL_DIR / f"{stem}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {weight_path} (+ .png, .json)")


if __name__ == "__main__":
    main()
