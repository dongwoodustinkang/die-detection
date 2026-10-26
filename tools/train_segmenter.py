import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.segmenter import SURFACE_H, SURFACE_W, SurfaceUNet, pick_device, preprocess
from tools.make_data import display_path
from tools.train_classifier import next_model_stem
from tools.train_graphs import save_history_plot

DEFAULT_DATA = ROOT / "dataset" / "infer" / "side_surface_od_v1" # 데이터셋 폴더 경로
BATCH_SIZE = 32
LABELS = ("OK", "NG")

# MODEL ───────────────────────────────────────────────

MODEL_DIR = ROOT / "models" / "Segmenter"
EPOCHS = 40
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4

# 판정 ───────────────────────────────────────────────

PIXEL_THRESHOLD = 0.5   # 픽셀 결함 확률이 이 값 이상이면 결함 픽셀
MIN_DEFECT_PIXELS = 4   # 결함 픽셀이 이 개수 이상이면 이미지를 NG로 판정 (make_sg_data.py의 MIN_BOX_PIXELS와 같음)


def load_split(data_dir, split, crop="fixed"):
    """A/{split}, B/{split}, mask/{split}의 같은 파일명을 읽어 (rows, x, y) 텐서로 만든다.
    crop="surface"(side_surface_det_v1 등)는 타일·마스크를 SURFACE_H×SURFACE_W로 맞춘다."""
    rows, inputs, masks = [], [], []
    for label in LABELS:
        for path in sorted((data_dir / "A" / split / label).glob("*.png")):
            relative = path.relative_to(data_dir / "A")
            image_a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            image_b = cv2.imread(str(data_dir / "B" / relative), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(data_dir / "mask" / relative), cv2.IMREAD_GRAYSCALE)
            if crop == "surface":
                mask = cv2.resize(mask, (SURFACE_W, SURFACE_H), interpolation=cv2.INTER_NEAREST)
            rows.append({"sample_id": path.stem, "label": label, "view": path.stem.split("_")[1], "path": path})
            inputs.append(preprocess(image_a, image_b, crop)) # (2, 48, 192) | (2, 32, 176)
            masks.append((mask > 0).astype(np.float32)[None]) # 0/255 → 0/1, (1, 48, 192)
    if not rows:
        raise SystemExit(f"이미지가 없습니다: {data_dir / 'A' / split} (OK/, NG/ 하위 폴더 필요)")
    return rows, torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(masks))


def dice_loss(logits, target, eps=1.0):
    """배치 전체 픽셀로 계산한 1 - Dice. 결함이 아주 작아도 결함 픽셀을 놓치면 크게 손해를 보게 한다."""
    prob = logits.sigmoid()
    inter = (prob * target).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


def random_hflip(x, y):
    """샘플마다 50% 확률로 좌우 반전 (입력과 마스크를 같이). 위아래는 기둥·볼 위치가 달라 뒤집지 않는다."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x, y = x.clone(), y.clone()
    x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
    return x, y


def run_epoch(model, loader, bce, device, optimizer=None):
    """optimizer에 따라 학습 or 평가. (loss, 이미지 OK/NG 정확도, 픽셀 Dice)를 돌려준다."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, count = 0.0, 0, 0
    inter, pred_sum, true_sum = 0.0, 0.0, 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                x, y = random_hflip(x, y)
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)

            pred = (out.sigmoid() >= PIXEL_THRESHOLD).float()
            inter += (pred * y).sum().item()
            pred_sum += pred.sum().item()
            true_sum += y.sum().item()
            pred_ng = pred.sum(dim=(1, 2, 3)) >= MIN_DEFECT_PIXELS # 이미지 판정
            true_ng = y.sum(dim=(1, 2, 3)) > 0
            correct += (pred_ng == true_ng).sum().item()
            count += len(y)
    dice = 2 * inter / (pred_sum + true_sum) if pred_sum + true_sum else 1.0
    return total_loss / count, correct / count, dice


def train(model, loaders, device):
    bce = nn.BCEWithLogitsLoss() # 픽셀별 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []} # 그래프용 epoch별 기록 (acc = 이미지 OK/NG 정확도)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_dice = run_epoch(model, loaders["train"], bce, device, optimizer) # 학습 에폭
        val_loss, val_acc, val_dice = run_epoch(model, loaders["val"], bce, device) # 검증 에폭
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"[{epoch:2d}] 1️⃣  학습세트 loss {train_loss:.4f} 정확도 {train_acc:.4f} Dice {train_dice:.4f}"
            f" | 2️⃣  검증세트 loss {val_loss:.4f} 정확도 {val_acc:.4f} Dice {val_dice:.4f}"
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
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="side_surface_od_v1 형식 데이터셋 폴더")
    parser.add_argument("--crop", choices=("fixed", "surface"), default="fixed",
                        help="fixed: 48×192 고정 크롭(side_surface_od_v1) | surface: 표면 타일(side_surface_det_v1)")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = pick_device(args.device)

    data = {split: load_split(args.data, split, args.crop) for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=(split == "train"))
        for split, (_, x, y) in data.items()
    }
    for split, (rows, x, y) in data.items():
        ng_ratio = sum(r["label"] == "NG" for r in rows) / len(rows)
        print(f"[{split}] {len(rows)}장  x={tuple(x.shape)}  NG 비율 {ng_ratio:.2f}  결함 픽셀 비율 {y.mean():.4f}")
    print(f"[device] {device}")

    model = SurfaceUNet().to(device)
    model, history, best_epoch = train(model, loaders, device)

    test_loss, test_acc, test_dice = run_epoch(model, loaders["test"], nn.BCEWithLogitsLoss(), device)
    print(f"[test] loss {test_loss:.4f} 정확도 {test_acc:.4f} Dice {test_dice:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = next_model_stem(MODEL_DIR, prefix="side_surface_unetAB")
    weight_path = MODEL_DIR / f"{stem}.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, weight_path)
    save_history_plot(history, best_epoch, MODEL_DIR / f"{stem}.png")
    config = dict(kind="AB", crop=args.crop, data=display_path(args.data), best_epoch=best_epoch,
                  pixel_threshold=PIXEL_THRESHOLD, min_defect_pixels=MIN_DEFECT_PIXELS)
    (MODEL_DIR / f"{stem}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {weight_path} (+ .png, .json)")


if __name__ == "__main__":
    main()