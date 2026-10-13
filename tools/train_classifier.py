import argparse
import sys
from pathlib import Path

from datetime import datetime 
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.classifier import SurfaceCNN, pick_device, preprocess

DEFAULT_DATA = ROOT / "dataset" / "infer" / "side_surface_v1" # 데이터셋 폴더 경로
BATCH_SIZE = 64
LABELS = ("OK", "NG")

# MODEL ───────────────────────────────────────────────

MODEL_DIR = ROOT / "models"
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4


def load_split(data_dir, split):
    """이미지를 읽어 (rows, x, y) 텐서로 만든다."""
    rows, tiles = [],[] # 샘플 정보와 이미지(numpy 배열)
    for target, label in enumerate(LABELS):
        for path in sorted((data_dir / "B" / split / label).glob("*.png")):
           rows.append({"sample_id": path.stem, "label": label, "view": path.stem.split("_")[1]}) # 샘플 정보를 row에 담기(파일명, 라벨, 촬영 방향)
           tiles.append(preprocess(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))) # 이미지 전처리
    if not rows:
        raise SystemExit(f"이미지가 없습니다: {data_dir / 'B' / split}")
    x = torch.from_numpy(np.stack(tiles)) # (6000, 1, 32, 176) to tensor
    y = torch.tensor([float(LABELS.index(r["label"])) for r in rows]) # label 문자열 -> 0, 1로 변환 (float)
    return rows, x, y


def next_model_stem(model_dir, prefix="side_surface_cnn"):
    """{prefix}_MMDDHH, 같은 시간대에 이미 있으면 _01, _02 ... 를 붙인 파일명(확장자 제외)."""
    stamp = datetime.now().strftime("%m%d%H")
    stem = f"{prefix}_{stamp}"
    n = 0
    while any((model_dir / f"{stem}{ext}").exists() for ext in (".pt", ".json")):
        n += 1
        stem = f"{prefix}_{stamp}_{n:02d}"
    return stem


def run_epoch(model, loader, criterion, device, optimizer=None):
    """optimizer에 따라 학습 or 평가"""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.sigmoid().round() == y).float().sum().item()
            count += len(y)
    return total_loss / count, correct / count



def train(model, loaders, device):
    criterion = nn.BCEWithLogitsLoss() # 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0 ,0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer) # 학습 에폭
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device) # 검증 에폭
        print(f"[{epoch:2d}] 1️⃣ 학습세트 loss {train_loss:.4f} 정확도 {train_acc:.4f} | 2️⃣ 검증세트 loss {val_loss:.4f} 정확도 {val_acc:.4f}")
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"조기 종료: patience {PATIENCE} 도달")
                break

    print(f"[BEST👍] 최고 검증 epoch {best_epoch} : loss {best_loss:.4f}")
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = pick_device(args.device)
    data_dir = DEFAULT_DATA 

    data = {split: load_split(data_dir, split) for split in ("train", "val", "test")} 
    loaders = {
        split: DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=(split == "train"))
        for split, (_, x, y) in data.items()
    }
    for split, (rows, x, y) in data.items():
        print(f"[{split}] {len(rows)}장  x={tuple(x.shape)}  NG 비율 {y.mean():.2f}")
    print(f"[device] {device}")

    model = SurfaceCNN().to(device)
    model = train(model, loaders, device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = next_model_stem(MODEL_DIR)
    weight_path = MODEL_DIR / f"{stem}.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, weight_path)
    print(f"저장: {weight_path}")


if __name__ == "__main__":
    main()


