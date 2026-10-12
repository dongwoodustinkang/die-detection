import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from side.classifier import pick_device, preprocess

DEFAULT_DATA = ROOT / "dataset" / "infer" / "side_surface_v1" # 데이터셋 폴더 경로
BATCH_SIZE = 64
LABELS = ("OK", "NG")


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
    print(f"device={device}")


if __name__ == "__main__":
    main()


