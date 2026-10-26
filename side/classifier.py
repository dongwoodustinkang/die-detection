import cv2
import numpy as np
import torch
import torch.nn as nn

from general import to_grayscale

INPUT_H, INPUT_W = 32, 176 # 모델 입력 크기 

def pick_device(name="auto"):
    # 운영체제에 따라 자동으로 GPU 장치 선택
    # CUDA -> MPS -> CPU
    if name != "auto":
        return torch.device(name) 
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def preprocess(image): # 모든 경로는 영문명으로 해야함! *윈도우에서 오류가 발생될 우려있음
    """Numpy 배열 이미지 전처리 작업 (그레이스케일, 정규화등)"""
    gray = to_grayscale(image) # 그레일스케일 변환
    if gray.dtype != np.uint8:
        raise ValueError(f"이미지의 색상 타입은 uint8 타입만 지원합니다.")
    x = cv2.resize(gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA) # 각 이미지를 고정 크기로 맞추기 위한 리사이장
    x = 1.0 - x.astype(np.float32) / 255.0 # 정규화와 색상 반전으로 결함 파악 
    return x[None]


def conv_block(in_ch, out_ch, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), # 컨볼루션 레이어
        nn.BatchNorm2d(out_ch), # 배치 정규화
        nn.ReLU(inplace=True) # 활성화 함수
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 2x2로 절반 감소
    return nn.Sequential(*layers) # 리스트를 nn.Sequential로 감싸서 레이어를 순차적으로 적용 


class SurfaceCNN(nn.Module):
    """## OK/NG 분류용 경량 CNN"""

    def __init__(self, in_ch=1):  # in_ch=2는 A·B를 채널로 겹친 입력
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_ch, 16), # 16, 16, 88
            conv_block(16, 32), # 32, 8, 44
            conv_block(32, 64), # 64, 4, 22
            conv_block(64, 64, pool=False),
        )
        self.classifier = nn.Linear(64, 1) # OK->1, NG->0으로 이진 분류


    def forward(self, x):
        f = self.features(x) # (64, 1, 32, 176) -> (64, 64, 4, 22)
        pooled = f.mean(dim=(2,3)) # (64, 64, 4, 22) -> (64, 64) # 88칸의 값을 평균으로 압축, 위치 정보 없앰
        return self.classifier(pooled).squeeze(1) # (64, 1) -> (64,)  



    