import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from general import to_grayscale
from side.classifier import pick_device

# 고정 크롭 크기
INPUT_H, INPUT_W = 48, 192
SURFACE_H, SURFACE_W = 32, 176  # 표면 타일(약 33×175)은 분류기·검출과 같은 크기로 맞춘다
IN_CHANNELS = 2

def preprocess(image_a, image_b, crop="fixed"):
    """이미지 전처리 작업(그레이 스케일, 정규화). crop="surface"면 표면 타일을 SURFACE_H×SURFACE_W로 resize한다."""
    channels = []
    for image in (image_a, image_b):
        gray = to_grayscale(image)
        if gray.dtype != np.uint8:
            raise ValueError("이미지의 색상 타입은 uint8 타입만 지원합니다.")
        if crop == "surface":
            gray = cv2.resize(gray, (SURFACE_W, SURFACE_H), interpolation=cv2.INTER_AREA)
        elif gray.shape != (INPUT_H, INPUT_W):
            raise ValueError(f"입력 크기는 {INPUT_H}×{INPUT_W}이어야 합니다: {gray.shape}")
        channels.append(1.0 - gray.astype(np.float32) / 255.0)  # 정규화와 색상 반전으로 결함 파악 
    return np.stack(channels)


def double_conv(in_ch, out_ch):
    # 두번의 컨볼루션 레이어를 순차적으로 적용
    return nn.Sequential(
        # 1
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        # 2
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class SurfaceUNet(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 사이즈로 절반 감소

        # 인코딩
        self.enc1 = double_conv(in_ch, 16)  # 16, 48, 192
        self.enc2 = double_conv(16, 32)           # 32, 24, 96
        self.enc3 = double_conv(32, 64)           # 64, 12, 48
        self.middle = double_conv(64, 64)         # 64,  6, 24

        # 디코딩 : 키우면서 같은 크기의 인코더 특징(skip)과 합친다
        self.dec3 = double_conv(64 + 64, 32)      # 32, 12, 48
        self.dec2 = double_conv(32 + 32, 16)      # 16, 24, 96
        self.dec1 = double_conv(16 + 16, 16)      # 16, 48, 192
        self.head = nn.Conv2d(16, 1, kernel_size=1)  # 픽셀별 logit

    @staticmethod
    def up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 2배 확대

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.head(d1)