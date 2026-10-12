import cv2
import numpy as np
import torch

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



    