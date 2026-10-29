import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

from general import to_grayscale
from side.classifier import pick_device

# 표면 타일(약 33×175)을 분류기·U-Net(surface)과 같은 크기로 맞춘다
INPUT_H, INPUT_W = 32, 176
UPSCALE = 4  # 모델 안에서 128×704로 키워 1~2px 높이의 가는 결함도 anchor에 걸리게 한다

NUM_CLASSES = 2       # 배경 + 결함
SCORE_THRESHOLD = 0.5 # 박스 점수가 이 값 이상인 박스가 하나라도 있으면 NG
NMS_IOU = 0.3         # 겹친 박스 제거 기준

# FPN 단계별 anchor 크기(128×704 기준 px)와 세로/가로 비율. 결함은 대부분 가로로 긴 사선·선 (중앙값 51×8)
ANCHOR_SIZES = ((12,), (24,), (48,), (96,), (192,))
ANCHOR_RATIOS = ((0.1, 0.25, 0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)


def preprocess(image):
    """표면 타일 → (1, 32, 176) float32, 반전·0~1 정규화. 원래 크기도 돌려준다 (박스 좌표 변환용)."""
    gray = to_grayscale(image)
    if gray.dtype != np.uint8:
        raise ValueError("이미지의 색상 타입은 uint8 타입만 지원합니다.")
    height, width = gray.shape[:2]
    x = cv2.resize(gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    x = 1.0 - x.astype(np.float32) / 255.0 # 정규화와 색상 반전으로 결함 파악
    return x[None], (height, width)


def scale_boxes(boxes, size, to_input=True):
    """타일 좌표 ↔ 모델 입력(32×176) 좌표. boxes: (N, 4) x1, y1, x2, y2."""
    height, width = size
    sx, sy = INPUT_W / width, INPUT_H / height
    scale = np.array([sx, sy, sx, sy], np.float32)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return boxes * scale if to_input else boxes / scale


def build_model(in_ch=1):
    """ResNet-18 + FPN backbone(처음부터 학습)을 쓰는 Faster R-CNN. in_ch=2는 A·B를 채널로 겹친 입력."""
    resnet = resnet18(weights=None, norm_layer=nn.BatchNorm2d) # 사전학습 가중치 없이, BN도 같이 학습
    resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False) # 그레이스케일 채널 수
    backbone = _resnet_fpn_extractor(resnet, trainable_layers=5) # stride 4·8·16·32 + pool

    representation = 256 # 기본 1024보다 작게 해서 가중치 크기를 줄인다
    return FasterRCNN(
        backbone,
        num_classes=None,
        min_size=INPUT_H * UPSCALE, max_size=INPUT_W * UPSCALE,
        image_mean=[0.0] * in_ch, image_std=[1.0] * in_ch, # preprocess에서 이미 정규화
        rpn_anchor_generator=AnchorGenerator(ANCHOR_SIZES, ANCHOR_RATIOS),
        box_head=TwoMLPHead(backbone.out_channels * 7 * 7, representation),
        box_predictor=FastRCNNPredictor(representation, NUM_CLASSES),
        box_score_thresh=0.05, # 낮은 점수 박스도 남겨 분석에 쓴다 (판정은 SCORE_THRESHOLD)
        box_nms_thresh=NMS_IOU,
        box_detections_per_img=10,
    )


def load_model(path, device="auto", in_ch=1):
    device = pick_device(device) if isinstance(device, str) else device
    model = build_model(in_ch)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval(), device


def predict(model, x, device, batch_size=32):
    """(N, C, 32, 176) → 샘플마다 {"boxes": (K, 4) 입력 좌표, "scores": (K,)} numpy."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = [image.to(device) for image in x[start: start + batch_size]]
            for out in model(batch):
                outputs.append({"boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy()})
            if device.type == "mps":
                torch.mps.empty_cache() # 검출 결과 크기가 매번 달라 MPS 캐시가 계속 커지는 것을 막는다
    return outputs
