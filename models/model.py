"""
완전한 YOLOv3 모델 구현 - 모든 컴포넌트 통합

구성 요소:
- Backbone: 특징 추출 (Darknet-53)
- Neck: 특징 융합 (FPN)  
- Head: 객체 검출
"""

import torch
import torch.nn as nn
from models.backbone import Backbone
from models.neck import Neck  
from configs.config import Config

class Head(nn.Module):
    """
    검출 헤드
    각 스케일별로 최종 예측 수행
    """
    def __init__(self, in_channels, num_classes, anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        
        # 출력 채널 계산: 앵커 수 × (bbox 4개 + 객체성 1개 + 클래스 수)
        out_channels = self.num_anchors * (5 + num_classes)  # 3 × (5 + 25) = 90
        
        # 최종 예측 레이어
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )
    
    def forward(self, x):
        return self.prediction(x)

class YOLOv3(nn.Module):
    """
    완전한 YOLOv3 모델
    완전한 YOLOv3 아키텍처 구현
    
    구조:
    Input → Backbone → FPN → Detection Heads → 3 Scale Outputs
    """
    def __init__(self, num_classes=25, anchors=None):
        super().__init__()
        self.num_classes = num_classes
        
        # Config에서 앵커 가져오기
        config = Config()
        if anchors is None:
            self.anchors = config.ANCHORS
        else:
            self.anchors = anchors
        
        print(f"모델 초기화: {num_classes}개 클래스, 3개 스케일 검출")
        
        # 1. 백본 네트워크 (특징 추출)
        self.backbone = Backbone()
        
        # 2. Neck 네트워크 (특징 융합)
        self.neck = Neck()
        
        # 3. 검출 헤드들 (최종 예측)
        self.head_small = Head(256, num_classes, self.anchors[0])    # 52x52
        self.head_medium = Head(512, num_classes, self.anchors[1])   # 26x26
        self.head_large = Head(1024, num_classes, self.anchors[2])   # 13x13
        
    def forward(self, x):
        """
        순전파 과정
        YOLOv3 흐름: Backbone → FPN → Heads
        """
        # 1. 백본에서 3개 스케일 특징 추출
        route1, route2, route3 = self.backbone(x)
        
        # 2. Neck으로 특징 융합 
        feat_small, feat_medium, feat_large = self.neck(route1, route2, route3)
        
        # 3. 각 스케일에서 검출 수행
        pred_small = self.head_small(feat_small)    # [B, 90, 52, 52] 작은 객체
        pred_medium = self.head_medium(feat_medium) # [B, 90, 26, 26] 중간 객체  
        pred_large = self.head_large(feat_large)    # [B, 90, 13, 13] 큰 객체
        
        return pred_small, pred_medium, pred_large
    
    def get_anchors(self):
        """앵커 박스 반환"""
        return self.anchors

def create_model(num_classes=25):
    """
    YOLOv3 모델 생성 함수
    """
    config = Config()
    model = YOLOv3(num_classes=num_classes, anchors=config.ANCHORS)
    return model

def test_complete_model():
    """완전한 모델 테스트"""
    print("=== 완전한 YOLOv3 모델 테스트 ===")
    print("완전한 YOLOv3 아키텍처")
    
    config = Config()
    model = create_model(config.NUM_CLASSES)
    
    # 테스트 입력
    test_input = torch.randn(2, 3, 416, 416)  # 배치 2개
    
    print(f"\n입력: {test_input.shape}")
    print(f"클래스 수: {config.NUM_CLASSES}")
    print(f"앵커 박스: {model.get_anchors()}")
    
    # 순전파
    with torch.no_grad():
        pred_small, pred_medium, pred_large = model(test_input)
    
    print(f"\n출력:")
    print(f"  작은 객체 (52x52): {pred_small.shape}")   # [2, 90, 52, 52]
    print(f"  중간 객체 (26x26): {pred_medium.shape}")  # [2, 90, 26, 26]
    print(f"  큰 객체 (13x13): {pred_large.shape}")    # [2, 90, 13, 13]
    
    print(f"\n모델 정보:")
    print(f"✅ 아키텍처: Backbone → FPN → Heads")
    print(f"✅ 25개 클래스")
    print(f"✅ 3스케일 검출")
    print(f"✅ 출력 형태: 3 × (5 + 25) = 90 채널")
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n총 파라미터 수: {total_params:,}")
    
    return model

if __name__ == "__main__":
    model = test_complete_model()