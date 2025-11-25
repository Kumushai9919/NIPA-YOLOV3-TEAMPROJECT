"""
Feature Pyramid Network (FPN) - 특징 피라미드 네트워크
Neck 역할을 담당하는 네트워크

역할:
- 백본에서 나온 3개 스케일 특징맵을 융합
- Top-down pathway로 고수준 의미정보를 저수준에 전달
- 각 스케일별로 검출에 최적화된 특징맵 생성
"""

import torch
import torch.nn as nn
from models.backbone import ConvBlock

class Neck(nn.Module):
    """
    Neck 네트워크 (Feature Pyramid Network)
    YOLO의 neck 부분을 담당하며 이해하기 쉽게 구현
    
    작동 원리:
    1. 가장 큰 특징맵(13x13)부터 시작
    2. 업샘플링하여 다음 스케일과 합침  
    3. 각 스케일에서 검출용 특징맵 생성
    """
    def __init__(self):
        super().__init__()
        
        # Scale 1: 13x13 (큰 객체) - 1024 channels 처리
        self.conv_set1 = self._make_conv_set(1024, 512)
        self.conv1x1_1 = ConvBlock(512, 256, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Scale 2: 26x26 (중간 객체) - 512 + 256 = 768 channels 처리  
        self.conv_set2 = self._make_conv_set(768, 256)
        self.conv1x1_2 = ConvBlock(256, 128, 1) 
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Scale 3: 52x52 (작은 객체) - 256 + 128 = 384 channels 처리
        self.conv_set3 = self._make_conv_set(384, 128)
        
        # 최종 검출용 특징맵 생성
        self.detection_conv1 = ConvBlock(512, 1024, 3, padding=1)  # 13x13용
        self.detection_conv2 = ConvBlock(256, 512, 3, padding=1)   # 26x26용  
        self.detection_conv3 = ConvBlock(128, 256, 3, padding=1)   # 52x52용
    
    def _make_conv_set(self, in_channels, out_channels):
        """
        표준 Conv×5 패턴
        연속된 합성곱으로 특징 정제
        """
        return nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels * 2, 3, padding=1),
            ConvBlock(out_channels * 2, out_channels, 1),
            ConvBlock(out_channels, out_channels * 2, 3, padding=1),
            ConvBlock(out_channels * 2, out_channels, 1)
        )
    
    def forward(self, route1, route2, route3):
        """
        특징 피라미드 구성
        
        Args:
            route1: [B, 256, 52, 52] - 작은 객체용 특징맵
            route2: [B, 512, 26, 26] - 중간 객체용 특징맵  
            route3: [B, 1024, 13, 13] - 큰 객체용 특징맵
            
        Returns:
            detection_features: 3개 스케일의 검출용 특징맵
        """
        
        # Scale 1: 13x13 (큰 객체) 처리
        x3 = self.conv_set1(route3)           # [B, 512, 13, 13]
        detection_feat3 = self.detection_conv1(x3)  # [B, 1024, 13, 13]
        
        # Top-down: 13x13 → 26x26로 업샘플링
        x3_up = self.conv1x1_1(x3)           # [B, 256, 13, 13]
        x3_up = self.upsample1(x3_up)        # [B, 256, 26, 26]
        
        # Scale 2: 26x26 (중간 객체) - 융합
        x2 = torch.cat([route2, x3_up], dim=1)  # [B, 768, 26, 26]
        x2 = self.conv_set2(x2)                 # [B, 256, 26, 26]
        detection_feat2 = self.detection_conv2(x2)  # [B, 512, 26, 26]
        
        # Top-down: 26x26 → 52x52로 업샘플링
        x2_up = self.conv1x1_2(x2)           # [B, 128, 26, 26] 
        x2_up = self.upsample2(x2_up)        # [B, 128, 52, 52]
        
        # Scale 3: 52x52 (작은 객체) - 융합
        x1 = torch.cat([route1, x2_up], dim=1)  # [B, 384, 52, 52]
        x1 = self.conv_set3(x1)                 # [B, 128, 52, 52]
        detection_feat1 = self.detection_conv3(x1)  # [B, 256, 52, 52]
        
        # 표준 순서로 반환: [작은, 중간, 큰]
        return detection_feat1, detection_feat2, detection_feat3

def test_neck():
    """Neck 테스트"""
    print("=== Neck Network 테스트 ===")
    
    neck = Neck()
    
    # 백본에서 나올 것 같은 특징맵들 시뮬레이션
    route1 = torch.randn(1, 256, 52, 52)   # 작은 객체용
    route2 = torch.randn(1, 512, 26, 26)   # 중간 객체용
    route3 = torch.randn(1, 1024, 13, 13)  # 큰 객체용
    
    with torch.no_grad():
        feat1, feat2, feat3 = neck(route1, route2, route3)
    
    print(f"백본 출력:")
    print(f"  Route 1: {route1.shape}")
    print(f"  Route 2: {route2.shape}")
    print(f"  Route 3: {route3.shape}")
    
    print(f"Neck 출력 (검출용):")
    print(f"  작은 객체: {feat1.shape}")  # [1, 256, 52, 52]
    print(f"  중간 객체: {feat2.shape}")  # [1, 512, 26, 26] 
    print(f"  큰 객체: {feat3.shape}")   # [1, 1024, 13, 13]
    
    return neck

if __name__ == "__main__":
    test_neck()