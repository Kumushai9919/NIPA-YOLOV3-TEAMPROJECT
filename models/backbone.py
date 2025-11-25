"""
백본 네트워크 (Backbone Network)
YOLOv3의 특징 추출기 - Darknet-53 구현

역할:
- 입력 이미지에서 다양한 스케일의 특징맵 추출
- 3개의 다른 해상도 출력: 52x52, 26x26, 13x13
- ResidualBlock을 사용한 깊은 네트워크 구성
"""

import torch
import torch.nn as nn
from configs.config import Config

class ConvBlock(nn.Module):
    """
    기본 합성곱 블록
    Conv2d → BatchNorm2d → LeakyReLU 순서
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    잔차 블록 (Residual Block)
    Skip Connection으로 그래디언트 소실 방지
    """
    def __init__(self, channels):
        super().__init__()
        reduced_channels = channels // 2
        
        # 1x1 conv로 채널 축소 → 3x3 conv → 1x1 conv로 원래 채널 복원
        self.conv1 = ConvBlock(channels, reduced_channels, 1)
        self.conv2 = ConvBlock(reduced_channels, channels, 3, padding=1)
    
    def forward(self, x):
        # Skip connection: 입력 + 변환된 출력
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class Backbone(nn.Module):
    """
    Darknet-53 백본 네트워크
    완전한 구현 (이해하기 쉽게 유지)
    
    특징:
    - 53개 합성곱 레이어
    - 5번의 다운샘플링 (416→208→104→52→26→13)
    - 3개의 스케일 출력 (multi-scale detection용)
    """
    def __init__(self):
        super().__init__()
        
        # 초기 레이어
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        self.conv2 = ConvBlock(32, 64, 3, stride=2, padding=1)
        
        # 첫 번째 Residual 스테이지 [64 channels]
        self.res_block1 = self._make_layer(64, 1)
        self.conv3 = ConvBlock(64, 128, 3, stride=2, padding=1)
        
        # 두 번째 Residual 스테이지 [128 channels]  
        self.res_block2 = self._make_layer(128, 2)
        self.conv4 = ConvBlock(128, 256, 3, stride=2, padding=1)
        
        # 세 번째 Residual 스테이지 [256 channels] - Route 1 출력
        self.res_block3 = self._make_layer(256, 8)
        self.conv5 = ConvBlock(256, 512, 3, stride=2, padding=1)
        
        # 네 번째 Residual 스테이지 [512 channels] - Route 2 출력
        self.res_block4 = self._make_layer(512, 8) 
        self.conv6 = ConvBlock(512, 1024, 3, stride=2, padding=1)
        
        # 다섯 번째 Residual 스테이지 [1024 channels] - Route 3 출력
        self.res_block5 = self._make_layer(1024, 4)
    
    def _make_layer(self, channels, num_blocks):
        """
        연속된 Residual Block들을 생성
        체계적으로 구성
        """
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass - 3개의 스케일 특징맵 반환
        표준 YOLOv3 출력 구조
        """
        # 초기 특징 추출
        x = self.conv1(x)      # 416x416x32
        x = self.conv2(x)      # 208x208x64
        x = self.res_block1(x) # 208x208x64
        
        x = self.conv3(x)      # 104x104x128
        x = self.res_block2(x) # 104x104x128
        
        x = self.conv4(x)      # 52x52x256
        route_1 = self.res_block3(x)  # 52x52x256 (small objects)
        
        x = self.conv5(route_1)  # 26x26x512
        route_2 = self.res_block4(x)  # 26x26x512 (medium objects)
        
        x = self.conv6(route_2)  # 13x13x1024
        route_3 = self.res_block5(x)  # 13x13x1024 (large objects)
        
        # 표준 순서: [작은, 중간, 큰] 객체용 특징맵
        return route_1, route_2, route_3

def test_backbone():
    """백본 네트워크 테스트"""
    print("=== Backbone 네트워크 테스트 ===")
    
    backbone = Backbone()
    test_input = torch.randn(1, 3, 416, 416)
    
    with torch.no_grad():
        route1, route2, route3 = backbone(test_input)
    
    print(f"입력: {test_input.shape}")
    print(f"Route 1 (작은 객체): {route1.shape}")  # [1, 256, 52, 52]
    print(f"Route 2 (중간 객체): {route2.shape}")  # [1, 512, 26, 26] 
    print(f"Route 3 (큰 객체): {route3.shape}")   # [1, 1024, 13, 13]
    
    return backbone

if __name__ == "__main__":
    test_backbone()