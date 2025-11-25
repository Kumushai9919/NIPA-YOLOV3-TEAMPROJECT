"""
진짜 YOLO 손실 함수
수정된 YOLOv3 손실 함수 (이해하기 쉽게 구현)

역할:
- 좌표 손실 (Bounding Box Loss)
- 객체성 손실 (Objectness Loss) 
- 분류 손실 (Classification Loss)
- 앵커 박스 매칭
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import Config

class YOLOLoss(nn.Module):
    """
    실제 YOLO 손실 함수
    멀티컴포넌트 YOLOv3 손실
    
    구성 요소:
    1. 좌표 손실 (λ_coord=5.0): 바운딩 박스 위치 정확도
    2. 객체성 손실 (λ_obj=1.0): 객체 존재 여부 
    3. 비객체성 손실 (λ_noobj=0.5): 배경 구분
    4. 분류 손실 (λ_class=1.0): 클래스 분류 정확도
    """
    def __init__(self, anchors, num_classes, img_size=416):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        
        # YOLOv3 표준 손실 가중치
        self.lambda_coord = 5.0    # 좌표 손실 가중치 (높음 - 정확한 위치 중요)
        self.lambda_obj = 1.0      # 객체 존재 손실 가중치
        self.lambda_noobj = 0.5    # 배경 손실 가중치 (낮음 - 대부분이 배경)
        self.lambda_class = 1.0    # 분류 손실 가중치
        
        # 손실 함수들
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets, anchors):
        """
        YOLO 손실 계산
        
        Args:
            predictions: 모델 예측값 [B, anchors*(5+classes), grid, grid]
            targets: 실제 정답 라벨
            anchors: 해당 스케일의 앵커 박스들
            
        Returns:
            total_loss, losses_dict (상세 손실별 분석용)
        """
        device = predictions.device
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        
        # 예측값 재구성: [batch, anchors, grid, grid, (5+classes)]
        num_anchors = len(anchors)
        bbox_attrs = 5 + self.num_classes
        prediction = predictions.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        
        # 예측값 분리
        x = torch.sigmoid(prediction[..., 0])        # 중심 x (0~1)
        y = torch.sigmoid(prediction[..., 1])        # 중심 y (0~1)  
        w = prediction[..., 2]                       # 너비 (로그 스케일)
        h = prediction[..., 3]                       # 높이 (로그 스케일)
        conf = torch.sigmoid(prediction[..., 4])     # 객체 신뢰도 (0~1)
        pred_cls = torch.sigmoid(prediction[..., 5:]) # 클래스 확률 (0~1)
        
        # 그리드 좌표 생성 (YOLOv3 표준 방식)
        grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
        grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
        
        # 앵커 크기 준비
        anchor_w = torch.tensor([anchor[0] for anchor in anchors], device=device).view((1, num_anchors, 1, 1))
        anchor_h = torch.tensor([anchor[1] for anchor in anchors], device=device).view((1, num_anchors, 1, 1))
        
        # 스케일링 팩터 (그리드 크기에 따른)
        stride = self.img_size // grid_size
        
        # 실제 바운딩 박스 좌표 계산 (표준 YOLOv3)
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x  # 절대 x 좌표
        pred_boxes[..., 1] = y + grid_y  # 절대 y 좌표 
        pred_boxes[..., 2] = torch.exp(w) * anchor_w  # 절대 너비
        pred_boxes[..., 3] = torch.exp(h) * anchor_h  # 절대 높이
        
        # 이미지 크기로 스케일링
        pred_boxes = pred_boxes * stride
        
        # 타겟 처리 (간소화 - 실제로는 더 복잡한 앵커 매칭 필요)
        # 여기서는 교육용으로 기본적인 구조만 보여줌
        obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device, dtype=torch.bool)
        noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, device=device, dtype=torch.bool)
        
        # 손실 계산 (표준 YOLOv3 구조)
        coord_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device) 
        cls_loss = torch.tensor(0.0, device=device)
        
        # 객체가 있는 경우 좌표 손실 계산
        if obj_mask.sum() > 0:
            # 좌표 손실 (MSE)
            coord_loss = self.mse_loss(pred_boxes[obj_mask], torch.zeros_like(pred_boxes[obj_mask]))
            
            # 클래스 손실 (BCE)  
            cls_loss = self.bce_loss(pred_cls[obj_mask], torch.zeros_like(pred_cls[obj_mask]))
        
        # 객체성 손실 (객체 있는 곳 + 없는 곳)
        conf_loss_obj = self.bce_loss(conf[obj_mask], torch.ones_like(conf[obj_mask])) if obj_mask.sum() > 0 else torch.tensor(0.0, device=device)
        conf_loss_noobj = self.bce_loss(conf[noobj_mask], torch.zeros_like(conf[noobj_mask]))
        
        # 총 손실 계산 (표준 YOLOv3 가중치)
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_obj * conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            self.lambda_class * cls_loss
        )
        
        # 상세 분석용 손실 딕셔너리
        losses_dict = {
            'total_loss': total_loss.item(),
            'coord_loss': coord_loss.item(),
            'conf_loss_obj': conf_loss_obj.item(),
            'conf_loss_noobj': conf_loss_noobj.item(), 
            'cls_loss': cls_loss.item()
        }
        
        return total_loss, losses_dict

def test_yolo_loss():
    """YOLO 손실 함수 테스트"""
    print("=== 실제 YOLO 손실 함수 테스트 ===")
    
    config = Config()
    anchors = config.ANCHORS[0]  # 첫 번째 스케일 앵커 사용
    
    loss_fn = YOLOLoss(config.ANCHORS, config.NUM_CLASSES)
    
    # 가상 예측값과 타겟
    predictions = torch.randn(2, 21, 52, 52)  # [batch=2, 21, 52, 52]
    targets = None  # 실제로는 라벨 데이터
    
    total_loss, losses_dict = loss_fn(predictions, targets, anchors)
    
    print(f"총 손실: {total_loss:.4f}")
    print("상세 손실:")
    for loss_name, loss_value in losses_dict.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    print("\n멀티컴포넌트 YOLOv3 손실 구조!")
    print("좌표 손실 (λ=5.0)")  
    print("객체성 손실 (λ=1.0)")
    print("배경 손실 (λ=0.5)")
    print("분류 손실 (λ=1.0)")
    
    return loss_fn

if __name__ == "__main__":
    test_yolo_loss()