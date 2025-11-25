"""
생산용 YOLOv3 훈련 스크립트
Production YOLOv3 Training Script

GPU 최적화 및 실제 데이터셋 훈련을 위한 전문 스크립트
Professional script optimized for GPU training with real datasets

Usage:
    python train_production.py --data-dir /path/to/dataset --epochs 100 --batch-size 32
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
from pathlib import Path
# from tqdm import tqdm  # 현재 주석 처리
import time
import json
from datetime import datetime

# 프로젝트 모듈 import
import sys
import os
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import YOLOv3, create_model
from training.loss import YOLOLoss
from training.yolo_dataset import YOLODataset
from configs.config import Config

class ProductionTrainer:
    """
    생산용 YOLOv3 훈련 클래스
    GPU 최적화, 체크포인트 시스템, 실제 데이터 처리
    """
    
    def __init__(self, args):
        """
        훈련 환경 초기화
        
        Args:
            args: 커맨드라인 인자
        """
        self.args = args
        self.setup_device()
        self.setup_config()
        self.setup_model()
        self.setup_loss_and_optimizer()
        self.setup_dataloaders()
        self.setup_checkpoints()
        
        # 훈련 메트릭 추적
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print("\n" + "="*60)
        print("생산용 YOLOv3 훈련 준비 완료")
        print("="*60)
        
    def setup_device(self):
        """디바이스 설정 (GPU 우선, CPU 대비)"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"GPU 감지: {torch.cuda.get_device_name()}")
            print(f"   CUDA 버전: {torch.version.cuda}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # GPU 최적화 설정
            torch.backends.cudnn.benchmark = True  # 성능 최적화
            torch.backends.cudnn.deterministic = False  # 속도 우선
            
        else:
            self.device = torch.device('cpu')
            print("GPU를 사용할 수 없습니다. CPU로 훈련합니다.")
            print("   GPU 훈련을 위해서는 PyTorch GPU 버전을 설치하세요.")
            
    def setup_config(self):
        """설정 로드 및 수정"""
        self.config = Config()
        
        # 커맨드라인 인자로 설정 덮어쓰기
        if self.args.num_classes:
            self.config.NUM_CLASSES = self.args.num_classes
        if self.args.img_size:
            self.config.IMG_SIZE = self.args.img_size
            
        print(f"훈련 설정:")
        print(f"   클래스 수: {self.config.NUM_CLASSES}")
        print(f"   이미지 크기: {self.config.IMG_SIZE}")
        print(f"   배치 크기: {self.args.batch_size}")
        print(f"   학습률: {self.args.lr}")
        print(f"   에포크: {self.args.epochs}")
        
    def setup_model(self):
        """모델 생성 및 GPU 이동"""
        print("\n모델 초기화...")
        
        self.model = create_model(self.config.NUM_CLASSES).to(self.device)
        
        # 멀티 GPU 지원 (GPU가 여러 개인 경우)
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()}개 GPU 감지, DataParallel 사용")
            self.model = nn.DataParallel(self.model)
            
        # 모델 정보 출력
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   전체 파라미터: {total_params:,}")
        print(f"   훈련 가능 파라미터: {trainable_params:,}")
        
    def setup_loss_and_optimizer(self):
        """손실 함수 및 옵티마이저 설정"""
        # 손실 함수
        self.criterion = YOLOLoss(
            anchors=self.config.ANCHORS,
            num_classes=self.config.NUM_CLASSES,
            img_size=self.config.IMG_SIZE
        )
        
        # 옵티마이저 (AdamW 사용 - 더 안정적)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 학습률 스케줄러
        if self.args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
            
        print(f"옵티마이저: AdamW (lr={self.args.lr})")
        print(f"스케줄러: {self.args.scheduler or 'None'}")
        
    def setup_dataloaders(self):
        """데이터 로더 설정"""
        print(f"\n데이터 로딩: {self.args.data_dir}")
        
        # 훈련 데이터셋
        train_txt = Path(self.args.data_dir) / 'train.txt' if self.args.data_dir else 'train.txt'
        val_txt = Path(self.args.data_dir) / 'val.txt' if self.args.data_dir else 'val.txt'
        
        if not train_txt.exists():
            raise FileNotFoundError(f"훈련 파일을 찾을 수 없습니다: {train_txt}")
        if not val_txt.exists():
            raise FileNotFoundError(f"검증 파일을 찾을 수 없습니다: {val_txt}")
            
        try:
            # 데이터셋 생성            
            self.train_dataset = YOLODataset(
                img_paths_file=str(train_txt),
                img_size=self.config.IMG_SIZE,
                transform=True  # 데이터 증강 적용
            )
            
            self.val_dataset = YOLODataset(
                img_paths_file=str(val_txt),
                img_size=self.config.IMG_SIZE,
                transform=False  # 검증시에는 증강 안함
            )
            
            # 데이터로더 생성 (GPU 최적화)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,  # GPU 전송 최적화
                drop_last=True,
                persistent_workers=True if self.args.num_workers > 0 else False
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if self.args.num_workers > 0 else False
            )
            
            print(f"   훈련 이미지: {len(self.train_dataset)}")
            print(f"   검증 이미지: {len(self.val_dataset)}")
            print(f"   훈련 배치: {len(self.train_loader)}")
            print(f"   검증 배치: {len(self.val_loader)}")
            
        except Exception as e:
            print(f"데이터 로딩 실패: {e}")
            print("   prepare_data.py를 먼저 실행하여 데이터를 준비하세요.")
            sys.exit(1)
            
    def setup_checkpoints(self):
        """체크포인트 디렉토리 설정"""
        self.checkpoint_dir = Path(self.args.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 저장
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
            
        print(f"체크포인트 디렉토리: {self.checkpoint_dir}")
        
    def train_epoch(self, epoch):
        """한 에포크 훈련"""
        self.model.train()
        
        total_loss = 0
        total_batches = len(self.train_loader)
        
        # 진행률 표시
        print(f"Training Epoch {epoch}/{self.args.epochs}...")
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # GPU로 데이터 이동 (non_blocking으로 최적화)
            images = images.to(self.device, non_blocking=True)
            
            # targets는 리스트 형태로 올 수 있음
            if isinstance(targets, list):
                targets = [t.to(self.device, non_blocking=True) if t is not None else None for t in targets]
            else:
                targets = targets.to(self.device, non_blocking=True) if targets is not None else None
            
            # Forward pass
            outputs = self.model(images)
            
            # Loss 계산
            total_loss_value = 0
            for i, (output, anchors) in enumerate(zip(outputs, self.config.ANCHORS)):
                scale_loss, _ = self.criterion(output, targets, anchors)
                total_loss_value += scale_loss
                
            # 역방향 패스
            self.optimizer.zero_grad()
            total_loss_value.backward()
            
            # 그래디언트 클리핑 (그래디언트 폭발 방지)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Loss 기록
            current_loss = total_loss_value.item()
            total_loss += current_loss
            
            # Print progress every 5 batches
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx+1}/{total_batches}: loss={current_loss:.4f}, avg_loss={total_loss/(batch_idx+1):.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}")
            
        avg_loss = total_loss / total_batches
        return avg_loss
        
    @torch.no_grad()
    def validate_epoch(self, epoch):
        """검증 에포크"""
        self.model.eval()
        
        total_loss = 0
        total_batches = len(self.val_loader)
        
        print(f"Validating Epoch {epoch}/{self.args.epochs}...")
        
        for batch_idx, (images, targets) in enumerate(self.val_loader):
            # GPU로 데이터 이동
            images = images.to(self.device, non_blocking=True)
            
            if isinstance(targets, list):
                targets = [t.to(self.device, non_blocking=True) if t is not None else None for t in targets]
            else:
                targets = targets.to(self.device, non_blocking=True) if targets is not None else None
            
            # Forward pass
            outputs = self.model(images)
            
            # Loss 계산
            total_loss_value = 0
            for i, (output, anchors) in enumerate(zip(outputs, self.config.ANCHORS)):
                scale_loss, _ = self.criterion(output, targets, anchors)
                total_loss_value += scale_loss
                
            # Loss 기록
            current_loss = total_loss_value.item()
            total_loss += current_loss
            
            # Print progress every 5 batches
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx+1}/{total_batches}: loss={current_loss:.4f}, avg_loss={total_loss/(batch_idx+1):.4f}")
            
        avg_loss = total_loss / total_batches
        return avg_loss
        
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """체크포인트 저장"""
        # 모델 상태 (DataParallel 처리)
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': vars(self.args)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # 최신 체크포인트
        latest_path = self.checkpoint_dir / 'last.pt'
        torch.save(checkpoint, latest_path)
        
        # Best 모델
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"   Best 모델 저장: {best_path}")
            
        # 주기적 체크포인트
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
            
    def train(self):
        """전체 훈련 프로세스"""
        print("\n훈련 시작...")
        start_time = time.time()
        
        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            
            # 훈련
            train_loss = self.train_epoch(epoch)
            
            # 검증
            val_loss = self.validate_epoch(epoch)
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
                
            # 메트릭 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 최고 모델 확인
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            # 체크포인트 저장
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            # 결과 출력
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch:3d}/{self.args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")
                  
            if is_best:
                print(f"   New best validation loss!")
                
        # 훈련 완료
        total_time = time.time() - start_time
        print("\n훈련 완료!")
        print(f"   총 시간: {total_time/3600:.1f}시간")
        print(f"   최고 검증 손실: {self.best_val_loss:.4f}")
        print(f"   저장 위치: {self.checkpoint_dir}")

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='YOLOv3 Production Training')
    
    # 데이터 관련
    parser.add_argument('--data-dir', type=str, required=True,
                       help='데이터셋 디렉토리 (train.txt, val.txt 포함)')
    parser.add_argument('--num-classes', type=int, default=25,
                       help='클래스 개수')
    parser.add_argument('--img-size', type=int, default=416,
                       help='입력 이미지 크기')
    
    # 훈련 관련
    parser.add_argument('--epochs', type=int, default=100,
                       help='훈련 에포크 수')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='배치 크기 (GPU 메모리에 따라 조정)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='학습률')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='가중치 감쇠')
    
    # 스케줄러
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='학습률 스케줄러')
    
    # 데이터 로딩
    parser.add_argument('--num-workers', type=int, default=4,
                       help='데이터 로더 워커 수')
    
    # 저장
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='체크포인트 저장 디렉토리')
    
    return parser.parse_args()

if __name__ == '__main__':
    # 인자 파싱
    args = parse_args()
    
    # 훈련 시작
    trainer = ProductionTrainer(args)
    trainer.train()