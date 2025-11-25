"""
데이터 준비 스크립트 - 실제 데이터셋용
Real Dataset Preparation Script 
"""
import os
import random
import argparse
from pathlib import Path
# from tqdm import tqdm  # 현재 주석 처리
from PIL import Image

def validate_yolo_label(label_path, num_classes=25):
    """
    YOLO 라벨 파일 검증
    
    Args:
        label_path: 라벨 파일 경로
        num_classes: 클래스 개수
        
    Returns:
        bool: 유효한 라벨 파일인지 여부
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:  # 빈 줄 무시
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"Invalid format in {label_path}: {line}")
                return False
                
            class_id, x, y, w, h = parts
            
            # 클래스 ID 검증
            class_id = int(class_id)
            if class_id < 0 or class_id >= num_classes:
                print(f"Invalid class_id {class_id} in {label_path}")
                return False
                
            # 좌표 검증 (0~1 범위)
            x, y, w, h = float(x), float(y), float(w), float(h)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"Invalid coordinates in {label_path}: {line}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return False

def validate_image(image_path):
    """
    이미지 파일 검증
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        bool: 유효한 이미지 파일인지 여부
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # 이미지 파일 검증
        return True
    except Exception as e:
        print(f"Invalid image {image_path}: {e}")
        return False

def scan_dataset(data_dir, num_classes=25, extensions=('.jpg', '.jpeg', '.png')):
    """
    데이터셋 스캔 및 검증
    
    Args:
        data_dir: 데이터셋 디렉토리
        num_classes: 클래스 개수
        extensions: 지원하는 이미지 확장자
        
    Returns:
        valid_pairs: 유효한 (이미지, 라벨) 경로 쌍 리스트
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
    print(f"Dataset directory: {data_dir}")
    print(f"   📷 Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    
    # 이미지 파일 찾기
    image_files = []
    for ext in extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    print(f"\nFound {len(image_files)} image files")
    
    # 이미지-라벨 쌍 검증
    valid_pairs = []
    invalid_images = []
    missing_labels = []
    invalid_labels = []
    
    print("🔍 Validating dataset...")
    for i, img_path in enumerate(image_files):
        if i % 5 == 0:  # 5개 파일마다 진행률 출력
            print(f"Validating files... {i+1}/{len(image_files)}")
        # 대응하는 라벨 파일 찾기
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # 이미지 검증
        if not validate_image(img_path):
            invalid_images.append(img_path.name)
            continue
            
        # 라벨 파일 존재 여부 확인
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue
            
        # 라벨 파일 검증
        if not validate_yolo_label(label_path, num_classes):
            invalid_labels.append(label_path.name)
            continue
            
        valid_pairs.append((img_path, label_path))
    
    # 검증 결과 출력
    print(f"\nValidation Results:")
    print(f"   Valid pairs: {len(valid_pairs)}")
    print(f"   Invalid images: {len(invalid_images)}")
    print(f"   Missing labels: {len(missing_labels)}")
    print(f"   Invalid labels: {len(invalid_labels)}")
    
    if invalid_images and len(invalid_images) <= 5:
        print(f"   Invalid images: {', '.join(invalid_images)}")
    if missing_labels and len(missing_labels) <= 5:
        print(f"   Missing labels: {', '.join(missing_labels)}")
    if invalid_labels and len(invalid_labels) <= 5:
        print(f"   Invalid labels: {', '.join(invalid_labels)}")
    
    return valid_pairs

def split_dataset(pairs, train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    데이터셋을 train/val로 분할
    
    Args:
        pairs: (이미지, 라벨) 쌍 리스트
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
        
    Returns:
        train_pairs, val_pairs: 분할된 데이터
    """
    if train_ratio + val_ratio != 1.0:
        raise ValueError(f"train_ratio + val_ratio must equal 1.0, got {train_ratio + val_ratio}")
    
    random.seed(seed)
    random.shuffle(pairs)
    
    total = len(pairs)
    train_size = int(total * train_ratio)
    
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    
    print(f"\nDataset Split:")
    print(f"   🚂 Train: {len(train_pairs)} images ({len(train_pairs)/total*100:.1f}%)")
    print(f"   🔬 Val:   {len(val_pairs)} images ({len(val_pairs)/total*100:.1f}%)")
    
    return train_pairs, val_pairs

def save_splits(train_pairs, val_pairs, output_dir):
    """
    분할된 데이터를 train.txt, val.txt로 저장
    
    Args:
        train_pairs: 훈련 데이터 쌍
        val_pairs: 검증 데이터 쌍
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # train.txt 저장
    train_file = output_dir / 'train.txt'
    with open(train_file, 'w') as f:
        for img_path, label_path in train_pairs:
            f.write(f"{img_path}\n")
    
    # val.txt 저장
    val_file = output_dir / 'val.txt'
    with open(val_file, 'w') as f:
        for img_path, label_path in val_pairs:
            f.write(f"{img_path}\n")
    
    print(f"\n💾 Files created:")
    print(f"   📄 {train_file}")
    print(f"   📄 {val_file}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='YOLOv3 데이터 준비 스크립트')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='데이터셋 디렉토리 경로 (images/, labels/ 포함)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='train.txt, val.txt 출력 디렉토리')
    parser.add_argument('--num-classes', type=int, default=25,
                       help='클래스 개수 (기본값: 25)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='훈련 데이터 비율 (기본값: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 YOLOv3 데이터 준비 시작")
    print("="*60)
    print(f"📁 데이터 디렉토리: {args.data_dir}")
    print(f"📊 클래스 개수: {args.num_classes}")
    print(f"📈 Train/Val 비율: {args.train_ratio:.1f}/{1-args.train_ratio:.1f}")
    print(f"🎲 랜덤 시드: {args.seed}")
    
    try:
        # 1. 데이터셋 스캔 및 검증
        valid_pairs = scan_dataset(args.data_dir, args.num_classes)
        
        if len(valid_pairs) == 0:
            print("❌ 유효한 데이터가 없습니다!")
            return
        
        # 2. Train/Val 분할
        train_pairs, val_pairs = split_dataset(
            valid_pairs, 
            train_ratio=args.train_ratio,
            val_ratio=1-args.train_ratio,
            seed=args.seed
        )
        
        # 3. 분할 결과 저장
        save_splits(train_pairs, val_pairs, args.output_dir)
        
        print("\n" + "="*60)
        print("✅ 데이터 준비 완료!")
        print("="*60)
        print(f"🚀 훈련을 시작하려면:")
        print(f"   python train_production.py --data-dir {args.data_dir}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())