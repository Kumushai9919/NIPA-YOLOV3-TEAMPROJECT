# YOLOv3 Children Protection Zone Detection
## NIPA AI BOOTCAMP

> **어린이 보호구역 위험 탐지를 위한 YOLOv3 구현**  
> Production-ready YOLOv3 implementation for children protection zone risk detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 프로젝트 개요

어린이 보호구역에서의 위험 상황을 실시간으로 탐지하는 YOLOv3 객체 검출 시스템입니다. 논문 기반 완전 구현으로 교육적 가치와 실용성을 모두 갖추었습니다.
 

## 🚀 Get started

### 1. 설치
```bash
git clone https://github.com/Kumushai9919/NIPA-YOLOV3-TEAMPROJECT.git
cd NIPA-YOLOV3-TEAMPROJECT
pip install -r requirements.txt
```

### 2. 데이터 준비 (AI Hub JSON → YOLO 변환)
```bash
# 1) data/images/에 JPG 파일, data/annotations/에 JSON 파일 배치
# 2) JSON을 YOLO 형식으로 변환
python3 data/convert_data.py

# 3) 훈련/검증 분할 생성
python3 data/prepare_data.py --data-dir data/
```

### 3. 훈련 시작
```bash
cd training/
python3 train.py --data-dir ../data --epochs 50 --batch-size 8
```

## 📁 프로젝트 구조

```
yolo_children_protection/
├── .gitignore                    # Git 무시 목록
├── README.md                     # 프로젝트 문서  
├── requirements.txt              # 필요 패키지 
│
├── 📁 data/                      # 데이터 관리 (샘플 10장 테스트 데이터)
│   ├── images/                   # 훈련 이미지 (JPG) - 10장 샘플
│   ├── annotations/              # AI Hub JSON 파일 - 10장 샘플  
│   ├── labels/                   # YOLO 형식 라벨 (TXT) - 변환된 라벨
│   ├── prepare_data.py           # 데이터 준비 스크립트  
│   └── convert_data.py           # JSON → YOLO 변환기
│   
│   ⚠️  실제 훈련용: AI Hub (sample-600개) 데이터셋을 다운로드하세요
│   📥 https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=189
│
├── 📁 models/                    # 모델 아키텍처
│   ├── backbone.py               # Darknet-53 백본
│   ├── neck.py                   # Feature Pyramid Network
│   └── model.py                  # 완전한 YOLOv3 모델
│
├── 📁 training/                  # 훈련 파이프라인
│   ├── train.py                  # 메인 훈련 스크립트
│   ├── dataset.py                # 데이터셋 처리
│   ├── yolo_dataset.py           # YOLO 데이터셋 로더
│   ├── loss.py                   # YOLO 손실 함수
│   ├── postprocess.py            # NMS 후처리
│   ├── visualization.py          # 결과 시각화
│   └── checkpoints/              # 훈련 체크포인트 (자동 생성)
│       ├── best.pt               # 최고 성능 모델 (1GB)
│       ├── last.pt               # 최신 체크포인트
│       └── config.json           # 훈련 설정 백업
│
└── 📁 configs/                   # 설정 파일
    └── config.py                 # 모델/훈련 설정 (한국어 주석)
```

## ⚙️ 훈련 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data-dir` | 필수 | 데이터셋 디렉토리 경로 |
| `--epochs` | 100 | 훈련 에포크 수 |
| `--batch-size` | 16 | 배치 크기 (GPU 메모리에 따라 조정) |
| `--lr` | 1e-4 | 학습률 |
| `--num-classes` | 25 | 클래스 개수 |
| `--img-size` | 416 | 입력 이미지 크기 |
| `--scheduler` | cosine | 학습률 스케줄러 (cosine/step/none) |
| `--num-workers` | 4 | 데이터 로더 워커 수 |

전체 옵션 확인:
```bash
python training/train.py --help
```
 
### 훈련 재개
```bash
python training/train.py \
    --data-dir data \
    --resume training/checkpoints/last.pt
```

 
