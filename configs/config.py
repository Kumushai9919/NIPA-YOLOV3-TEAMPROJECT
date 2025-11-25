# 어린이 보호구역 탐지를 위한 YOLOv3 설정
import torch

class Config:
    # 모델 설정
    MODEL_NAME = "YOLOv3_Children_Protection"
    
    # 실제 객체 탐지를 위한 25개 클래스
    NUM_CLASSES = 25  # 실제 클래스 수
    INPUT_SIZE = 416  # 표준 YOLO 입력 크기
    IMG_SIZE = 416    # 데이터셋 처리를 위한 이미지 크기
    
    # 25개 클래스 (AI Hub 데이터 기준)
    CLASSES = {
        0: "child", 1: "adult", 2: "stroller", 3: "bicycle", 4: "pm",
        5: "umbrella", 6: "bus", 7: "truck", 8: "car", 9: "motorcycle",
        10: "etc_vehicle", 11: "school_car_s", 12: "school_car_l", 
        13: "bollard", 14: "ped_signal", 15: "car_signal", 16: "fence",
        17: "sign", 18: "sidewalk", 19: "crosswalk", 20: "speed_bump",
        21: "road", 22: "lane", 23: "stop_line", 24: "center_line"
    }
    
    CLASS_NAMES = [
        "child", "adult", "stroller", "bicycle", "pm", "umbrella",
        "bus", "truck", "car", "motorcycle", "etc_vehicle", 
        "school_car_s", "school_car_l", "bollard", "ped_signal",
        "car_signal", "fence", "sign", "sidewalk", "crosswalk",
        "speed_bump", "road", "lane", "stop_line", "center_line"
    ]
    
    # 훈련 설정
    BATCH_SIZE = 8  # M1 Pro용 작은 배치 크기
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    EPOCHS = 5  # 10개 샘플용 데모 에폭
    WEIGHT_DECAY = 5e-4
    
    # 데이터 설정
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.05
    
    # 장치 설정
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # YOLOv3의 3개 스케일용 앵커
    ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],      # 작은 객체
        [(30, 61), (62, 45), (59, 119)],    # 중간 객체  
        [(116, 90), (156, 198), (373, 326)] # 큰 객체
    ]
    
    # 탐지 임계값 (참조 프로젝트로부터)
    CONF_THRESHOLD = 0.5    # 탐지를 위한 최소 신뢰도
    NMS_THRESHOLD = 0.4     # 비최대 억제 임계값
    
    # 경로 설정
    DATA_DIR = "demo_detection_data"  # 10개 이미지 데모 데이터 사용
    
    # 데이터 증강
    AUGMENT = True
    MOSAIC_PROB = 0.5