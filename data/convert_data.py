#!/usr/bin/env python3
"""
AI Hub JSON to YOLO converter for updated annotation format
Converts AI Hub JSON format to YOLO format
"""
import json
import os
from pathlib import Path

# AI Hub 클래스를 YOLO 클래스 ID로 매핑
CLASS_MAPPING = {
    'person': 0,        # person -> child/adult (단순화)
    'road_etc': 13,     # road_etc -> bollard (예제 매핑)
    'vehicle': 8,       # vehicle -> car (예제 매핑)
    'sign': 17,         # sign -> sign
    'traffic_light': 14 # traffic_light -> ped_signal
}

def convert_json_to_yolo(data_dir="data"):
    """Convert AI Hub JSON annotations to YOLO format"""
    images_dir = Path(data_dir) / "images"
    annotations_dir = Path(data_dir) / "annotations"
    labels_dir = Path(data_dir) / "labels"
    
    # labels 디렉토리 생성
    labels_dir.mkdir(exist_ok=True)
    
    print(f"🔄 Converting AI Hub JSON → YOLO format...")
    print(f"   Images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"   JSON files: {len(list(annotations_dir.glob('*.json')))}")
    
    converted = 0
    total_objects = 0
    
    for json_file in annotations_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 정보 가져오기
            filename = data['file']['filename']
            img_width = data['info']['width']
            img_height = data['info']['height']
            
            # 해당 .txt 파일 생성
            txt_file = labels_dir / f"{json_file.stem}.txt"
            
            with open(txt_file, 'w') as f:
                # 어노테이션 처리
                if 'annotations' in data and data['annotations']:
                    for ann in data['annotations']:
                        if 'bbox' in ann and 'object super class' in ann:
                            # bbox 좌표 가져오기 [x, y, w, h]
                            x, y, w, h = ann['bbox']
                            
                            # YOLO 형식으로 변환 (정규화된 중심 좌표)
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            norm_w = w / img_width
                            norm_h = h / img_height
                            
                            # 클래스를 YOLO 클래스 ID로 매핑
                            class_name = ann['object super class']
                            class_id = CLASS_MAPPING.get(class_name, 0)  # 기본값 0 (person)
                            
                            # YOLO 형식 라인 작성
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                            total_objects += 1
            
            converted += 1
            
        except Exception as e:
            print(f"   ⚠️  Error converting {json_file.name}: {e}")
    
    print(f"✅ Converted {converted} files to YOLO format")
    print(f"📦 Total objects: {total_objects}")
    return converted > 0

if __name__ == "__main__":
    success = convert_json_to_yolo()
    if success:
        print("\n🚀 Now run: python3 data/prepare_data.py --data-dir data/")
    else:
        print("\n❌ Conversion failed")