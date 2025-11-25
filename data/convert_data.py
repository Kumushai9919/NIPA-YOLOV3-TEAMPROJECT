#!/usr/bin/env python3
"""
AI Hub JSON to YOLO converter for updated annotation format
Converts AI Hub JSON format to YOLO format
"""
import json
import os
from pathlib import Path

# AI Hub class mapping to YOLO class IDs
CLASS_MAPPING = {
    'person': 0,        # person -> child/adult (simplified)
    'road_etc': 13,     # road_etc -> bollard (example mapping)
    'vehicle': 8,       # vehicle -> car (example mapping)
    'sign': 17,         # sign -> sign
    'traffic_light': 14 # traffic_light -> ped_signal
}

def convert_json_to_yolo(data_dir="data"):
    """Convert AI Hub JSON annotations to YOLO format"""
    images_dir = Path(data_dir) / "images"
    annotations_dir = Path(data_dir) / "annotations"
    labels_dir = Path(data_dir) / "labels"
    
    # Create labels directory
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
            
            # Get image info
            filename = data['file']['filename']
            img_width = data['info']['width']
            img_height = data['info']['height']
            
            # Create corresponding .txt file
            txt_file = labels_dir / f"{json_file.stem}.txt"
            
            with open(txt_file, 'w') as f:
                # Process annotations
                if 'annotations' in data and data['annotations']:
                    for ann in data['annotations']:
                        if 'bbox' in ann and 'object super class' in ann:
                            # Get bbox coordinates [x, y, w, h]
                            x, y, w, h = ann['bbox']
                            
                            # Convert to YOLO format (normalized center coordinates)
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            norm_w = w / img_width
                            norm_h = h / img_height
                            
                            # Map class to YOLO class ID
                            class_name = ann['object super class']
                            class_id = CLASS_MAPPING.get(class_name, 0)  # Default to 0 (person)
                            
                            # Write YOLO format line
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