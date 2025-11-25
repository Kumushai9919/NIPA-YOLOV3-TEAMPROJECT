import json
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from configs.config import Config

class ChildrenProtectionDataset(Dataset):
    """Dataset for Children Protection Zone Detection with AI Hub format"""
    
    def __init__(self, images_dir, annotations_dir, transform=None, is_train=True, max_samples=None):
        """
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing JSON annotations
            transform: Image transformations
            is_train: Training or validation mode
            max_samples: Maximum number of samples to load (for demo)
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.is_train = is_train
        self.config = Config()
        
        # Load all annotations
        self.data = self.load_ai_hub_data(max_samples)
        
        print(f"Dataset loaded: {len(self.data)} samples")
        
    def load_ai_hub_data(self, max_samples=None):
        """Load and process AI Hub JSON annotations"""
        data = []
        
        # Get all JSON files
        json_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.json')]
        json_files = sorted(json_files)
        
        if max_samples:
            json_files = json_files[:max_samples]
        
        for json_file in json_files:
            # Find corresponding image
            image_name = os.path.splitext(json_file)[0] + '.jpg'
            image_path = os.path.join(self.images_dir, image_name)
            json_path = os.path.join(self.annotations_dir, json_file)
            
            # Check if both files exist
            if os.path.exists(image_path) and os.path.exists(json_path):
                try:
                    annotation_data = self.parse_ai_hub_json(json_path, image_name)
                    if annotation_data and len(annotation_data['boxes']) > 0:  # Only keep images with annotations
                        data.append(annotation_data)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
        
        print(f"Successfully loaded {len(data)} valid samples")
        return data
    
    def parse_ai_hub_json(self, json_path, image_name):
        """Parse AI Hub JSON format"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get image to determine dimensions
        image_path = os.path.join(self.images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        height, width = image.shape[:2]
        
        # Process annotations
        boxes = []
        labels = []
        
        for ann in data.get('annotations', []):
            # Map classes to our simplified version
            object_class = ann.get('object super class', '')
            if 'person' in object_class.lower():
                label = 0  # person
            else:
                label = 1  # road_etc (everything else)
            
            # Handle bbox annotations
            if 'bbox' in ann:
                bbox = ann['bbox']  # [x, y, w, h]
                x, y, w, h = bbox
                
                # Skip invalid boxes
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    continue
                
                # Convert to [x1, y1, x2, y2] format
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                # Skip boxes that are too small after clamping
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                
                # Normalize coordinates to [0, 1]
                x1_norm = x1 / width
                y1_norm = y1 / height
                x2_norm = x2 / width
                y2_norm = y2 / height
                
                boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                labels.append(label)
        
        return {
            'image_name': image_name,
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'height': height,
            'width': width
        }
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, sample['image_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model input size
        target_size = (self.config.IMG_SIZE, self.config.IMG_SIZE)
        image = cv2.resize(image, target_size)
        
        # Convert to tensor and normalize
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Get bounding boxes and labels (already normalized)
        boxes = torch.FloatTensor(sample['boxes'])
        labels = torch.LongTensor(sample['labels'])
        
        # For YOLO training, we need to convert boxes to proper format
        # This is a simplified version - you may need to adjust for your YOLO implementation
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_name': sample['image_name']
        }
    
    def get_sample_info(self):
        """Get information about the dataset"""
        total_boxes = sum(len(sample['boxes']) for sample in self.data)
        
        # Count classes
        person_count = 0
        road_count = 0
        
        for sample in self.data:
            for label in sample['labels']:
                if label == 0:
                    person_count += 1
                else:
                    road_count += 1
        
        return {
            'total_samples': len(self.data),
            'total_boxes': total_boxes,
            'person_objects': person_count,
            'road_objects': road_count
        }

def create_demo_dataloader(demo_data_path, batch_size=2, max_samples=10):
    """Create a dataloader for demo training"""
    images_dir = os.path.join(demo_data_path, 'images')
    annotations_dir = os.path.join(demo_data_path, 'annotations')
    
    # Create dataset
    dataset = ChildrenProtectionDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        max_samples=max_samples
    )
    
    # Print dataset info
    info = dataset.get_sample_info()
    print(f"\nDemo Dataset Info:")
    print(f"   📸 Samples: {info['total_samples']}")
    print(f"   📦 Total boxes: {info['total_boxes']}")
    print(f"   👥 Person objects: {info['person_objects']}")
    print(f"   Road objects: {info['road_objects']}")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function for YOLO data"""
        images = torch.stack([item['image'] for item in batch])
        
        # For simplicity, we'll return boxes and labels as lists
        # In a full implementation, you'd convert these to YOLO target format
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        names = [item['image_name'] for item in batch]
        
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'image_names': names
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    return dataloader, dataset