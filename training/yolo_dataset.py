"""
Simple YOLO Dataset for train.txt/val.txt file lists
"""
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class YOLODataset(Dataset):
    """Simple YOLO format dataset that works with train.txt/val.txt"""
    
    def __init__(self, img_paths_file, img_size=416, transform=False):
        """
        Args:
            img_paths_file: Path to train.txt or val.txt
            img_size: Target image size
            transform: Apply data augmentation
        """
        self.img_size = img_size
        self.transform = transform
        
        # Load image paths
        with open(img_paths_file, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.img_paths)} image paths from {img_paths_file}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        
        # Fix path for different working directories
        if not img_path.startswith('/') and not img_path.startswith('../'):
            if img_path.startswith('data/'):
                img_path = '../' + img_path  # Add ../ prefix when running from training/
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Return dummy data with proper normalization
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(1, 5)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Load corresponding label
        label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if lines:
                # Parse YOLO format: class x_center y_center width height
                targets = []
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            targets.append([class_id, x_center, y_center, width, height])
                
                if targets:
                    targets = torch.tensor(targets, dtype=torch.float32)
                else:
                    targets = torch.zeros(1, 5)
            else:
                targets = torch.zeros(1, 5)
                
        except FileNotFoundError:
            print(f"Warning: Label file not found for {img_path}")
            targets = torch.zeros(1, 5)
        
        return image, targets