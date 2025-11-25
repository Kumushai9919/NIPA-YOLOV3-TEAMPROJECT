"""
Visualization utilities for YOLOv3 children protection project
Shows bounding boxes on images with class labels and confidence scores
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from typing import List, Tuple, Optional
import json

class BBoxVisualizer:
    """Visualize bounding boxes on images for YOLO training/testing"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize visualizer
        Args:
            class_names: List of class names (default: ['person', 'road_etc'])
        """
        self.class_names = class_names or ['person', 'road_etc']
        self.colors = [
            (255, 0, 0),    # Red for person
            (0, 255, 0),    # Green for road_etc
            (0, 0, 255),    # Blue for additional classes
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
    
    def draw_bbox_cv2(self, image: np.ndarray, bboxes: List, labels: List, 
                      confidences: List = None, save_path: str = None) -> np.ndarray:
        """
        Draw bounding boxes using OpenCV
        Args:
            image: Input image as numpy array (BGR format)
            bboxes: List of bounding boxes [(x1,y1,x2,y2), ...]
            labels: List of class labels [0, 1, ...]
            confidences: List of confidence scores [0.9, 0.8, ...]
            save_path: If provided, save image to this path
        Returns:
            Image with drawn bounding boxes
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Convert normalized coords to pixel coords if needed
            if all(coord <= 1.0 for coord in bbox):
                x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
            else:
                x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this class
            color = self.colors[label % len(self.colors)]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            class_name = self.class_names[label] if label < len(self.class_names) else f'class_{label}'
            if confidences and i < len(confidences):
                label_text = f'{class_name}: {confidences[i]:.2f}'
            else:
                label_text = class_name
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, img)
        
        return img
    
    def draw_bbox_matplotlib(self, image: np.ndarray, bboxes: List, labels: List, 
                            confidences: List = None, figsize: Tuple = (12, 8), 
                            save_path: str = None, show: bool = True):
        """
        Draw bounding boxes using matplotlib (better for Jupyter)
        Args:
            image: Input image as numpy array (RGB format)
            bboxes: List of bounding boxes [(x1,y1,x2,y2), ...]
            labels: List of class labels [0, 1, ...]
            confidences: List of confidence scores [0.9, 0.8, ...]
            figsize: Figure size for matplotlib
            save_path: If provided, save image to this path
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (OpenCV format)
            if np.mean(image[:,:,2]) < np.mean(image[:,:,0]):  # More blue than red suggests BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(image)
        h, w = image.shape[:2]
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Convert normalized coords to pixel coords if needed
            if all(coord <= 1.0 for coord in bbox):
                x1, y1, x2, y2 = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
            else:
                x1, y1, x2, y2 = bbox
            
            width, height = x2 - x1, y2 - y1
            
            # Get color for this class
            color_rgb = tuple(c/255.0 for c in self.colors[label % len(self.colors)])
            
            # Create rectangle patch
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color_rgb, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            class_name = self.class_names[label] if label < len(self.class_names) else f'class_{label}'
            if confidences and i < len(confidences):
                label_text = f'{class_name}: {confidences[i]:.2f}'
            else:
                label_text = class_name
            
            ax.text(x1, y1 - 10, label_text, 
                   bbox=dict(facecolor=color_rgb, alpha=0.8), 
                   fontsize=10, color='white')
        
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Flip y-axis for image coordinates
        ax.axis('off')
        ax.set_title('YOLO Detection Results')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_ground_truth(self, image_path: str, json_path: str, 
                              save_path: str = None, method: str = 'matplotlib'):
        """
        Visualize ground truth annotations from AI Hub JSON format
        Args:
            image_path: Path to image file
            json_path: Path to corresponding JSON annotation file
            save_path: If provided, save visualization
            method: 'matplotlib' or 'cv2'
        """
        # Load image
        if method == 'matplotlib':
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path)
        
        # Load annotations
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bboxes = []
        labels = []
        
        # Parse AI Hub format
        for annotation in data.get('annotations', []):
            bbox = annotation.get('bbox', [])
            if len(bbox) == 4:
                # bbox format: [x, y, width, height] -> convert to [x1, y1, x2, y2]
                x, y, w, h = bbox
                bboxes.append([x, y, x + w, y + h])
                
                # Map category to our classes
                category = annotation.get('category', '')
                if 'person' in category.lower():
                    labels.append(0)  # person
                else:
                    labels.append(1)  # road_etc
        
        # Visualize
        if method == 'matplotlib':
            self.draw_bbox_matplotlib(image, bboxes, labels, save_path=save_path)
        else:
            result = self.draw_bbox_cv2(image, bboxes, labels, save_path=save_path)
            return result
    
    def visualize_predictions(self, image_path: str, predictions: dict, 
                             save_path: str = None, method: str = 'matplotlib'):
        """
        Visualize model predictions
        Args:
            image_path: Path to image file
            predictions: Dict with 'boxes', 'labels', 'scores'
            save_path: If provided, save visualization
            method: 'matplotlib' or 'cv2'
        """
        # Load image
        if method == 'matplotlib':
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path)
        
        bboxes = predictions.get('boxes', [])
        labels = predictions.get('labels', [])
        scores = predictions.get('scores', [])
        
        # Visualize
        if method == 'matplotlib':
            self.draw_bbox_matplotlib(image, bboxes, labels, confidences=scores, save_path=save_path)
        else:
            result = self.draw_bbox_cv2(image, bboxes, labels, confidences=scores, save_path=save_path)
            return result
    
    def create_visualization_grid(self, image_paths: List[str], json_paths: List[str], 
                                 save_path: str = None, max_images: int = 6):
        """
        Create a grid visualization of multiple images with ground truth
        Args:
            image_paths: List of image file paths
            json_paths: List of corresponding JSON annotation paths
            save_path: If provided, save the grid
            max_images: Maximum number of images to show in grid
        """
        n_images = min(len(image_paths), max_images)
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(n_images):
            # Load image
            image = cv2.imread(image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load annotations
            with open(json_paths[i], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            bboxes = []
            labels = []
            
            # Parse annotations
            for annotation in data.get('annotations', []):
                bbox = annotation.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    bboxes.append([x, y, x + w, y + h])
                    
                    category = annotation.get('category', '')
                    if 'person' in category.lower():
                        labels.append(0)
                    else:
                        labels.append(1)
            
            # Plot on subplot
            axes[i].imshow(image)
            h, w = image.shape[:2]
            
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                color = tuple(c/255.0 for c in self.colors[label % len(self.colors)])
                
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                axes[i].add_patch(rect)
                
                class_name = self.class_names[label]
                axes[i].text(x1, y1 - 10, class_name, 
                           bbox=dict(facecolor=color, alpha=0.8), 
                           fontsize=8, color='white')
            
            axes[i].set_xlim(0, w)
            axes[i].set_ylim(h, 0)
            axes[i].axis('off')
            axes[i].set_title(f'Image {i+1}', fontsize=10)
        
        # Hide empty subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()

# Example usage functions
def demo_visualization():
    """Demo function showing how to use the visualizer"""
    visualizer = BBoxVisualizer(['person', 'road_etc'])
    
    print("BBox Visualizer initialized!")
    print("Available methods:")
    print("1. visualize_ground_truth() - Show ground truth annotations")
    print("2. visualize_predictions() - Show model predictions") 
    print("3. create_visualization_grid() - Show multiple images in grid")
    print("4. draw_bbox_matplotlib() - Direct matplotlib drawing")
    print("5. draw_bbox_cv2() - Direct OpenCV drawing")
    
    return visualizer

if __name__ == "__main__":
    demo_visualization()