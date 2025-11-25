"""
YOLOv3 Prediction Decoding and Post-Processing
Critical missing functions to convert raw model outputs to actual bounding boxes
"""

import torch
import torch.nn.functional as F
import numpy as np
from configs.config import Config

def decode_predictions(prediction, anchors, num_classes, conf_threshold=0.5):
    """
    Convert raw YOLO predictions to bounding boxes
    
    Args:
        prediction: [batch_size, num_anchors*(5+num_classes), grid_h, grid_w]
        anchors: List of (w, h) tuples for this scale
        num_classes: Number of object classes
        conf_threshold: Minimum confidence threshold
    
    Returns:
        List of detections: [(x, y, w, h, conf, class_pred), ...]
    """
    device = prediction.device
    batch_size = prediction.size(0)
    grid_size = prediction.size(2)
    
    # Reshape prediction: [batch, anchors, grid, grid, (5 + num_classes)]
    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes
    prediction = prediction.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
    
    # Get outputs
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]                 # Width
    h = prediction[..., 3]                 # Height
    conf = torch.sigmoid(prediction[..., 4])  # Confidence
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Class probabilities
    
    # Calculate offsets for each grid
    grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
    grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
    
    # Anchor dimensions
    anchor_w = torch.tensor([a[0] for a in anchors], device=device).view((1, num_anchors, 1, 1))
    anchor_h = torch.tensor([a[1] for a in anchors], device=device).view((1, num_anchors, 1, 1))
    
    # Add offset and scale with anchors
    # Convert from grid coordinates to image coordinates
    pred_boxes = torch.zeros_like(prediction[..., :4])
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h
    
    # Scale to image size
    stride = Config.IMG_SIZE // grid_size
    pred_boxes[..., :4] *= stride
    
    # Filter by confidence
    conf_mask = conf >= conf_threshold
    
    detections = []
    for batch_idx in range(batch_size):
        batch_detections = []
        
        for anchor_idx in range(num_anchors):
            for grid_x_idx in range(grid_size):
                for grid_y_idx in range(grid_size):
                    if conf_mask[batch_idx, anchor_idx, grid_x_idx, grid_y_idx]:
                        # Get detection data
                        box = pred_boxes[batch_idx, anchor_idx, grid_x_idx, grid_y_idx]
                        confidence = conf[batch_idx, anchor_idx, grid_x_idx, grid_y_idx]
                        class_probs = pred_cls[batch_idx, anchor_idx, grid_x_idx, grid_y_idx]
                        
                        # Get best class
                        class_conf, class_pred = torch.max(class_probs, 0)
                        final_conf = confidence * class_conf
                        
                        if final_conf >= conf_threshold:
                            # Convert to [x1, y1, x2, y2] format for NMS
                            x_center, y_center, width, height = box
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            detection = torch.tensor([x1, y1, x2, y2, final_conf, class_pred.float()])
                            batch_detections.append(detection)
        
        if batch_detections:
            batch_detections = torch.stack(batch_detections)
            detections.append(batch_detections)
        else:
            detections.append(torch.empty(0, 6))
    
    return detections

def apply_nms(detections, nms_threshold=0.4):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        detections: [N, 6] tensor of [x1, y1, x2, y2, conf, class]
        nms_threshold: IoU threshold for NMS
    
    Returns:
        Filtered detections after NMS
    """
    if detections.size(0) == 0:
        return detections
    
    # Sort by confidence
    sort_idx = torch.argsort(detections[:, 4], descending=True)
    detections = detections[sort_idx]
    
    keep = []
    while detections.size(0) > 0:
        # Keep the detection with highest confidence
        keep.append(detections[0])
        
        if detections.size(0) == 1:
            break
        
        # Calculate IoU with remaining detections
        iou = calculate_iou(detections[0:1, :4], detections[1:, :4])
        
        # Keep detections with IoU below threshold
        mask = iou < nms_threshold
        detections = detections[1:][mask.squeeze()]
    
    if keep:
        return torch.stack(keep)
    else:
        return torch.empty(0, 6)

def calculate_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) between boxes
    
    Args:
        boxes1: [N, 4] tensor of [x1, y1, x2, y2]
        boxes2: [M, 4] tensor of [x1, y1, x2, y2]
    
    Returns:
        [N, M] tensor of IoU values
    """
    # Get intersection coordinates
    x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1]) 
    x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])
    
    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate areas of both boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate union
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-16)
    
    return iou

def postprocess_detections(predictions, anchors_list, num_classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Complete post-processing pipeline for YOLO predictions
    
    Args:
        predictions: List of [batch_size, num_anchors*(5+num_classes), grid_h, grid_w] for each scale
        anchors_list: List of anchor sets for each scale
        num_classes: Number of object classes
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
    
    Returns:
        List of final detections for each image in batch
    """
    all_detections = []
    batch_size = predictions[0].size(0)
    
    # Process each image in the batch
    for batch_idx in range(batch_size):
        image_detections = []
        
        # Process each scale
        for pred, anchors in zip(predictions, anchors_list):
            # Get detections for this scale
            scale_detections = decode_predictions(
                pred[batch_idx:batch_idx+1], 
                anchors, 
                num_classes, 
                conf_threshold
            )[0]  # Get first (and only) item from batch
            
            if scale_detections.size(0) > 0:
                image_detections.append(scale_detections)
        
        # Combine all scales
        if image_detections:
            combined_detections = torch.cat(image_detections, dim=0)
            
            # Apply NMS across all scales
            final_detections = apply_nms(combined_detections, nms_threshold)
            all_detections.append(final_detections)
        else:
            all_detections.append(torch.empty(0, 6))
    
    return all_detections