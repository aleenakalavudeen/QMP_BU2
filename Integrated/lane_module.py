"""
Lane Detection Module (YOLOPv2)

This module wraps the YOLOPv2 model for lane detection, drivable area segmentation,
and traffic object detection.

The model performs three tasks:
1. Traffic object detection (vehicles, pedestrians, etc.)
2. Drivable area segmentation
3. Lane line segmentation

Author: Integrated Traffic Perception System
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import os
import sys

# Import utilities from lane_det
# Add lane_det to path to import utilities
base_dir = Path(__file__).parent.parent
lane_det_path = base_dir / 'lane_det'
if lane_det_path.exists():
    lane_det_path_str = str(lane_det_path)
    if lane_det_path_str not in sys.path:
        sys.path.insert(0, lane_det_path_str)

try:
    from lane_utils.lane_utils import (
        select_device, scale_coords, non_max_suppression, 
        split_for_trace_model, driving_area_mask, lane_line_mask,
        letterbox, time_synchronized
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import lane detection utilities from {lane_det_path}. "
        f"Make sure lane_det directory exists and contains lane_utils/lane_utils.py. "
        f"Original error: {e}"
    )


class LaneDetector:
    """
    Wrapper class for lane detection, drivable area segmentation, and object detection.
    
    Uses YOLOPv2 model (TorchScript) to perform:
    - Traffic object detection
    - Drivable area segmentation
    - Lane line segmentation
    
    Usage:
        detector = LaneDetector()
        results = detector.detect(frame)
    """
    
    def __init__(self, weights_path=None, device='', img_size=640):
        """
        Initialize the lane detection model.
        
        Args:
            weights_path (str): Path to YOLOPv2 model weights.
                              Default: '../lane_det/data/weights/yolopv2.pt'
            device (str): Device to run on ('cuda', 'cpu', or '' for auto)
            img_size (int): Input image size (default: 640)
        """
        # Resolve device
        if device == '':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            normalized = device.lower()
            if normalized == 'gpu':
                normalized = 'cuda'
            if normalized.startswith('cuda') and torch.cuda.is_available():
                self.device = torch.device(normalized)
            else:
                self.device = torch.device('cpu')
        print(f"Using device for lane model: {self.device}")
        
        # Set default path relative to this file's location
        base_dir = Path(__file__).parent.parent
        
        # Default path for weights
        if weights_path is None:
            weights_path = base_dir / 'lane_det' / 'data' / 'weights' / 'yolopv2.pt'
        
        weights_path = str(weights_path)
        
        # Check if weights exist
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Lane detection weights not found at: {weights_path}")
        
        print(f"Loading lane detection model from: {weights_path}")
        
        # Load TorchScript model
        try:
            # Use map_location to properly map CUDA tensors to CPU if needed
            self.model = torch.jit.load(weights_path, map_location=self.device)
            
            # Use half precision on CUDA for faster inference
            self.half = self.device.type != 'cpu'
            if self.half:
                self.model.half()
            
            self.model.eval()
            
            # Warm up model
            if self.device.type != 'cpu':
                dummy_input = torch.zeros(1, 3, img_size, img_size).to(self.device)
                if self.half:
                    dummy_input = dummy_input.half()
                self.model(dummy_input)
            
            print("  ✓ Lane detection model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load lane detection model: {e}")
        
        self.img_size = img_size
        self.stride = 32
        
        # Object detection classes (common traffic objects)
        # Note: YOLOPv2 detects various traffic objects, but class names may vary
        # We'll use generic names or extract from model if available
        self.classes = ['vehicle', 'pedestrian', 'cyclist', 'traffic_light', 'traffic_sign']
    
    def detect(self, frame, conf_threshold=0.3, iou_threshold=0.45):
        """
        Detect objects, drivable area, and lane lines in frame.
        
        Args:
            frame (numpy.ndarray): BGR image frame (H, W, 3)
            conf_threshold (float): Confidence threshold for object detection (default: 0.3)
            iou_threshold (float): IoU threshold for NMS (default: 0.45)
        
        Returns:
            dict: Detection results containing:
                - detections: List[Dict] with standard format (object detections)
                - drivable_area_mask: np.ndarray (H, W) segmentation mask
                - lane_line_mask: np.ndarray (H, W) segmentation mask
        """
        # Store original frame dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess frame (resize and pad to img_size)
        # Note: lane_det resizes to 1280x720 first, then letterboxes to 640x640
        # For integration, we'll resize directly to img_size
        frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img, ratio, pad = letterbox(frame_resized, self.img_size, stride=self.stride)
        
        # Convert to tensor
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        
        # Normalize to [0, 1]
        if self.half:
            img = img.half()
        else:
            img = img.float()
        img /= 255.0
        
        # Add batch dimension
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img)
        
        # Post-process predictions
        # 1. Process object detections
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(
            pred, 
            conf_thres=conf_threshold, 
            iou_thres=iou_threshold,
            classes=None,
            agnostic=False
        )
        
        # 2. Process segmentation masks
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        
        # Resize masks back to original frame size
        da_seg_mask_resized = cv2.resize(
            da_seg_mask, 
            (orig_w, orig_h), 
            interpolation=cv2.INTER_NEAREST
        )
        ll_seg_mask_resized = cv2.resize(
            ll_seg_mask, 
            (orig_w, orig_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert detections to standard format
        detections = []
        img_shape = img.shape[2:]  # (H, W) of model input
        
        for det in pred:
            if len(det) > 0:
                # Rescale boxes from model input size to original frame size
                # First scale to letterboxed size (1280x720), then to original
                det_rescaled = det.clone()
                det_rescaled[:, :4] = scale_coords(img_shape, det[:, :4], frame_resized.shape)
                
                # Now scale from 1280x720 to original size
                scale_x = orig_w / 1280.0
                scale_y = orig_h / 720.0
                det_rescaled[:, [0, 2]] *= scale_x
                det_rescaled[:, [1, 3]] *= scale_y
                
                # Clip to image bounds
                det_rescaled[:, [0, 2]].clamp_(0, orig_w)
                det_rescaled[:, [1, 3]].clamp_(0, orig_h)
                
                # Convert to detection dictionaries
                for *xyxy, conf, cls in det_rescaled:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    # Get class name (use generic if model doesn't provide names)
                    class_id = int(cls)
                    if class_id < len(self.classes):
                        class_name = self.classes[class_id]
                    else:
                        class_name = f'vehicle_{class_id}'  # Default to vehicle
                    
                    detection = {
                        'model_type': 'lane_detection',  # Use 'lane_detection' as model type
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2)
                    }
                    detections.append(detection)
        
        return {
            'detections': detections,
            'drivable_area_mask': da_seg_mask_resized,
            'lane_line_mask': ll_seg_mask_resized
        }
