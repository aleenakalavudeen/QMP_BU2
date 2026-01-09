"""
Driving Risk Assessment and Congestion Detection Module

This module wraps the SqueezeDet-based risk and congestion detection model.
It detects vehicles and pedestrians, then computes:
- Risk levels: LOW, MODERATE, HIGH, CRITICAL
- Congestion metrics: PCU (Passenger Car Unit) values

This wrapper preserves the original model's behavior exactly as-is,
using the original src/ directory without modification.

Author: Integrated Traffic Perception System
"""

import sys
from pathlib import Path
import os
import importlib.util

# Add driving_risk_and_congestion/src to PYTHONPATH (dynamic, portable)
SRC_ROOT = (Path(__file__).resolve().parent.parent /
            "driving_risk_and_congestion" / "src")

# Convert to absolute path and verify it exists
SRC_ROOT = SRC_ROOT.resolve()
if not SRC_ROOT.exists():
    raise ImportError(
        f"driving_risk_and_congestion/src directory not found at: {SRC_ROOT}\n"
        f"Please ensure the directory structure is correct."
    )

# Normalize the path (important for Windows)
src_path = os.path.normpath(str(SRC_ROOT))

# Remove any existing entries for this path to avoid duplicates
sys.path = [p for p in sys.path if os.path.normpath(p) != src_path]

# Insert at the beginning to ensure it's checked first
sys.path.insert(0, src_path)

# Force import utils using importlib to bypass any caching issues
utils_init_path = SRC_ROOT / 'utils' / '__init__.py'
if utils_init_path.exists():
    spec = importlib.util.spec_from_file_location("utils", utils_init_path)
    utils_module = importlib.util.module_from_spec(spec)
    sys.modules['utils'] = utils_module
    spec.loader.exec_module(utils_module)

# Now import utils.image explicitly
utils_image_path = SRC_ROOT / 'utils' / 'image.py'
if utils_image_path.exists():
    spec = importlib.util.spec_from_file_location("utils.image", utils_image_path)
    utils_image_module = importlib.util.module_from_spec(spec)
    sys.modules['utils.image'] = utils_image_module
    spec.loader.exec_module(utils_image_module)

# Debug: verify path is set correctly
print(f"DEBUG: SRC_ROOT = {SRC_ROOT}")
print(f"DEBUG: sys.path[0] = {sys.path[0]}")
print(f"DEBUG: utils.image exists? {(SRC_ROOT / 'utils' / 'image.py').exists()}")

import numpy as np
import cv2
import torch
from pathlib import Path
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import helpers
import helpers.image 

# Import from the original model (exactly as-is)
from datasets.idd import IDD
from engine.detector import Detector
from model.squeezedet import SqueezeDet
from helpers.config import Config
from helpers.model import load_model

# Risk and congestion constants (from original model)
class_ids_list = [0, 1, 2, 3, 4]  # lmv, person, two_wheeler, three_wheeler, hmv
obj_factor = [1, 1, 1, 1, 1]

# Risk centroids
low_centroid = np.array([0.01207104])
moderate_centroid = np.array([0.04221627])
high_centroid = np.array([0.08057645])
critical_centroid = np.array([0.12709216])

# PCU constants
pcu_factors = [1.00, 0.00, 0.75, 2.00, 3.70]  # LMV, Person, 2W, 3W, HMV
widths_m = [1.8, 0.5, 0.8, 1.3, 2.6]
line_left = (0.0, 1.0, 0.4, 0.4)
line_right = (1.0, 1.0, 0.6, 0.4)

# Class names mapping
CLASS_NAMES = ['lmv', 'person', 'two_wheeler', 'three_wheeler', 'hmv']


def draw_text_with_bg(
    img,
    text,
    org,
    font,
    font_scale,
    text_color,
    bg_color,
    thickness,
    padding=10
):
    """
    Draws text with a filled background rectangle.
    (From original demoidd_video_riskest_cong.py)
    """
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org

    # Background rectangle
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        -1
    )

    # Text
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


def point_position(line, point):
    """Check if point is above line (from original model)."""
    x1, y1, x2, y2 = line
    x, y = point
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return y > (m * x + c)


def compute_pcu_from_boxes(boxes, class_ids, H, W):
    """
    Compute PCU from bounding boxes (from original model).
    
    Uses CROPPED image coordinates only.
    Returns:
      - counts per class
      - PCU per class
      - PCU sum
    """
    counts = [0] * 5

    for i in range(boxes.shape[0]):
        cls = int(class_ids[i])
        xmin, ymin, xmax, ymax = boxes[i]

        mid_x = (xmin + xmax) / 2.0

        xmin_n = xmin / W
        xmax_n = xmax / W
        ymax_n = ymax / H

        # ROI filtering
        if xmax_n < 0.5:
            if not point_position(line_left, (xmax_n, ymax_n)):
                continue
        else:
            if not point_position(line_right, (xmin_n, ymax_n)):
                continue

        # Distance proxy filtering
        perceived_width = xmax - xmin
        if perceived_width <= 0:
            continue

        if 0.3 * W < mid_x < 0.7 * W:
            if widths_m[cls] / perceived_width > 0.015:
                continue

        counts[cls] += 1

    pcus = [c * p for c, p in zip(counts, pcu_factors)]
    return counts, pcus, sum(pcus)


def compute_risk_from_detections(boxes, class_ids, H, W):
    """
    Compute risk metrics from detections (from original model).
    
    Returns:
        dict with:
            - counts: list of counts per class
            - areas: list of normalized areas per class
            - x_disps: list of x-axis displacements per class
            - combined: list of combined metrics per class
            - B_sum: total combined metric
            - risk_label: LOW/MODERATE/HIGH/CRITICAL
    """
    counts = [list(class_ids).count(cid) for cid in class_ids_list]
    
    areas = [0] * 5
    disps = [0] * 5
    combs = [0] * 5
    
    area_max = H * W
    
    for i in range(boxes.shape[0]):
        cls = int(class_ids[i])
        xmin, ymin, xmax, ymax = boxes[i]
        
        single_area = (xmax - xmin) * (ymax - ymin) / area_max
        areas[cls] += single_area
        
        x_center = (xmin + xmax) / 2.0
        x_disp = x_center * (W - x_center) / (W * W)
        disps[cls] += x_disp
        combs[cls] += x_disp * obj_factor[cls] * single_area
    
    B_sum = sum(combs)
    
    # Compute risk label using centroids
    dists = [
        np.linalg.norm(B_sum - c)
        for c in [low_centroid, moderate_centroid, high_centroid, critical_centroid]
    ]
    risk_label = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'][np.argmin(dists)]
    
    return {
        'counts': counts,
        'areas': areas,
        'x_disps': disps,
        'combined': combs,
        'B_sum': B_sum,
        'risk_label': risk_label
    }


class RiskCongestionDetector:
    """
    Wrapper class for driving risk assessment and congestion detection.
    
    This class wraps the original SqueezeDet model and risk/congestion
    computation logic without modifying any internal behavior.
    
    Usage:
        detector = RiskCongestionDetector()
        results = detector.detect(frame)
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = '',
                 gpu_id: int = 0,
                 fps: Optional[float] = None):
        """
        Initialize the risk and congestion detection model.
        
        Args:
            model_path (str | None): Path to trained model weights.
                                    Default: '../driving_risk_and_congestion/models/model_best_cropped_full_AdamW.pth'
            device (str): Device to run on ('cuda', 'cpu', or '' for auto)
            gpu_id (int): GPU ID to use (default: 0)
            fps (float | None): FPS for video processing (used to set window size). 
                               If None, defaults to 100 frames window.
        """
        # Set default path relative to this file's location
        base_dir = Path(__file__).parent.parent
        
        if model_path is None:
            model_path = base_dir / 'driving_risk_and_congestion' / 'models' / 'model_best_cropped_full_AdamW.pth'
        
        model_path = str(model_path)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Risk & Congestion model weights not found at: {model_path}\n"
                f"Please ensure the model file exists."
            )
        
        print(f"Loading risk & congestion detection model from: {model_path}")
        
        # Create a minimal config object (matching original model's requirements)
        # We use the original Config class but with minimal arguments
        self.cfg = Config().parse(args=['demoidd_riskest_cong', '--load_model', model_path])
        
        # Set device
        if device == '':
            self.cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            normalized = device.lower()
            if normalized == 'gpu':
                normalized = 'cuda'
            if normalized.startswith('cuda') and torch.cuda.is_available():
                self.cfg.device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cuda')
            else:
                self.cfg.device = torch.device('cpu')
        
        # Set GPU configuration
        self.cfg.gpus = [gpu_id] if self.cfg.device.type == 'cuda' else [-1]
        self.cfg.gpus_str = str(gpu_id) if self.cfg.device.type == 'cuda' else '-1'
        
        # Initialize dataset to get configuration info
        dataset = IDD('val', self.cfg)
        self.cfg = Config().update_dataset_info(self.cfg, dataset)
        
        # Store preprocessing function
        self.preprocess_func = dataset.preprocess
        del dataset
        
        # Load model
        print(f"  Initializing model on device: {self.cfg.device}")
        model = SqueezeDet(self.cfg)
        model = load_model(model, model_path)
        self.detector = Detector(model.to(self.cfg.device), self.cfg)
        
        # Initialize risk smoothing window (for video processing)
        # Match original: window_size = int(ip_fps * 2) for 2 seconds of frames
        if fps is not None and fps > 0:
            window_size = int(fps * 2)
        else:
            window_size = 100  # Default fallback
        self.label_window = deque(maxlen=window_size)
        self.decay_beta = 0.001
        
        print(f"  ✓ Risk & Congestion model loaded successfully (window_size={window_size})")
    
    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Dict:
        """
        Detect vehicles/pedestrians and compute risk/congestion metrics for a single frame.
        
        Args:
            frame (np.ndarray): BGR image frame (H, W, 3)
            frame_id (int | None): Optional frame ID for tracking
        
        Returns:
            dict: Detection results containing:
                - detections: List of detection dictionaries (standardized format)
                - risk_metrics: Dict with risk computation results
                - congestion_metrics: Dict with PCU computation results
                - annotated_frame: Frame with bounding boxes (if debug enabled)
        """
        # Handle frame cropping (bottom 65% as per original model)
        orig_height, orig_width = frame.shape[:2]
        crop_ratio = 0.65
        crop_height = int(orig_height * crop_ratio)
        y_offset = orig_height - crop_height
        
        # Crop frame (matching original model behavior)
        cropped_frame = frame[y_offset:orig_height, :, :].copy()
        
        # Prepare image metadata
        image_meta = {
            'image_id': str(frame_id) if frame_id is not None else 'frame',
            'orig_size': np.array(cropped_frame.shape, dtype=np.int32)
        }
        
        # Preprocess (using original model's preprocessing)
        processed_image, image_meta, _ = self.preprocess_func(cropped_frame, image_meta)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).unsqueeze(0).to(self.cfg.device)
        
        image_meta_t = {
            k: (torch.from_numpy(v).unsqueeze(0).to(self.cfg.device) if isinstance(v, np.ndarray) else [v])
            for k, v in image_meta.items()
        }
        
        # Run detection
        result = self.detector.detect({'image': image_tensor, 'image_meta': image_meta_t})
        det = result[0] if result else {}
        
        # Initialize results
        detections = []
        risk_metrics = None
        congestion_metrics = None
        
        if 'class_ids' in det and 'boxes' in det and len(det['class_ids']) > 0:
            # Copy cropped boxes for risk/PCU computation
            boxes_crop = det['boxes'].copy()
            
            # Shift boxes back to original frame coordinates for visualization
            boxes_original = det['boxes'].copy()
            boxes_original[:, 1] += y_offset
            boxes_original[:, 3] += y_offset
            
            Hc, Wc = image_meta['orig_size'][0], image_meta['orig_size'][1]
            
            # Convert detections to standardized format
            for i in range(len(det['class_ids'])):
                cls_id = int(det['class_ids'][i])
                box = boxes_original[i]
                score = float(det['scores'][i])
                
                detections.append({
                    'model_type': 'risk_congestion',
                    'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}',
                    'confidence': score,
                    'bbox': (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                })
            
            # Compute risk metrics
            risk_metrics = compute_risk_from_detections(
                boxes_crop,
                det['class_ids'],
                Hc, Wc
            )
            
            # Update smoothing window and compute smoothed label
            self.label_window.append(risk_metrics['risk_label'])
            
            if len(self.label_window) >= self.label_window.maxlen:
                scores = {k: 0.0 for k in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']}
                for i, lbl in enumerate(reversed(self.label_window)):
                    scores[lbl] += np.exp(-self.decay_beta * i)
                risk_metrics['smoothed_label'] = max(scores, key=scores.get)
            else:
                risk_metrics['smoothed_label'] = None
            
            # Compute congestion metrics
            pcu_counts, pcu_values, pcu_sum = compute_pcu_from_boxes(
                boxes_crop,
                det['class_ids'],
                Hc, Wc
            )
            
            congestion_metrics = {
                'counts': pcu_counts,
                'pcu_per_class': pcu_values,
                'pcu_sum': pcu_sum
            }
        
        return {
            'detections': detections,
            'risk_metrics': risk_metrics,
            'congestion_metrics': congestion_metrics
        }
    
    def reset_smoothing_window(self, window_size: int = 100):
        """Reset the risk label smoothing window."""
        self.label_window = deque(maxlen=window_size)
    
    def set_smoothing_params(self, window_size: int = None, decay_beta: float = None):
        """Update smoothing parameters."""
        if window_size is not None:
            self.label_window = deque(maxlen=window_size)
        if decay_beta is not None:
            self.decay_beta = decay_beta

