"""
Road Anomaly Detection Module (Potholes and Speedbumps)

This module uses the same logic as speedbump_pothole_det/detect.py for detecting road anomalies.
It supports both YOLOv5 (using DetectMultiBackend) and YOLOv8 (using ultralytics).
The module detects:
- Potholes
- Speedbumps

Note: YOLOv5 with DetectMultiBackend is preferred to match the original detect.py logic.
YOLOv8 is available as an alternative.

Author: Integrated Traffic Perception System
"""

import numpy as np
import cv2
import torch
from pathlib import Path, WindowsPath
import os
import sys
import pickle

# Try to import YOLOv5 utilities from signal_det directory (same as detect.py uses)
signal_det_path = Path(__file__).parent.parent / 'signal_det' / 'signal_det'
DetectMultiBackend = None
select_device = None
non_max_suppression = None
scale_boxes = None
check_img_size = None

if signal_det_path.exists():
    if str(signal_det_path) not in sys.path:
        sys.path.insert(0, str(signal_det_path))
    
    try:
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
        from utils.general import non_max_suppression, scale_boxes, check_img_size
    except ImportError:
        # Local YOLOv5 not available, will use fallback
        DetectMultiBackend = None
        select_device = None
        non_max_suppression = None
        scale_boxes = None
        check_img_size = None

# Try to import ultralytics YOLO (for YOLOv8)
try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    UltralyticsYOLO = None


def load_yolov5_weights_windows_fix(weights_path):
    """
    Load YOLOv5 weights with Windows compatibility fix.
    
    Some YOLOv5 weights contain PosixPath objects (from Linux) that can't
    be loaded on Windows. This function patches pathlib temporarily to handle this.
    
    Args:
        weights_path (str): Path to weights file
    
    Returns:
        torch model: Loaded model
    """
    import pathlib
    
    # Temporarily patch pathlib to convert PosixPath to WindowsPath
    original_posix_path = pathlib.PosixPath
    
    class PatchedPosixPath(WindowsPath):
        """Patched PosixPath that works on Windows"""
        pass
    
    # Replace PosixPath temporarily
    pathlib.PosixPath = PatchedPosixPath
    
    try:
        # Now try loading with torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=weights_path, 
                              force_reload=False,
                              source='github',
                              _verbose=False)
        model.eval()
        return model
    finally:
        # Restore original PosixPath
        pathlib.PosixPath = original_posix_path


class RoadAnomalyDetector:
    """
    Wrapper class for road anomaly detection (potholes and speedbumps).
    
    Uses YOLOv8 model (best performing model) to detect road anomalies.
    Falls back to YOLOv5 if YOLOv8 weights are not available.
    
    Usage:
        detector = RoadAnomalyDetector()
        detections = detector.detect(frame)
    """
    
    def __init__(self, weights_path=None, use_yolov8=True, device: str = ''):
        """
        Initialize the road anomaly detection model.
        
        Args:
            weights_path (str): Path to model weights.
                               If None, tries to find YOLOv8 weights first, then YOLOv5.
            use_yolov8 (bool): Prefer YOLOv8 if available (default: True)
            device (str): Device to run on ('cuda', 'cpu', or '' for auto)
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
        print(f"Using device for road anomaly model: {self.device}")
        
        # Set default paths relative to this file's location
        base_dir = Path(__file__).parent.parent
        
        # Try to find weights
        if weights_path is None:
            # First try YOLOv8 weights (best model)
            yolov8_path = base_dir / 'speedbump_pothole_det' / 'speedbump_pothole_det' / 'yolov8_trained_model.pt'
            yolov5_path = base_dir / 'speedbump_pothole_det' / 'speedbump_pothole_det' / 'weights_yolov5' / 'best.pt'
            
            if use_yolov8 and yolov8_path.exists():
                weights_path = yolov8_path
                self.model_type = 'yolov8'
            elif yolov5_path.exists():
                weights_path = yolov5_path
                self.model_type = 'yolov5'
                print("Warning: Using YOLOv5 weights. YOLOv8 is recommended for best performance.")
            else:
                # Try to find any .pt file in the weights directory
                weights_dir = base_dir / 'speedbump_pothole_det' / 'speedbump_pothole_det' / 'weights_yolov5'
                if weights_dir.exists():
                    pt_files = list(weights_dir.glob('*.pt'))
                    if pt_files:
                        weights_path = pt_files[0]
                        self.model_type = 'yolov5'
                        print("Warning: Using YOLOv5 weights. YOLOv8 is recommended for best performance.")
                    else:
                        raise FileNotFoundError(
                            f"No model weights found. Expected YOLOv8 weights at: {yolov8_path}\n"
                            f"Or YOLOv5 weights at: {yolov5_path}"
                        )
                else:
                    raise FileNotFoundError(
                        f"No model weights found. Expected YOLOv8 weights at: {yolov8_path}\n"
                        f"Or YOLOv5 weights at: {yolov5_path}"
                    )
        else:
            weights_path = Path(weights_path)
            # Determine model type from path or file
            if 'yolov8' in str(weights_path).lower() or use_yolov8:
                self.model_type = 'yolov8'
            else:
                self.model_type = 'yolov5'
        
        weights_path = str(weights_path)
        
        # Check if weights exist
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Road anomaly detection weights not found at: {weights_path}")
        
        print(f"Loading road anomaly detection model ({self.model_type.upper()}) from: {weights_path}")
        
        # Try to find data.yaml file (same as detect.py)
        base_dir = Path(__file__).parent.parent
        data_yaml_path = None
        possible_data_yamls = [
            base_dir / 'speedbump_pothole_det' / 'speedbump_pothole_det' / 'dataset' / 'data.yaml',
        ]
        
        for data_yaml in possible_data_yamls:
            if data_yaml.exists():
                data_yaml_path = str(data_yaml)
                print(f"  Found data.yaml at: {data_yaml_path}")
                break
        
        # Load model based on type - prioritize DetectMultiBackend for YOLOv5 (same as detect.py)
        model_loaded = False
        
        # Method 1: For YOLOv5, try DetectMultiBackend first (same logic as detect.py)
        if self.model_type == 'yolov5' and DetectMultiBackend and select_device:
            try:
                device_obj = select_device(device if device else '')
                self.device = device_obj
                # Use DetectMultiBackend with same parameters as detect.py
                self.model = DetectMultiBackend(
                    weights_path,
                    device=device_obj,
                    dnn=False,
                    fp16=False,
                    data=data_yaml_path if data_yaml_path else None
                )
                self.model.eval()
                model_loaded = True
                print("  ✓ Loaded using DetectMultiBackend (YOLOv5) - same as detect.py")
            except Exception as e:
                print(f"  ⚠ DetectMultiBackend loading failed: {e}")
                print("  Trying torch.hub fallback...")
        
        # Method 2: For YOLOv8, try ultralytics YOLO
        if not model_loaded and self.model_type == 'yolov8' and ULTRALYTICS_AVAILABLE:
            try:
                self.model = UltralyticsYOLO(weights_path)
                model_loaded = True
                print("  ✓ Loaded using ultralytics YOLO (YOLOv8)")
            except Exception as e:
                print(f"  ⚠ ultralytics YOLO loading failed: {e}")
                print("  Trying torch.hub fallback...")
        
        # Method 3: Fallback to torch.hub for YOLOv5 weights
        if not model_loaded:
            try:
                print("  Using torch.hub for YOLOv5 weights...")
                # Try loading with force_reload to clear cache issues
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                               path=weights_path, 
                                               force_reload=True,  # Force reload to avoid cache issues
                                               source='github',
                                               _verbose=False)
                except Exception as e1:
                    # If force_reload fails, try without it
                    print(f"  ⚠ Force reload failed, trying cached version...")
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                               path=weights_path, 
                                               force_reload=False,
                                               source='github',
                                               _verbose=False)
                
                self.model.eval()
                model_loaded = True
                print("  ✓ Loaded using torch.hub (YOLOv5)")
            except Exception as e:
                print(f"  ⚠ torch.hub loading failed: {e}")
                print("  Trying Windows compatibility fix...")
                # Method 4: Try loading with Windows PosixPath fix
                try:
                    self.model = load_yolov5_weights_windows_fix(weights_path)
                    model_loaded = True
                    print("  ✓ Loaded using Windows-compatible loader")
                except Exception as e2:
                    print(f"  ⚠ Windows-compatible loading also failed: {e2}")
                    print("  Suggestion: Try clearing torch cache: rmdir /s /q %USERPROFILE%\\.cache\\torch\\hub")
        
        if not model_loaded:
            raise RuntimeError(
                f"Failed to load model from {weights_path}. "
                "Tried DetectMultiBackend, ultralytics YOLO, and torch.hub. "
                "Please check that the weights file is valid."
            )
        
        # Move model to device if supported
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        
        # Road anomaly class names - same as detect.py gets from model
        # Try to get class names from the model (same logic as detect.py)
        self.classes = ['pothole', 'speedbump']  # Default names
        
        if hasattr(self.model, 'names'):
            # Get names from model (same as detect.py line 167)
            names = self.model.names
            if isinstance(names, dict):
                # Convert dict to list, handling both int keys and string values
                max_key = max(names.keys()) if names else -1
                if max_key >= 0:
                    self.classes = [names.get(i, f'class_{i}') for i in range(max_key + 1)]
                else:
                    self.classes = list(names.values()) if names else ['pothole', 'speedbump']
            elif isinstance(names, list):
                self.classes = names
            else:
                # Fallback: try to get from model attributes
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                    model_names = self.model.model.names
                    if isinstance(model_names, dict):
                        max_key = max(model_names.keys()) if model_names else -1
                        if max_key >= 0:
                            self.classes = [model_names.get(i, f'class_{i}') for i in range(max_key + 1)]
                        else:
                            self.classes = list(model_names.values()) if model_names else ['pothole', 'speedbump']
                    elif isinstance(model_names, list):
                        self.classes = model_names
        
        print(f"Road anomaly detection model loaded successfully!")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Classes: {self.classes}")
    
    
    def detect(self, frame, conf_threshold=0.25, iou_threshold=0.45, max_det=1000):
        """
        Detect road anomalies (potholes and speedbumps) in a video frame.
        
        This method uses the same logic as speedbump_pothole_det/detect.py:
        - For YOLOv5: Uses DetectMultiBackend with proper preprocessing, NMS, and box scaling
        - For YOLOv8: Uses ultralytics YOLO
        
        Args:
            frame (numpy.ndarray): BGR image (height, width, 3 channels)
            conf_threshold (float): Minimum confidence for detections (default: 0.25, same as detect.py)
            iou_threshold (float): NMS IoU threshold (default: 0.45, same as detect.py)
            max_det (int): Maximum detections per image (default: 1000, same as detect.py)
            
        Returns:
            list: List of detection dictionaries, each containing:
                - model_type (str): "road_anomaly"
                - class_name (str): Type of anomaly ("pothole", "speedbump", etc.)
                - confidence (float): Detection confidence score
                - bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        """
        detections = []
        
        # Method 1: DetectMultiBackend (YOLOv5) - same logic as detect.py
        if hasattr(self.model, 'stride') and non_max_suppression and scale_boxes:
            try:
                # Preprocess image - same as detect.py lines 186-191
                im = torch.from_numpy(frame).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                
                # Inference - same as detect.py line 207
                pred = self.model(im, augment=False, visualize=False)
                
                # NMS - same as detect.py line 210
                pred = non_max_suppression(pred, conf_threshold, iou_threshold, None, False, max_det=max_det)
                
                # Process predictions - same as detect.py lines 245-280
                if len(pred) > 0 and len(pred[0]) > 0:
                    det = pred[0]
                    
                    # Rescale boxes from img_size to im0 size - same as detect.py line 247
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    
                    # Get class names from model
                    names = self.model.names if hasattr(self.model, 'names') else self.classes
                    if isinstance(names, dict):
                        # Convert dict to list if needed
                        max_class_id = max(names.keys()) if names else 0
                        names_list = [names.get(i, f'class_{i}') for i in range(max_class_id + 1)]
                    else:
                        names_list = names if isinstance(names, list) else self.classes
                    
                    # Process each detection - same as detect.py lines 255-280
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        confidence = float(conf)
                        
                        # Get class name
                        if c < len(names_list):
                            class_name = names_list[c]
                        elif c < len(self.classes):
                            class_name = self.classes[c]
                        else:
                            class_name = f"anomaly_{c}"
                        
                        # Create detection dictionary
                        detection = {
                            'model_type': 'road_anomaly',
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        }
                        
                        detections.append(detection)
                
                return detections  # Successfully processed with DetectMultiBackend
            except Exception as e:
                print(f"  ⚠ DetectMultiBackend inference failed: {e}")
                print("  Trying alternative methods...")
        
        # Method 2: Ultralytics YOLO format (YOLOv8)
        if hasattr(self.model, 'predict') and not hasattr(self.model, 'names'):
            try:
                results = self.model.predict(frame, conf=conf_threshold, verbose=False)
                
                # Process results
                if len(results) > 0:
                    result = results[0]
                    
                    # Get bounding boxes and predictions
                    if result.boxes is not None:
                        boxes = result.boxes
                        
                        # Extract information for each detection
                        for i in range(len(boxes)):
                            # Get bounding box coordinates
                            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = box
                            
                            # Get confidence
                            confidence = float(boxes.conf[i].cpu().numpy())
                            
                            # Get class ID and name
                            class_id = int(boxes.cls[i].cpu().numpy())
                            
                            # Get class name
                            if class_id < len(self.classes):
                                class_name = self.classes[class_id]
                            else:
                                class_name = f"anomaly_{class_id}"
                            
                            # Create detection dictionary
                            detection = {
                                'model_type': 'road_anomaly',
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            }
                            
                            detections.append(detection)
                
                return detections  # Successfully processed with ultralytics YOLO
            except Exception as e:
                print(f"  ⚠ Ultralytics YOLO inference failed: {e}")
        
        # Method 3: torch.hub YOLOv5 format (fallback)
        if hasattr(self.model, 'names'):
            try:
                results = self.model(frame)
                
                # Get bounding boxes
                # results.xyxy[0] contains: [x1, y1, x2, y2, confidence, class_id]
                if len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    bounding_boxes = results.xyxy[0].cpu().numpy()
                    
                    # Get class names from model
                    model_classes = self.model.names if hasattr(self.model, 'names') else {0: 'pothole'}
                    
                    # Process each detection
                    for box in bounding_boxes:
                        x_min, y_min, x_max, y_max, confidence, class_id = box
                        class_id = int(class_id)
                        
                        # Filter by confidence threshold
                        if confidence > conf_threshold:
                            # Get class name
                            if isinstance(model_classes, dict):
                                class_name = model_classes.get(class_id, f"anomaly_{class_id}")
                            elif class_id < len(self.classes):
                                class_name = self.classes[class_id]
                            else:
                                class_name = f"anomaly_{class_id}"
                            
                            # Create detection dictionary
                            detection = {
                                'model_type': 'road_anomaly',
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max))
                            }
                            
                            detections.append(detection)
                
                return detections  # Successfully processed with torch.hub
            except Exception as e:
                print(f"  ⚠ torch.hub inference failed: {e}")
        
        # If we get here, no method worked
        print("Warning: Could not determine model format for road anomaly detection")
        return detections


# Example usage and testing function
if __name__ == "__main__":
    """
    Test the road anomaly detector module.
    Run this file directly to test with a sample image.
    """
    import sys
    
    # Create detector instance
    try:
        detector = RoadAnomalyDetector()
        print("\n✓ Road anomaly detector initialized successfully!")
    except Exception as e:
        print(f"\n✗ Error initializing detector: {e}")
        print("\nMake sure:")
        print("1. Weight file exists (preferably YOLOv8, or YOLOv5 as fallback)")
        print("2. Expected locations:")
        print("   - YOLOv8: speedbump_pothole_det/speedbump_pothole_det/yolov8_trained_model.pt")
        print("   - YOLOv5: speedbump_pothole_det/speedbump_pothole_det/weights_yolov5/best.pt")
        sys.exit(1)
    
    # Test with a sample image (if provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
        else:
            print(f"\nTesting detection on: {image_path}")
            detections = detector.detect(frame)
            
            print(f"\nFound {len(detections)} road anomalies:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
                print(f"     BBox: {det['bbox']}")
    else:
        print("\nTo test with an image, run:")
        print("  python road_module.py path/to/image.jpg")

