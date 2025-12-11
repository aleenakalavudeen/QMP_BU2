"""
Traffic Signal (Light) Detection Module

This module wraps the YOLOv5 model for detecting traffic lights.
The best performing model is YOLOv5, which detects and classifies:
- Red traffic lights
- Green traffic lights  
- Yellow traffic lights

Author: Integrated Traffic Perception System
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import os
import sys

# Try to import YOLOv5 utilities from signal_det directory
signal_det_path = Path(__file__).parent.parent / 'signal_det' / 'signal_det'
DetectMultiBackend = None
select_device = None

if signal_det_path.exists():
    if str(signal_det_path) not in sys.path:
        sys.path.insert(0, str(signal_det_path))
    
    try:
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
    except ImportError:
        # Local YOLOv5 not available, will use fallback
        DetectMultiBackend = None
        select_device = None


class SignalDetector:
    """
    Wrapper class for traffic signal (light) detection.
    
    Uses YOLOv5 model (best performing model) to detect and classify
    traffic lights into red, green, and yellow.
    
    Usage:
        detector = SignalDetector()
        detections = detector.detect(frame)
    """
    
    def __init__(self, weights_path=None, device=''):
        """
        Initialize the traffic signal detection model.
        
        Args:
            weights_path (str): Path to YOLOv5 model weights.
                               Default: '../signal_det/signal_det/yolov5 trained model.pt'
            device (str): Device to run on ('cuda', 'cpu', or '' for auto)
        """
        # Set default path relative to this file's location
        base_dir = Path(__file__).parent.parent
        
        if weights_path is None:
            weights_path = base_dir / 'signal_det' / 'signal_det' / 'yolov5 trained model.pt'
        
        weights_path = str(weights_path)
        
        # Check if weights exist
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Signal detection weights not found at: {weights_path}")
        
        print(f"Loading traffic signal detection model from: {weights_path}")
        
        # Try multiple loading methods
        model_loaded = False
        
        # Method 1: Try using local YOLOv5 DetectMultiBackend (best)
        if DetectMultiBackend and select_device:
            try:
                self.device = select_device(device)
                self.model = DetectMultiBackend(weights_path, device=self.device, dnn=False, fp16=False)
                self.model.eval()
                model_loaded = True
                print("  ✓ Loaded using local YOLOv5 DetectMultiBackend")
            except Exception as e:
                print(f"  ⚠ Local YOLOv5 loading failed: {e}")
                print("  Trying fallback method...")
        
        # Method 2: Try torch.hub without version specification (more compatible)
        if not model_loaded:
            try:
                print("  Using torch.hub (ultralytics/yolov5)...")
                self.device = torch.device('cpu' if device == '' else device)
                # Load without version tag to avoid compatibility issues
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=weights_path, 
                                           force_reload=False,  # Don't force reload to use cache
                                           source='github',
                                           device=self.device)
                self.model.eval()
                model_loaded = True
                print("  ✓ Loaded using torch.hub")
            except Exception as e:
                print(f"  ⚠ torch.hub loading failed: {e}")
        
        # Method 3: Try ultralytics YOLO (can load YOLOv5 weights)
        if not model_loaded:
            try:
                print("  Trying ultralytics YOLO as fallback...")
                from ultralytics import YOLO
                self.device = torch.device('cpu' if device == '' else device)
                self.model = YOLO(weights_path)
                model_loaded = True
                print("  ✓ Loaded using ultralytics YOLO")
            except Exception as e:
                print(f"  ⚠ ultralytics YOLO loading failed: {e}")
        
        if not model_loaded:
            raise RuntimeError(
                f"Failed to load model from {weights_path}. "
                "Tried DetectMultiBackend, torch.hub, and ultralytics YOLO. "
                "Please check that the weights file is valid."
            )
        
        # Set device for compatibility
        if not hasattr(self, 'device'):
            self.device = torch.device('cpu')
        
        # Traffic light class names (3 classes)
        self.classes = ['green', 'red', 'yellow']
        
        print("Traffic signal detection model loaded successfully!")
    
    def _extract_feature_vector(self, image, bbox):
        """
        Extract a simple feature vector from the detected traffic light region.
        
        This creates a compact representation of the detected traffic light.
        We resize the cropped region and extract color and shape features.
        
        Args:
            image (numpy.ndarray): Full image
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            
        Returns:
            numpy.ndarray: Feature vector of fixed size
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Crop the detected region
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            # Return zero vector if crop is invalid
            return np.zeros(128, dtype=np.float32)
        
        # Resize to fixed size
        cropped_resized = cv2.resize(cropped, (16, 16))
        
        # Convert to HSV to capture color information
        hsv = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2HSV)
        
        # Extract features: mean HSV values and flattened image
        mean_hsv = np.mean(hsv, axis=(0, 1)) / 255.0  # Normalize
        flattened = cropped_resized.flatten() / 255.0  # Normalize
        
        # Combine into feature vector (3 HSV means + flattened image)
        feature = np.concatenate([mean_hsv, flattened[:125]])  # Keep total size 128
        
        return feature.astype(np.float32)
    
    def detect(self, frame, conf_threshold=0.25):
        """
        Detect traffic signals (lights) in a video frame.
        
        This is the main method you'll call. It takes a BGR image and returns
        a list of detection dictionaries.
        
        Args:
            frame (numpy.ndarray): BGR image (height, width, 3 channels)
            conf_threshold (float): Minimum confidence for detections (default: 0.25)
            
        Returns:
            list: List of detection dictionaries, each containing:
                - model_type (str): "traffic_signal"
                - class_name (str): Color of the light ("red", "green", or "yellow")
                - confidence (float): Detection confidence score
                - bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
                - feature_vector (numpy.ndarray): Feature representation of the detection
        """
        detections = []
        
        # Run model inference
        # Handle different model types (DetectMultiBackend, torch.hub, or ultralytics YOLO)
        
        # Check if it's ultralytics YOLO (has .predict method and returns Results objects)
        if hasattr(self.model, 'predict'):
            try:
                # Try ultralytics YOLO format first
                results = self.model.predict(frame, conf=conf_threshold, verbose=False)
                
                # Check if results is a list of Results objects (ultralytics format)
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            box = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = box
                            confidence = float(boxes.conf[i].cpu().numpy())
                            class_id = int(boxes.cls[i].cpu().numpy())
                            
                            # Get class name
                            if class_id < len(self.classes):
                                class_name = self.classes[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            # Extract feature vector
                            feature_vector = self._extract_feature_vector(frame, (x1, y1, x2, y2))
                            
                            detection = {
                                'model_type': 'traffic_signal',
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'feature_vector': feature_vector
                            }
                            detections.append(detection)
                        return detections  # Successfully processed, return early
            except Exception as e:
                # If predict fails or format is wrong, continue to other methods
                pass
        
        # Check if it's torch.hub YOLOv5 (has .names attribute and returns results object with .xyxy)
        if hasattr(self.model, 'names'):
            try:
                # torch.hub YOLOv5 format
                results = self.model(frame)
                
                # Check if results has .xyxy attribute (torch.hub format)
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    bounding_boxes = results.xyxy[0].cpu().numpy()
                    
                    # Get class names from model
                    model_classes = self.model.names if hasattr(self.model, 'names') else {0: 'green', 1: 'red', 2: 'yellow'}
                    
                    # Process each detection
                    for box in bounding_boxes:
                        x_min, y_min, x_max, y_max, confidence, class_id = box
                        class_id = int(class_id)
                        
                        # Filter by confidence threshold
                        if confidence > conf_threshold:
                            # Get class name
                            if class_id in model_classes:
                                class_name = model_classes[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            # Extract feature vector
                            feature_vector = self._extract_feature_vector(frame, (x_min, y_min, x_max, y_max))
                            
                            # Create detection dictionary
                            detection = {
                                'model_type': 'traffic_signal',
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                                'feature_vector': feature_vector
                            }
                            
                            detections.append(detection)
                    return detections  # Successfully processed, return early
            except Exception as e:
                # If torch.hub format fails, continue to DetectMultiBackend
                pass
        
        # Check if it's DetectMultiBackend (has .stride attribute)
        if hasattr(self.model, 'stride'):
            try:
                from utils.general import non_max_suppression, scale_boxes
                
                # Preprocess image
                img = torch.from_numpy(frame).to(self.device)
                img = img.float() / 255.0  # Normalize to 0-1
                img = img.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
                
                # Run inference
                pred = self.model(img, augment=False)
                
                # Apply NMS
                pred = non_max_suppression(pred, conf_threshold, 0.45, max_det=1000)[0]
                
                # Process predictions
                if len(pred) > 0:
                    # Scale boxes to original image size
                    pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
                    
                    for det in pred:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                        class_id = int(cls)
                        
                        if class_id < len(self.classes):
                            class_name = self.classes[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        # Extract feature vector
                        feature_vector = self._extract_feature_vector(frame, (x1, y1, x2, y2))
                        
                        detection = {
                            'model_type': 'traffic_signal',
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'feature_vector': feature_vector
                        }
                        
                        detections.append(detection)
                    return detections  # Successfully processed, return early
            except ImportError:
                print("Warning: Could not import YOLOv5 utilities for DetectMultiBackend")
        
        # If we get here, no method worked
        print("Warning: Could not determine model format for signal detection")
        return detections
        
        return detections


# Example usage and testing function
if __name__ == "__main__":
    """
    Test the signal detector module.
    Run this file directly to test with a sample image.
    """
    import sys
    
    # Create detector instance
    try:
        detector = SignalDetector()
        print("\n✓ Traffic signal detector initialized successfully!")
    except Exception as e:
        print(f"\n✗ Error initializing detector: {e}")
        print("\nMake sure:")
        print("1. Weight file exists at signal_det/signal_det/yolov5 trained model.pt")
        print("2. YOLOv5 utilities are available")
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
            
            print(f"\nFound {len(detections)} traffic signals:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']} light (confidence: {det['confidence']:.2f})")
                print(f"     BBox: {det['bbox']}")
    else:
        print("\nTo test with an image, run:")
        print("  python signal_module.py path/to/image.jpg")

