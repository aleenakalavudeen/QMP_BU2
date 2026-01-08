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

# YOLOv5 models are loaded via torch.hub or ultralytics YOLO
# These methods handle preprocessing automatically


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
        
        # Method 1: Try torch.hub without version specification (more compatible)
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
                error_msg = str(e)
                print(f"  ⚠ torch.hub loading failed: {e}")
                # Check if dill is missing and try to install it
                if 'dill' in error_msg.lower() or 'No module named \'dill\'' in error_msg:
                    print("  Attempting to install missing 'dill' package...")
                    try:
                        import subprocess
                        import sys
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dill', '--quiet'])
                        print("  ✓ Successfully installed 'dill'. Retrying torch.hub...")
                        # Retry loading
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                   path=weights_path, 
                                                   force_reload=False,
                                                   source='github',
                                                   device=self.device)
                        self.model.eval()
                        model_loaded = True
                        print("  ✓ Loaded using torch.hub (after installing dill)")
                    except Exception as install_error:
                        print(f"  ⚠ Failed to install dill: {install_error}")
                        print("  Please install dill manually: pip install dill")
        
        # Method 2: Try ultralytics YOLO (can load YOLOv5 weights)
        if not model_loaded:
            try:
                print("  Trying ultralytics YOLO as fallback...")
                from ultralytics import YOLO
                self.device = torch.device('cpu' if device == '' else device)
                self.model = YOLO(weights_path)
                model_loaded = True
                print("  ✓ Loaded using ultralytics YOLO")
            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠ ultralytics YOLO loading failed: {e}")
                # Check if dill is missing and try to install it
                if 'dill' in error_msg.lower() or 'No module named \'dill\'' in error_msg:
                    print("  Attempting to install missing 'dill' package...")
                    try:
                        import subprocess
                        import sys
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dill', '--quiet'])
                        print("  ✓ Successfully installed 'dill'. Retrying ultralytics YOLO...")
                        # Retry loading
                        from ultralytics import YOLO
                        self.model = YOLO(weights_path)
                        model_loaded = True
                        print("  ✓ Loaded using ultralytics YOLO (after installing dill)")
                    except Exception as install_error:
                        print(f"  ⚠ Failed to install dill: {install_error}")
                        print("  Please install dill manually: pip install dill")
        
        if not model_loaded:
            raise RuntimeError(
                f"Failed to load model from {weights_path}. "
                "Tried torch.hub and ultralytics YOLO. "
                "Please check that the weights file is valid."
            )
        
        # Set device for compatibility
        if not hasattr(self, 'device'):
            self.device = torch.device('cpu')
        
        # Traffic light class names (3 classes) - matching data.yaml
        self.classes = ['green', 'red', 'yellow']
        
        # Verify model.names is loaded correctly
        if hasattr(self.model, 'names'):
            model_names = self.model.names
            if isinstance(model_names, dict):
                print(f"  Model class names: {model_names}")
            elif isinstance(model_names, (list, tuple)):
                print(f"  Model class names (list): {list(model_names)}")
            else:
                print(f"  Model class names (unknown format): {model_names}")
        else:
            print(f"  Warning: model.names not found, using default: {self.classes}")
        
        print("Traffic signal detection model loaded successfully!")
    
    
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
        """
        detections = []
        
        # Run model inference
        # Handle different model types (torch.hub or ultralytics YOLO)
        
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
                            
                            detection = {
                                'model_type': 'traffic_signal',
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            }
                            detections.append(detection)
                        return detections  # Successfully processed, return early
            except Exception as e:
                # If predict fails or format is wrong, continue to other methods
                pass
        
        # Check if it's torch.hub YOLOv5 (has .names attribute)
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
                            
                            # Create detection dictionary
                            detection = {
                                'model_type': 'traffic_signal',
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max))
                            }
                            
                            detections.append(detection)
                    return detections  # Successfully processed, return early
            except Exception as e:
                # If torch.hub format fails
                pass
        
        # If we get here, no method worked
        print("Warning: Could not determine model format for signal detection")
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
        print("2. torch.hub or ultralytics YOLO is available")
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

