"""
Road Anomaly Detection Module (Potholes and Speedbumps)

This module wraps the YOLOv8 model for detecting road anomalies.
The best performing model is YOLOv8, which detects:
- Potholes
- Speedbumps

Note: If YOLOv8 weights are not available, this will attempt to use YOLOv5 weights
as a fallback, but YOLOv8 is recommended for best performance.

Author: Integrated Traffic Perception System
"""

import numpy as np
import cv2
import torch
from pathlib import Path, WindowsPath
import os
import sys
import pickle

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
        
        # Load model based on type
        model_loaded = False
        
        if self.model_type == 'yolov8' and ULTRALYTICS_AVAILABLE:
            # Method 1: Try ultralytics YOLO for YOLOv8 weights
            try:
                self.model = UltralyticsYOLO(weights_path)
                model_loaded = True
                print("  ✓ Loaded using ultralytics YOLO (YOLOv8)")
            except Exception as e:
                print(f"  ⚠ ultralytics YOLO loading failed: {e}")
                print("  Trying torch.hub fallback...")
        
        if not model_loaded:
            # Method 2: Use torch.hub for YOLOv5 weights (more compatible)
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
                # Method 3: Try loading with Windows PosixPath fix
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
                "Tried ultralytics YOLO and torch.hub. "
                "Please check that the weights file is valid."
            )
        
        # Move model to device if supported
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        
        # Road anomaly class names
        # Note: The model may detect potholes and speedbumps, or just potholes
        # We'll use generic names and let the model tell us what it detects
        self.classes = ['pothole', 'speedbump']  # Default names
        
        # Try to get class names from the model
        if hasattr(self.model, 'names'):
            self.classes = list(self.model.names.values())
        
        print(f"Road anomaly detection model loaded successfully!")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Classes: {self.classes}")
    
    def _extract_feature_vector(self, image, bbox):
        """
        Extract a simple feature vector from the detected road anomaly region.
        
        This creates a compact representation of the detected anomaly.
        We resize the cropped region and extract texture/shape features.
        
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
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using simple statistics
        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0
        
        # Flatten the image
        flattened = gray.flatten() / 255.0
        
        # Combine features: mean, std, and flattened image
        feature = np.concatenate([[mean_intensity, std_intensity], flattened[:126]])
        
        return feature.astype(np.float32)
    
    def detect(self, frame, conf_threshold=0.25):
        """
        Detect road anomalies (potholes and speedbumps) in a video frame.
        
        This is the main method you'll call. It takes a BGR image and returns
        a list of detection dictionaries.
        
        Args:
            frame (numpy.ndarray): BGR image (height, width, 3 channels)
            conf_threshold (float): Minimum confidence for detections (default: 0.25)
            
        Returns:
            list: List of detection dictionaries, each containing:
                - model_type (str): "road_anomaly"
                - class_name (str): Type of anomaly ("pothole", "speedbump", etc.)
                - confidence (float): Detection confidence score
                - bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
                - feature_vector (numpy.ndarray): Feature representation of the detection
        """
        detections = []
        
        # Run model inference - handle both ultralytics YOLO and torch.hub formats
        if hasattr(self.model, 'predict') and not hasattr(self.model, 'names'):
            # Ultralytics YOLO format (YOLOv8)
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
                        
                        # Extract feature vector
                        feature_vector = self._extract_feature_vector(frame, (x1, y1, x2, y2))
                        
                        # Create detection dictionary
                        detection = {
                            'model_type': 'road_anomaly',
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'feature_vector': feature_vector
                        }
                        
                        detections.append(detection)
        
        elif hasattr(self.model, 'names'):
            # torch.hub YOLOv5 format
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
                        if class_id in model_classes:
                            class_name = model_classes[class_id]
                        elif class_id < len(self.classes):
                            class_name = self.classes[class_id]
                        else:
                            class_name = f"anomaly_{class_id}"
                        
                        # Extract feature vector
                        feature_vector = self._extract_feature_vector(frame, (x_min, y_min, x_max, y_max))
                        
                        # Create detection dictionary
                        detection = {
                            'model_type': 'road_anomaly',
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                            'feature_vector': feature_vector
                        }
                        
                        detections.append(detection)
        
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

