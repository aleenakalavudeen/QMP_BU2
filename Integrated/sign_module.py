"""
Traffic Sign Detection and Recognition Module

This module wraps the two-stage sign detection pipeline:
1. YOLOv5 for detecting traffic signs (best model)
2. YOLOv8 for recognizing the specific sign type (best model)

Author: Integrated Traffic Perception System
"""

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import os
import sys

# Try to import YOLOv5 utilities from various possible locations
DetectMultiBackend = None
AutoShape = None
select_device = None

# Try to find and import from various YOLOv5 locations
base_dir = Path(__file__).parent.parent
possible_yolov5_paths = [
    base_dir / 'signal_det' / 'signal_det',  # Check signal_det first (might have YOLOv5)
    base_dir / 'yolov5-master',
    base_dir / 'yolov5',
    Path('../yolov5-master'),
    Path('../signal_det/signal_det'),
]

_yolov5_path_used = None
for yolov5_path in possible_yolov5_paths:
    if yolov5_path.exists():
        yolov5_path_str = str(yolov5_path.resolve())  # Use absolute path
        
        # Add YOLOv5 path first
        if yolov5_path_str not in sys.path:
            sys.path.insert(0, yolov5_path_str)
        else:
            # Move to front
            sys.path.remove(yolov5_path_str)
            sys.path.insert(0, yolov5_path_str)
        
        try:
            from models.common import DetectMultiBackend, AutoShape
            from utils.torch_utils import select_device
            _yolov5_path_used = yolov5_path_str
            print(f"  ✓ Found YOLOv5 at: {yolov5_path_str}")
            break
        except ImportError as e:
            # Remove from path if import failed to avoid conflicts
            if yolov5_path_str in sys.path:
                sys.path.remove(yolov5_path_str)


class SignDetector:
    """
    Wrapper class for traffic sign detection and recognition.
    
    This class combines:
    - YOLOv5 model for detecting traffic signs in images
    - YOLOv8 model for recognizing the specific type of sign
    
    Usage:
        detector = SignDetector()
        detections = detector.detect(frame)
    """
    
    def __init__(self, 
                 detect_weights_path=None, 
                 recog_weights_path=None,
                 yolov5_repo_path=None,
                 device: str = ''):
        """
        Initialize the sign detection and recognition models.
        
        Args:
            detect_weights_path (str): Path to YOLOv5 detection weights.
                                      Default: '../sign_det/sign_det/weights/detect_weights.pt'
            recog_weights_path (str): Path to YOLOv8 recognition weights.
                                      Default: '../sign_det/sign_det/weights/recog_weights.pt'
            yolov5_repo_path (str): Path to yolov5 repository folder.
                                    Default: '../yolov5-master/' (or tries to find it)
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
        print(f"Using device for sign models: {self.device}")
        
        # Set default paths relative to this file's location
        base_dir = Path(__file__).parent.parent
        
        # Default paths for weights
        if detect_weights_path is None:
            detect_weights_path = base_dir / 'sign_det' / 'sign_det' / 'weights' / 'detect_weights.pt'
        if recog_weights_path is None:
            recog_weights_path = base_dir / 'sign_det' / 'sign_det' / 'weights' / 'recog_weights.pt'
        
        # Convert to string paths
        detect_weights_path = str(detect_weights_path)
        recog_weights_path = str(recog_weights_path)
        
        # Check if weights exist
        if not os.path.exists(detect_weights_path):
            raise FileNotFoundError(f"Detection weights not found at: {detect_weights_path}")
        if not os.path.exists(recog_weights_path):
            raise FileNotFoundError(f"Recognition weights not found at: {recog_weights_path}")
        
        print(f"Loading sign detection model from: {detect_weights_path}")
        print(f"Loading sign recognition model from: {recog_weights_path}")
        
        # Try loading YOLOv5 detection model using multiple methods
        model_loaded = False
        
        # Method 1: Try using DetectMultiBackend directly (most reliable, avoids torch.hub issues)
        if not (DetectMultiBackend and AutoShape and select_device and _yolov5_path_used):
            print(f"  ⚠ DetectMultiBackend not available (DetectMultiBackend={DetectMultiBackend is not None}, "
                  f"AutoShape={AutoShape is not None}, select_device={select_device is not None}, "
                  f"path={_yolov5_path_used})")
        else:
            try:
                print("  Loading YOLOv5 using DetectMultiBackend...")
                # Ensure YOLOv5 path is first in sys.path
                original_sys_path = sys.path.copy()
                
                if _yolov5_path_used in sys.path:
                    sys.path.remove(_yolov5_path_used)
                sys.path.insert(0, _yolov5_path_used)
                
                try:
                    device_obj = select_device(device if device else '')
                    # DetectMultiBackend will unpickle the model, which needs correct path
                    model = DetectMultiBackend(detect_weights_path, device=device_obj, dnn=False, fp16=False)
                    # Wrap with AutoShape for compatibility with torch.hub interface
                    self.detect_model = AutoShape(model)
                    self.detect_model.eval()
                    model_loaded = True
                    print("  ✓ Loaded using DetectMultiBackend")
                finally:
                    # Restore original sys.path
                    sys.path[:] = original_sys_path
                    # Re-add YOLOv5 path at position 0
                    sys.path.insert(0, _yolov5_path_used)
            except Exception as e:
                print(f"  ⚠ DetectMultiBackend loading failed: {e}")
                import traceback
                traceback.print_exc()
                print("  Trying fallback method...")
        
        # Method 2: Try using ultralytics/yolov5 from GitHub
        if not model_loaded:
            try:
                print("  Loading YOLOv5 from ultralytics hub...")
                # Temporarily ensure YOLOv5 path is first for unpickling
                original_sys_path = sys.path.copy()
                
                if _yolov5_path_used and _yolov5_path_used not in sys.path:
                    sys.path.insert(0, _yolov5_path_used)
                elif _yolov5_path_used and _yolov5_path_used in sys.path:
                    sys.path.remove(_yolov5_path_used)
                    sys.path.insert(0, _yolov5_path_used)
                
                try:
                    # Convert device to string format for torch.hub.load
                    device_str = str(self.device) if isinstance(self.device, torch.device) else self.device
                    self.detect_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                       path=detect_weights_path, 
                                                       force_reload=True,  # Force reload to avoid cache issues
                                                       source='github',
                                                       device=device_str)
                    self.detect_model.eval()
                    model_loaded = True
                    print("  ✓ Loaded using ultralytics/yolov5 hub")
                finally:
                    sys.path[:] = original_sys_path
                    if _yolov5_path_used:
                        sys.path.insert(0, _yolov5_path_used)
            except Exception as e:
                print(f"  ⚠ GitHub loading failed: {e}")
                import traceback
                traceback.print_exc()
                print("  Trying ultralytics YOLO fallback...")
        
        # Method 3: Try ultralytics YOLO (can load YOLOv5 weights)
        if not model_loaded:
            try:
                print("  Trying ultralytics YOLO as fallback...")
                # Temporarily ensure YOLOv5 path is first for unpickling YOLOv5 weights
                original_sys_path = sys.path.copy()
                
                if _yolov5_path_used and _yolov5_path_used not in sys.path:
                    sys.path.insert(0, _yolov5_path_used)
                elif _yolov5_path_used and _yolov5_path_used in sys.path:
                    sys.path.remove(_yolov5_path_used)
                    sys.path.insert(0, _yolov5_path_used)
                
                try:
                    self.detect_model = YOLO(detect_weights_path)
                    self.detect_model.eval()
                    model_loaded = True
                    print("  ✓ Loaded using ultralytics YOLO")
                finally:
                    sys.path[:] = original_sys_path
                    if _yolov5_path_used:
                        sys.path.insert(0, _yolov5_path_used)
            except Exception as e:
                print(f"  ⚠ ultralytics YOLO loading failed: {e}")
                import traceback
                traceback.print_exc()
        
        if not model_loaded:
            raise RuntimeError(
                f"Failed to load YOLOv5 detection model from {detect_weights_path}. "
                "Tried DetectMultiBackend, torch.hub, and ultralytics YOLO. "
                "Please check that the weights file is valid."
            )
        
        # Load YOLOv8 recognition model
        self.recog_model = YOLO(recog_weights_path)
        
        # Move models to device if supported
        if hasattr(self.detect_model, 'to'):
            self.detect_model.to(self.device)
        if hasattr(self.recog_model, 'to'):
            self.recog_model.to(self.device)
        
        # Sign class names (12 classes as per the original pipeline)
        self.classes = [
            "20kmhr", "30kmhr", "40kmhr", "50kmhr", 
            "Intersection", "Men At Work", "Narrow Road", 
            "No Entry", "NoParking", "PedCross", "Stop", "Yield"
        ]
        
        # Set models to evaluation mode
        self.detect_model.eval()
        
        print("Sign detection and recognition models loaded successfully!")
    
    def _red_color_segmentation(self, image):
        """
        Apply red color segmentation to filter traffic signs.
        This matches the original sign_det pipeline logic exactly.
        This helps improve detection accuracy by focusing on red-colored signs.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            numpy.ndarray: Gray mask where red regions are filtered by area
        """
        # Convert BGR to HSV color space (better for color-based segmentation)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges in HSV
        # Red wraps around in HSV, so we need two ranges
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 120, 0])
        upper1 = np.array([10, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160, 120, 0])
        upper2 = np.array([185, 255, 255])
        
        # Create masks for both red ranges
        lower_mask = cv2.inRange(hsv_image, lower1, upper1)
        upper_mask = cv2.inRange(hsv_image, lower2, upper2)
        red_mask = lower_mask | upper_mask  # Combine both masks
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)  # don't change 5
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create red segmented image
        red_segmented_image = cv2.bitwise_and(image, image, mask=red_mask)
        gray = cv2.cvtColor(red_segmented_image, cv2.COLOR_BGR2GRAY)
        
        # Find contours in the opened mask
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_threshold = 100
        max_area_threshold = 8000
        
        # Filter out contours with area less than the threshold and fill them with black color
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold or area > max_area_threshold:
                cv2.drawContours(gray, [contour], 0, (0, 255, 0), -1)
        
        red_segmented_image = cv2.bitwise_and(image, image, mask=gray)
        
        return gray
    
    
    def detect(self, frame, conf_threshold=0.3):
        """
        Detect and recognize traffic signs in a video frame.
        
        This is the main method you'll call. It takes a BGR image (like from cv2.imread
        or video capture) and returns a list of detection dictionaries.
        
        Args:
            frame (numpy.ndarray): BGR image (height, width, 3 channels)
            conf_threshold (float): Minimum confidence for detections (default: 0.3)
            
        Returns:
            list: List of detection dictionaries, each containing:
                - model_type (str): "traffic_sign"
                - class_name (str): Name of the sign (e.g., "Stop", "Yield")
                - confidence (float): Detection confidence score
                - bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        """
        detections = []
        
        # Run detection model on the frame
        results = self.detect_model(frame)
        
        # Get bounding boxes from detection results
        # results.xyxy[0] contains: [x1, y1, x2, y2, confidence, class_id]
        bounding_boxes = results.xyxy[0].cpu().numpy()
        
        # Apply red color segmentation for filtering
        red_mask = self._red_color_segmentation(frame)
        
        # Process each detected bounding box
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max, confidence, _ = box
            
            # Adjust confidence based on red mask presence
            # If the detected region doesn't overlap with red areas, reduce confidence
            # This matches the original pipeline logic exactly
            try:
                box_mask_region = red_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
                if not box_mask_region.any():
                    confidence = confidence * 0.5  # Reduce confidence if no red detected
            except (IndexError, ValueError):
                # If mask region is out of bounds, skip confidence adjustment
                pass
            
            # Only process detections above threshold
            if confidence > conf_threshold:
                # Crop the detected sign region
                detected_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                # Run recognition model on the cropped sign
                recog_pred = self.recog_model.predict(detected_img, conf=0.5, verbose=False)
                
                # Check if recognition found a class
                if len(recog_pred) != 0 and recog_pred[0].probs is not None:
                    # Get class probabilities
                    preds = recog_pred[0].probs.cpu().numpy()
                    class_id = np.argmax(preds.data)
                    recog_confidence = float(preds.data[class_id])
                    
                    # Get class name
                    class_name = self.classes[int(class_id)]
                    
                    # Create detection dictionary
                    detection = {
                        'model_type': 'traffic_sign',
                        'class_name': class_name,
                        'confidence': float(confidence * recog_confidence),  # Combined confidence
                        'bbox': (int(x_min), int(y_min), int(x_max), int(y_max))
                    }
                    
                    detections.append(detection)
        
        return detections


# Example usage and testing function
if __name__ == "__main__":
    """
    Test the sign detector module.
    Run this file directly to test with a sample image.
    """
    import sys
    
    # Create detector instance
    try:
        detector = SignDetector()
        print("\n✓ Sign detector initialized successfully!")
    except Exception as e:
        print(f"\n✗ Error initializing detector: {e}")
        print("\nMake sure:")
        print("1. Weight files exist in sign_det/sign_det/weights/")
        print("2. YOLOv5 repository is available (or ultralytics will download)")
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
            
            print(f"\nFound {len(detections)} traffic signs:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
                print(f"     BBox: {det['bbox']}")
    else:
        print("\nTo test with an image, run:")
        print("  python sign_module.py path/to/image.jpg")

