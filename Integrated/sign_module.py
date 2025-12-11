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
        
        # Load YOLOv5 detection model
        # Try to find yolov5 repository path
        if yolov5_repo_path is None:
            # Try common locations
            possible_paths = [
                base_dir / 'yolov5-master',
                base_dir / 'yolov5',
                Path('../yolov5-master'),
            ]
            yolov5_repo_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    yolov5_repo_path = str(path)
                    break
        
        if yolov5_repo_path is None:
            # Fallback: use ultralytics hub
            print("Warning: Using ultralytics hub for yolov5: v6.2 (may download model)")
            self.detect_model = torch.hub.load('ultralytics/yolov5: v6.2', 'custom', 
                                               path=detect_weights_path, 
                                               force_reload=True, 
                                               source='github')
        else:
            # Load from local repository
            self.detect_model = torch.hub.load(yolov5_repo_path, 'custom', 
                                               path=detect_weights_path, 
                                               force_reload=True, 
                                               source='local')
        
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
        This helps improve detection accuracy by focusing on red-colored signs.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            numpy.ndarray: Binary mask where red regions are white
        """
        # Convert BGR to HSV color space (better for color-based segmentation)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges in HSV
        # Red wraps around in HSV, so we need two ranges
        lower1 = np.array([0, 120, 0])    # Lower red range
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 120, 0])  # Upper red range
        upper2 = np.array([185, 255, 255])
        
        # Create masks for both red ranges
        lower_mask = cv2.inRange(hsv_image, lower1, upper1)
        upper_mask = cv2.inRange(hsv_image, lower2, upper2)
        red_mask = lower_mask | upper_mask  # Combine both masks
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        return red_mask
    
    def _extract_feature_vector(self, image, bbox):
        """
        Extract a simple feature vector from the detected region.
        
        This creates a compact representation of the detected sign.
        For now, we use a simple approach: resize the cropped region and flatten it.
        Later, this can be replaced with more sophisticated feature extraction.
        
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
        
        # Resize to fixed size and convert to grayscale for compactness
        cropped_resized = cv2.resize(cropped, (16, 16))
        gray = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)
        
        # Flatten and normalize
        feature = gray.flatten().astype(np.float32) / 255.0
        
        return feature
    
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
                - feature_vector (numpy.ndarray): Feature representation of the detection
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
            box_mask_region = red_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
            if box_mask_region.size > 0 and not box_mask_region.any():
                confidence = confidence * 0.5  # Reduce confidence if no red detected
            
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
                    
                    # Extract feature vector
                    feature_vector = self._extract_feature_vector(frame, (x_min, y_min, x_max, y_max))
                    
                    # Create detection dictionary
                    detection = {
                        'model_type': 'traffic_sign',
                        'class_name': class_name,
                        'confidence': float(confidence * recog_confidence),  # Combined confidence
                        'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                        'feature_vector': feature_vector
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

