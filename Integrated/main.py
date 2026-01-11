"""
Integrated Traffic Perception Pipeline

This is the main pipeline that combines all four perception models:
1. Traffic Sign Detection & Recognition
2. Traffic Signal (Light) Detection
3. Road Anomaly Detection

For each frame, it produces:
- List of all detections
- Confidence summaries per model
- Annotated frame with bounding boxes

Parallel Processing:
This implementation supports thread-safe parallel inference across all models.
The system uses ThreadingLocked decorator from ultralytics (if available) or manual
threading locks to ensure thread safety when running model inferences in parallel.
Each model's detect() method is protected to prevent race conditions.

Author: Integrated Traffic Perception System
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import os
from scipy.stats import chi2
import pandas as pd
import csv

# Try to import ThreadingLocked from ultralytics for thread safety
try:
    from ultralytics.utils import ThreadingLocked
    THREADING_LOCKED_AVAILABLE = True
except ImportError:
    THREADING_LOCKED_AVAILABLE = False
    # Create a dummy decorator if not available
    def ThreadingLocked():
        def decorator(func):
            return func
        return decorator

# Import our wrapper modules
from sign_module import SignDetector
from signal_module import SignalDetector
from road_module import RoadAnomalyDetector
from risk_congestion_module import RiskCongestionDetector, draw_text_with_bg



VALID_MODELS = ['sign', 'signal', 'anomaly', 'risk_congestion']


class IntegratedTrafficPerception:
    """
    Main class that integrates all detection models.
    
    This class coordinates:
    - Loading all models (sign, signal, anomaly)
    - Processing frames through all models
    - Combining results
    - Drawing annotations (bounding boxes)
    """
    
    def __init__(self, selected_models: List[str] = None, device: str = '', use_parallel: bool = True):
        """
        Initialize detection models based on selected models.
        
        Args:
            selected_models (list[str] | None): Models to load. Valid entries:
                'sign', 'signal', 'anomaly', 'risk_congestion'. Defaults to all.
            device (str): Device to run models on ('cpu', 'cuda', or '' for auto)
            use_parallel (bool): Whether to run model inferences in parallel (default: True)
        """
        self.device = device
        self.use_parallel = use_parallel
        self.selected_models = selected_models or VALID_MODELS
        self.selected_models = [m for m in self.selected_models if m in VALID_MODELS]
        if not self.selected_models:
            self.selected_models = VALID_MODELS
        
        print("=" * 70)
        print("Initializing Integrated Traffic Perception System")
        print("=" * 70)
        
        # Load sign model if enabled
        if 'sign' in self.selected_models:
            print("\n[1/3] Loading sign detection model...")
            self.sign_detector = SignDetector(device=self.device)
        else:
            print("\n[1/3] Skipping sign detection model (disabled)")
            self.sign_detector = None
        
        # Load signal model if enabled
        # Note: SignalDetector supports backward compatibility:
        # - Can be initialized without device (uses default device='' for auto)
        # - detect() method accepts conf_threshold (defaults to 0.25)
        # This maintains compatibility with simpler initialization patterns
        if 'signal' in self.selected_models:
            print("\n[2/3] Loading signal detection model...")
            self.signal_detector = SignalDetector(device=self.device)
        else:
            print("\n[2/3] Skipping signal detection model (disabled)")
            self.signal_detector = None
        
        # Load road anomaly model if enabled
        if 'anomaly' in self.selected_models:
            print("\n[3/4] Loading road anomaly detection model...")
            self.road_detector = RoadAnomalyDetector(device=self.device)
        else:
            print("\n[3/4] Skipping road anomaly detection model (disabled)")
            self.road_detector = None
        
        # Load risk & congestion model if enabled
        if 'risk_congestion' in self.selected_models:
            print("\n[4/4] Loading risk & congestion detection model...")
            # FPS will be set later when processing video
            self.risk_congestion_detector = RiskCongestionDetector(device=self.device)
            self.risk_congestion_fps = None  # Will be set during video processing
        else:
            print("\n[4/4] Skipping risk & congestion detection model (disabled)")
            self.risk_congestion_detector = None
            self.risk_congestion_fps = None
        
        print("\n" + "=" * 70)
        print("All models loaded successfully! ✓")
        print("=" * 70)
        
        # Color scheme for drawing bounding boxes
        self.colors = {
            'traffic_sign': (0, 255, 0),      # Green
            'traffic_signal': (0, 0, 255),    # Red
            'road_anomaly': (255, 0, 0),      # Blue
            'risk_congestion': (255, 165, 0), # Orange
        }
        
        # Thread-safe detection methods using ThreadingLocked decorator
        # These ensure that only one thread accesses each model at a time
        if THREADING_LOCKED_AVAILABLE:
            self._detect_sign_threadsafe = ThreadingLocked()(self._detect_sign_impl)
            self._detect_signal_threadsafe = ThreadingLocked()(self._detect_signal_impl)
            self._detect_anomaly_threadsafe = ThreadingLocked()(self._detect_anomaly_impl)
            self._detect_risk_congestion_threadsafe = ThreadingLocked()(self._detect_risk_congestion_impl)
        else:
            # If ThreadingLocked not available, use locks manually
            self._sign_lock = threading.Lock()
            self._signal_lock = threading.Lock()
            self._anomaly_lock = threading.Lock()
            self._risk_congestion_lock = threading.Lock()
            self._detect_sign_threadsafe = self._detect_sign_with_lock
            self._detect_signal_threadsafe = self._detect_signal_with_lock
            self._detect_anomaly_threadsafe = self._detect_anomaly_with_lock
            self._detect_risk_congestion_threadsafe = self._detect_risk_congestion_with_lock
        
        # Initialize tracking for individual model outputs
        self._init_output_tracking()
    
    def _detect_sign_impl(self, frame: np.ndarray) -> List[Dict]:
        """Internal implementation for sign detection."""
        if self.sign_detector:
            return self.sign_detector.detect(frame)
        return []
    
    def _detect_signal_impl(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Internal implementation for signal detection.
        
        This method maintains backward compatibility with simpler versions:
        - conf_threshold defaults to 0.25 (matching signal_module default)
        - Returns empty list if signal_detector is None
        - Returns standardized detection format compatible with all processing methods
        """
        if self.signal_detector:
            return self.signal_detector.detect(frame, conf_threshold=conf_threshold)
        return []
    
    def _detect_anomaly_impl(self, frame: np.ndarray) -> List[Dict]:
        """Internal implementation for anomaly detection."""
        if self.road_detector:
            return self.road_detector.detect(frame)
        return []
    
    def _detect_risk_congestion_impl(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Dict:
        """Internal implementation for risk & congestion detection."""
        if self.risk_congestion_detector:
            return self.risk_congestion_detector.detect(frame, frame_id=frame_id)
        return {'detections': [], 'risk_metrics': None, 'congestion_metrics': None}
    
    def _detect_sign_with_lock(self, frame: np.ndarray) -> List[Dict]:
        """Thread-safe sign detection using manual lock."""
        with self._sign_lock:
            return self._detect_sign_impl(frame)
    
    def _detect_signal_with_lock(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        """Thread-safe signal detection using manual lock."""
        with self._signal_lock:
            return self._detect_signal_impl(frame, conf_threshold=conf_threshold)
    
    def _detect_anomaly_with_lock(self, frame: np.ndarray) -> List[Dict]:
        """Thread-safe anomaly detection using manual lock."""
        with self._anomaly_lock:
            return self._detect_anomaly_impl(frame)
    
    def _detect_risk_congestion_with_lock(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Dict:
        """Thread-safe risk & congestion detection using manual lock."""
        with self._risk_congestion_lock:
            return self._detect_risk_congestion_impl(frame, frame_id=frame_id)
    
    def _init_output_tracking(self):
        """Initialize tracking structures for individual model outputs."""
        # Signal detection tracking
        self.signal_detections_by_class = {}  # class_name -> count
        
        # Sign detection tracking
        self.sign_detections_by_class = {}  # class_name -> count
        
        # Road anomaly detection tracking
        self.anomaly_detections_by_class = {}  # class_name -> count
        
        # Risk & congestion detection tracking
        self.risk_congestion_detections_by_class = {}  # class_name -> count
        self.risk_metrics_history = []  # Store risk metrics over time
        self.congestion_metrics_history = []  # Store congestion metrics over time
        
    
    def process_frame(self, frame: np.ndarray, use_parallel: bool = None, conf_threshold: float = 0.25, frame_id: Optional[int] = None) -> Dict:
        """
        Process a single frame through all models.
        
        This is the main processing function. It:
        1. Runs all enabled models on the frame (in parallel if enabled)
        2. Collects all detections
        3. Extracts confidence summaries
        4. Draws annotations (bounding boxes)
        
        Args:
            frame (numpy.ndarray): BGR image frame
            use_parallel (bool | None): Whether to run model inferences in parallel. 
                                       If None, uses instance default (default: None)
            conf_threshold (float): Confidence threshold for detections (default: 0.25)
        
        Returns:
            dict: Complete frame analysis containing:
                - detections: List of all detection dictionaries
                - confidence_summaries: Dict with confidence lists per model
                - annotated_frame: Frame with bounding boxes drawn
                - processing_time: Time taken to process the frame in seconds
                - model_times: Dict with processing times for each individual model
        """
        # Start timing
        frame_start_time = time.time()
        
        # Use instance variable if not explicitly provided
        if use_parallel is None:
            use_parallel = self.use_parallel
        
        # Initialize model timing dictionaries
        model_times = {
            'sign_model': 0.0,
            'signal_model': 0.0,
            'anomaly_model': 0.0,
            'risk_congestion_model': 0.0
        }
        
        # Step 1: Run enabled models (in parallel if requested)
        if use_parallel and len(self.selected_models) > 1:
            # Parallel processing using ThreadPoolExecutor
            sign_detections = []
            signal_detections = []
            anomaly_detections = []
            risk_congestion_results = {'detections': [], 'risk_metrics': None, 'congestion_metrics': None}
            
            # Create a copy of the frame for each thread (to avoid potential issues)
            frame_copy = frame.copy()
            
            # Helper function to time a detection call
            def timed_detect(detector_func, frame_copy, model_name, conf_thresh=None, frame_id=None):
                start = time.time()
                if conf_thresh is not None and model_name == 'signal':
                    result = detector_func(frame_copy, conf_threshold=conf_thresh)
                elif model_name == 'risk_congestion':
                    result = detector_func(frame_copy, frame_id=frame_id)
                else:
                    result = detector_func(frame_copy)
                elapsed = time.time() - start
                return result, elapsed, model_name
            
            # Submit all detection tasks to thread pool
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                if self.sign_detector:
                    futures['sign'] = executor.submit(timed_detect, self._detect_sign_threadsafe, frame_copy, 'sign')
                if self.signal_detector:
                    futures['signal'] = executor.submit(timed_detect, self._detect_signal_threadsafe, frame_copy, 'signal', conf_thresh=conf_threshold)
                if self.road_detector:
                    futures['anomaly'] = executor.submit(timed_detect, self._detect_anomaly_threadsafe, frame_copy, 'anomaly')
                if self.risk_congestion_detector:
                    futures['risk_congestion'] = executor.submit(timed_detect, self._detect_risk_congestion_threadsafe, frame_copy, 'risk_congestion', frame_id=frame_id)
                
                # Collect results as they complete and track timing
                for model_name, future in futures.items():
                    try:
                        result, elapsed_time, _ = future.result()
                        if model_name == 'sign':
                            sign_detections = result
                            model_times['sign_model'] = elapsed_time
                        elif model_name == 'signal':
                            signal_detections = result
                            model_times['signal_model'] = elapsed_time
                        elif model_name == 'anomaly':
                            anomaly_detections = result
                            model_times['anomaly_model'] = elapsed_time
                        elif model_name == 'risk_congestion':
                            risk_congestion_results = result
                            model_times['risk_congestion_model'] = elapsed_time
                    except Exception as e:
                        print(f"Warning: Error in {model_name} detection: {e}")
                        if model_name == 'sign':
                            sign_detections = []
                        elif model_name == 'signal':
                            signal_detections = []
                        elif model_name == 'anomaly':
                            anomaly_detections = []
                        elif model_name == 'risk_congestion':
                            risk_congestion_results = {'detections': [], 'risk_metrics': None, 'congestion_metrics': None}
        else:
            # Sequential processing (fallback or when parallel disabled)
            if self.sign_detector:
                sign_start = time.time()
                sign_detections = self._detect_sign_threadsafe(frame)
                model_times['sign_model'] = time.time() - sign_start
            else:
                sign_detections = []
            
            if self.signal_detector:
                signal_start = time.time()
                signal_detections = self._detect_signal_threadsafe(frame, conf_threshold=conf_threshold)
                model_times['signal_model'] = time.time() - signal_start
            else:
                signal_detections = []
            
            if self.road_detector:
                anomaly_start = time.time()
                anomaly_detections = self._detect_anomaly_threadsafe(frame)
                model_times['anomaly_model'] = time.time() - anomaly_start
            else:
                anomaly_detections = []
            
            if self.risk_congestion_detector:
                risk_congestion_start = time.time()
                risk_congestion_results = self._detect_risk_congestion_threadsafe(frame, frame_id=frame_id)
                model_times['risk_congestion_model'] = time.time() - risk_congestion_start
            else:
                risk_congestion_results = {'detections': [], 'risk_metrics': None, 'congestion_metrics': None}
        
        # Extract risk_congestion detections
        risk_congestion_detections = risk_congestion_results.get('detections', [])
        risk_metrics = risk_congestion_results.get('risk_metrics', None)
        congestion_metrics = risk_congestion_results.get('congestion_metrics', None)
        
        # Step 2: Combine all detections into a single list
        all_detections = sign_detections + signal_detections + anomaly_detections + risk_congestion_detections
        
        # Step 3: Create confidence summaries
        sign_confidences = [det['confidence'] for det in sign_detections]
        signal_confidences = [det['confidence'] for det in signal_detections]
        anomaly_confidences = [det['confidence'] for det in anomaly_detections]
        risk_congestion_confidences = [det['confidence'] for det in risk_congestion_detections]
        
        confidence_summaries = {
            'sign_model': sign_confidences,
            'signal_model': signal_confidences,
            'anomaly_model': anomaly_confidences,
            'risk_congestion_model': risk_congestion_confidences
        }
        
        # Step 4: Create annotated frame (with risk/PCU overlays if available)
        annotated_frame = self._draw_annotations(
            frame.copy(), 
            all_detections,
            risk_metrics=risk_metrics,
            congestion_metrics=congestion_metrics
        )
        
        # Step 5: Calculate processing time
        processing_time = time.time() - frame_start_time
        
        # Step 6: Compile results
        # Note: Includes both combined format (for new features) and separate lists (for backward compatibility)
        results = {
            # Combined format (current version)
            'detections': all_detections,
            'confidence_summaries': confidence_summaries,
            'annotated_frame': annotated_frame,
            'processing_time': processing_time,
            'model_times': model_times,
            # Separate lists (backward compatibility with simpler version)
            'sign_detections': sign_detections,
            'signal_detections': signal_detections,
            'anomaly_detections': anomaly_detections,
            'risk_congestion_detections': risk_congestion_detections,
            # Risk and congestion metrics
            'risk_metrics': risk_metrics,
            'congestion_metrics': congestion_metrics
        }
        
        return results
    
    def _save_detections_to_csv(self, detections: List[Dict], source_name: str, output_path: Path, 
                                append: bool = False, frame_number: int = None):
        """
        Save all detections to CSV file.
        
        Args:
            detections: List of detection dictionaries
            source_name: Name of the source image/video/frame
            output_path: Path object for the output CSV file
            append: Whether to append to existing CSV (default: False)
            frame_number: Optional frame number for video frames
        """
        if not detections:
            return
        
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare CSV data
            csv_data = []
            for det in detections:
                x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
                row = {
                    'Source': source_name,
                    'Frame': frame_number if frame_number is not None else '',
                    'Model Type': det.get('model_type', 'unknown'),
                    'Class Name': det.get('class_name', 'unknown'),
                    'Confidence': f"{det.get('confidence', 0.0):.4f}",
                    'Bbox X1': int(x1),
                    'Bbox Y1': int(y1),
                    'Bbox X2': int(x2),
                    'Bbox Y2': int(y2)
                }
                csv_data.append(row)
            
            # Write to CSV
            fieldnames = ['Source', 'Frame', 'Model Type', 'Class Name', 'Confidence', 
                         'Bbox X1', 'Bbox Y1', 'Bbox X2', 'Bbox Y2']
            file_exists = output_path.is_file() and append
            
            mode = 'a' if append else 'w'
            with open(output_path, mode=mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(csv_data)
            
            return True
        except Exception as e:
            print(f"Warning: Failed to save detections to CSV: {e}")
            return False
    
    def _print_individual_model_outputs(self, signal_detections_by_class, sign_detections_by_class, 
                                       anomaly_detections_by_class, risk_congestion_detections_by_class,
                                       frame_count):
        """
        Print test outputs from individual models, similar to their standalone test outputs.
        
        Args:
            signal_detections_by_class: Dict of class_name -> count for signal detections
            sign_detections_by_class: Dict of class_name -> count for sign detections
            anomaly_detections_by_class: Dict of class_name -> count for anomaly detections
            risk_congestion_detections_by_class: Dict of class_name -> count for risk_congestion detections
            frame_count: Total number of frames processed
        """
        print(f"\n{'='*70}")
        print("INDIVIDUAL MODEL TEST OUTPUTS")
        print(f"{'='*70}")
        
        # Signal Detection Model Outputs
        if signal_detections_by_class or 'signal' in self.selected_models:
            print(f"\n[1] SIGNAL DETECTION MODEL OUTPUTS (from signal_det):")
            print(f"  {'Class':<20} {'Count':<10}")
            print(f"  {'-'*20} {'-'*10}")
            if signal_detections_by_class:
                total_signal = sum(signal_detections_by_class.values())
                for class_name, count in sorted(signal_detections_by_class.items()):
                    print(f"  {class_name:<20} {count:<10}")
                print(f"  {'TOTAL':<20} {total_signal:<10}")
            else:
                print(f"  {'No detections':<20} {'0':<10}")
            print(f"  Format: Similar to signal_det/signal_det/val.py output")
            print(f"  (Precision/Recall/mAP would require ground truth labels)")
        
        # Sign Detection Model Outputs
        if sign_detections_by_class or 'sign' in self.selected_models:
            print(f"\n[2] SIGN DETECTION MODEL OUTPUTS (from sign_det):")
            print(f"  {'Class':<20} {'Count':<10}")
            print(f"  {'-'*20} {'-'*10}")
            if sign_detections_by_class:
                total_sign = sum(sign_detections_by_class.values())
                for class_name, count in sorted(sign_detections_by_class.items()):
                    print(f"  {class_name:<20} {count:<10}")
                print(f"  {'TOTAL':<20} {total_sign:<10}")
            else:
                print(f"  {'No detections':<20} {'0':<10}")
            print(f"  Format: Similar to sign_det/sign_det/pipeline.py output")
        
        # Speedbump/Pothole Detection Model Outputs
        if anomaly_detections_by_class or 'anomaly' in self.selected_models:
            print(f"\n[3] SPEEDBUMP/POTHOLE DETECTION MODEL OUTPUTS (from speedbump_pothole_det):")
            print(f"  {'Class':<20} {'Count':<10}")
            print(f"  {'-'*20} {'-'*10}")
            if anomaly_detections_by_class:
                total_anomaly = sum(anomaly_detections_by_class.values())
                for class_name, count in sorted(anomaly_detections_by_class.items()):
                    print(f"  {class_name:<20} {count:<10}")
                print(f"  {'TOTAL':<20} {total_anomaly:<10}")
            else:
                print(f"  {'No detections':<20} {'0':<10}")
            print(f"  Format: Similar to speedbump_pothole_det/speedbump_pothole_det/val.py output")
            print(f"  (Precision/Recall/mAP would require ground truth labels)")
        
        # Risk & Congestion Detection Model Outputs
        if risk_congestion_detections_by_class or 'risk_congestion' in self.selected_models:
            print(f"\n[4] RISK & CONGESTION DETECTION MODEL OUTPUTS (from driving_risk_and_congestion):")
            print(f"  {'Class':<20} {'Count':<10}")
            print(f"  {'-'*20} {'-'*10}")
            if risk_congestion_detections_by_class:
                total_risk_congestion = sum(risk_congestion_detections_by_class.values())
                for class_name, count in sorted(risk_congestion_detections_by_class.items()):
                    print(f"  {class_name:<20} {count:<10}")
                print(f"  {'TOTAL':<20} {total_risk_congestion:<10}")
            else:
                print(f"  {'No detections':<20} {'0':<10}")
            print(f"  Format: Similar to driving_risk_and_congestion/src/demoidd_riskest_cong.py output")
        
        print(f"\n{'='*70}")
    
    def _draw_annotations(self, frame: np.ndarray, detections: List[Dict], 
                         risk_metrics: Optional[Dict] = None, 
                         congestion_metrics: Optional[Dict] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        Also overlays risk labels and PCU values if available (matching original model).
        
        Args:
            frame (numpy.ndarray): Original frame
            detections (list): List of detection dictionaries
            risk_metrics (dict | None): Risk metrics dict with 'risk_label' and 'smoothed_label'
            congestion_metrics (dict | None): Congestion metrics dict with 'pcu_sum'
        
        Returns:
            numpy.ndarray: Annotated frame
        """
        annotated = frame.copy()
        
        # Draw bounding boxes and labels for all detections
        for det in detections:
            # Get bounding box coordinates and ensure they are integers
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Get color based on model type
            model_type = det['model_type']
            color = self.colors.get(model_type, (255, 255, 255))  # Default white
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            class_name = det['class_name']
            confidence = det['confidence']
            label = f"{det['model_type']}: {class_name} ({confidence:.2f})"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Get color for label background
            model_type = det['model_type']
            color = self.colors.get(model_type, (255, 255, 255))
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
        
        # Add risk and PCU overlays (matching original demoidd_video_riskest_cong.py)
        if risk_metrics or congestion_metrics:
            orig_height, orig_width = annotated.shape[:2]
            x0 = int(0.7 * orig_width)
            y0 = int(0.1 * orig_height)
            
            # Risk overlay (ONLY if smoothed label is available)
            if risk_metrics and risk_metrics.get('smoothed_label'):
                smooth_label = risk_metrics['smoothed_label']
                color_map = {
                    'LOW': (0, 255, 0),        # Green
                    'MODERATE': (0, 165, 255),  # Orange
                    'HIGH': (0, 0, 255),       # Red
                    'CRITICAL': (0, 0, 139)     # Dark Red
                }
                
                risk_color = color_map.get(smooth_label, (255, 255, 255))
                
                cv2.putText(
                    annotated,
                    f"Risk: {smooth_label}",
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    risk_color,
                    10,
                    cv2.LINE_AA
                )
                
                pcu_y = y0 + 80
            else:
                # If risk not available yet, place PCU at default position
                pcu_y = y0
            
            # PCU overlay (ALWAYS shown if available)
            if congestion_metrics and congestion_metrics.get('pcu_sum') is not None:
                pc_sum = congestion_metrics['pcu_sum']
                draw_text_with_bg(
                    annotated,
                    f"PCU: {pc_sum:.2f}",
                    (x0, pcu_y),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=1.2,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    thickness=3,
                    padding=12
                )
        
        return annotated
    
    def process_image(self, image_path: str, output_path: str = None, show: bool = True, 
                     label_path: str = None, iou_threshold: float = 0.5, conf_threshold: float = 0.4):
        """
        Process a single image file.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image (optional). Can be a directory or full file path.
            show (bool): Whether to display the result
            label_path (str): Optional path to ground truth label file (YOLO format) for metrics calculation
            iou_threshold (float): IoU threshold for matching predictions to ground truth (default: 0.5)
            conf_threshold (float): Confidence threshold for predictions (default: 0.4)
        """
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        h, w = frame.shape[:2]
        
        # Process frame
        results = self.process_frame(frame)
        
        # Calculate metrics if ground truth is available
        metrics = None
        if label_path and Path(label_path).exists():
            gt_boxes = self._load_yolo_labels(Path(label_path), w, h)
            pred_detections = [d for d in results['detections'] 
                             if d['confidence'] >= conf_threshold]
            
            # Convert predictions to format for matching
            pred_boxes = []
            for det in pred_detections:
                x1, y1, x2, y2 = det['bbox']
                pred_boxes.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': det['class_name'],
                    'model_type': det['model_type'],
                    'confidence': det['confidence']
                })
            
            # Match predictions to ground truth
            matches, unmatched_preds, unmatched_gts = self._match_detections(
                pred_boxes, gt_boxes, iou_threshold
            )
            
            # Calculate overall metrics
            total_tp = len(matches)
            total_fp = len(unmatched_preds)
            total_fn = len(unmatched_gts)
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
            
            # Calculate per-model metrics
            model_stats = {
                'traffic_sign': {'tp': 0, 'fp': 0, 'fn': 0},
                'traffic_signal': {'tp': 0, 'fp': 0, 'fn': 0},
                'road_anomaly': {'tp': 0, 'fp': 0, 'fn': 0}
            }
            
            for match in matches:
                model_type = match['pred']['model_type']
                if model_type in model_stats:
                    model_stats[model_type]['tp'] += 1
            
            for unmatched_pred in unmatched_preds:
                model_type = unmatched_pred['model_type']
                if model_type in model_stats:
                    model_stats[model_type]['fp'] += 1
            
            for unmatched_gt in unmatched_gts:
                model_type = self._infer_model_type_from_gt(unmatched_gt)
                if model_type in model_stats:
                    model_stats[model_type]['fn'] += 1
            
            per_model_metrics = {}
            for model_type, stats in model_stats.items():
                tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                per_model_metrics[model_type] = {
                    'precision': p,
                    'recall': r,
                    'f1_score': f1,
                    'accuracy': acc,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'per_model_metrics': per_model_metrics
            }
        
        # Prepare for CSV output
        image_name = Path(image_path).name
        
        # Print summary
        print(f"\nDetection Summary:")
        print(f"  Traffic Signs: {len([d for d in results['detections'] if d['model_type'] == 'traffic_sign'])}")
        print(f"  Traffic Signals: {len([d for d in results['detections'] if d['model_type'] == 'traffic_signal'])}")
        print(f"  Road Anomalies: {len([d for d in results['detections'] if d['model_type'] == 'road_anomaly'])}")
        print(f"  Total Detections: {len(results['detections'])}")
        print(f"\nProcessing Time:")
        print(f"  Integrated Model: {results['processing_time']:.4f} seconds ({results['processing_time']*1000:.2f} ms)")
        if results.get('model_times'):
            print(f"  Sign Model: {results['model_times']['sign_model']:.4f} seconds ({results['model_times']['sign_model']*1000:.2f} ms)")
            print(f"  Signal Model: {results['model_times']['signal_model']:.4f} seconds ({results['model_times']['signal_model']*1000:.2f} ms)")
            print(f"  Anomaly Model: {results['model_times']['anomaly_model']:.4f} seconds ({results['model_times']['anomaly_model']*1000:.2f} ms)")
        # Save if requested
        final_output_path = None
        output_path_obj = None
        if output_path:
            output_path_obj = Path(output_path)
            
            # Determine if output_path is a directory or file path
            # If it's an existing directory, or has no extension and doesn't exist as a file, treat as directory
            is_directory = (
                output_path_obj.is_dir() or 
                (not output_path_obj.suffix and not output_path_obj.is_file())
            )
            
            if is_directory:
                # Create directory if it doesn't exist
                output_path_obj.mkdir(parents=True, exist_ok=True)
                # Generate output filename based on input filename
                input_path = Path(image_path)
                output_filename = f"annotated_{input_path.stem}{input_path.suffix}"
                final_output_path = output_path_obj / output_filename
            else:
                # It's a file path, create parent directory if needed
                final_output_path = output_path_obj
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the image
            annotated_frame = results['annotated_frame']
            if annotated_frame is None or annotated_frame.size == 0:
                print(f"\nError: Annotated frame is empty or None")
            else:
                # Ensure the path uses forward slashes or proper Windows path
                output_path_str = str(final_output_path).replace('\\', '/')
                success = cv2.imwrite(output_path_str, annotated_frame)
                if success:
                    print(f"\nAnnotated image saved to: {final_output_path}")
                else:
                    print(f"\nError: Failed to save image to: {final_output_path}")
                    print(f"  Check if the directory exists and you have write permissions")
                    print(f"  Attempted path: {final_output_path.absolute()}")
            
            # Save all detections to CSV
            if results['detections']:
                try:
                    # Determine output CSV path
                    if is_directory:
                        csv_path = output_path_obj / "detections.csv"
                    else:
                        csv_path = final_output_path.parent / "detections.csv"
                    
                    # Append if CSV file already exists (for directory processing)
                    csv_exists = csv_path.is_file()
                    
                    # Save all detections
                    success = self._save_detections_to_csv(
                        results['detections'],
                        image_name,
                        csv_path,
                        append=csv_exists
                    )
                    
                    if success:
                        print(f"Saved all detections to CSV: {csv_path}")
                        print(f"  Total detections: {len(results['detections'])}")
                        print(f"    - Traffic Signs: {len([d for d in results['detections'] if d['model_type'] == 'traffic_sign'])}")
                        print(f"    - Traffic Signals: {len([d for d in results['detections'] if d['model_type'] == 'traffic_signal'])}")
                        print(f"    - Road Anomalies: {len([d for d in results['detections'] if d['model_type'] == 'road_anomaly'])}")
                except Exception as e:
                    print(f"Warning: Failed to save detections CSV file: {e}")
        
        # Show if requested
        if show:
            cv2.imshow('Integrated Traffic Perception', results['annotated_frame'])
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Final Results Summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS - IMAGE PROCESSING")
        print(f"{'='*70}")
        print(f"\nTIMING METRICS (Images):")
        print(f"  Integrated Model - Average time per frame: {results.get('processing_time', 0.0):.4f} seconds ({results.get('processing_time', 0.0)*1000:.2f} ms)")
        
        model_times = results.get('model_times', {})
        print(f"  Sign Model - Average time per frame: {model_times.get('sign_model', 0.0):.4f} seconds ({model_times.get('sign_model', 0.0)*1000:.2f} ms)")
        print(f"  Signal Model - Average time per frame: {model_times.get('signal_model', 0.0):.4f} seconds ({model_times.get('signal_model', 0.0)*1000:.2f} ms)")
        print(f"  Anomaly Model - Average time per frame: {model_times.get('anomaly_model', 0.0):.4f} seconds ({model_times.get('anomaly_model', 0.0)*1000:.2f} ms)")
        if 'risk_congestion_model' in model_times:
            print(f"  Risk & Congestion Model - Average time per frame: {model_times.get('risk_congestion_model', 0.0):.4f} seconds ({model_times.get('risk_congestion_model', 0.0)*1000:.2f} ms)")
        
        if metrics:
            print(f"\nPERFORMANCE METRICS (with ground truth):")
            print(f"\n  Integrated Model (All models combined):")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1 Score:  {metrics['f1_score']:.4f}")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
            
            print(f"\n  Individual Models:")
            for model_type, model_metrics in metrics['per_model_metrics'].items():
                print(f"    {model_type}:")
                print(f"      Precision: {model_metrics['precision']:.4f}")
                print(f"      Recall:    {model_metrics['recall']:.4f}")
                print(f"      F1 Score:  {model_metrics['f1_score']:.4f}")
                print(f"      Accuracy:  {model_metrics['accuracy']:.4f}")
                print(f"      TP: {model_metrics['tp']}, FP: {model_metrics['fp']}, FN: {model_metrics['fn']}")
        else:
            print(f"\nPERFORMANCE METRICS:")
            print(f"  (Ground truth labels not provided - precision/recall/F1/accuracy not calculated)")
        
        print(f"\nTotal detections: {len(results['detections'])}")
        
        # Print individual model test outputs for image
        signal_detections_by_class = {}
        sign_detections_by_class = {}
        anomaly_detections_by_class = {}
        risk_congestion_detections_by_class = {}
        
        for det in results['detections']:
            model_type = det.get('model_type', 'unknown')
            class_name = det.get('class_name', 'unknown')
            
            if model_type == 'traffic_signal':
                signal_detections_by_class[class_name] = signal_detections_by_class.get(class_name, 0) + 1
            elif model_type == 'traffic_sign':
                sign_detections_by_class[class_name] = sign_detections_by_class.get(class_name, 0) + 1
            elif model_type == 'road_anomaly':
                anomaly_detections_by_class[class_name] = anomaly_detections_by_class.get(class_name, 0) + 1
            elif model_type == 'risk_congestion':
                risk_congestion_detections_by_class[class_name] = risk_congestion_detections_by_class.get(class_name, 0) + 1
        
        self._print_individual_model_outputs(
            signal_detections_by_class,
            sign_detections_by_class,
            anomaly_detections_by_class,
            risk_congestion_detections_by_class,
            1
        )
        
        print(f"{'='*70}\n")
        
        return results, metrics
    
    def process_video(self, video_path: str, output_path: str = None, show: bool = True,
                     labels_dir: str = None, iou_threshold: float = 0.5, conf_threshold: float = 0.4):
        """
        Process a video file frame by frame.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save annotated video (optional). Can be a directory or full file path.
            show (bool): Whether to display the result
            labels_dir (str): Optional path to directory containing ground truth label files (YOLO format) for metrics calculation
            iou_threshold (float): IoU threshold for matching predictions to ground truth (default: 0.5)
            conf_threshold (float): Confidence threshold for predictions (default: 0.4)
        
        Returns:
            dict: Processing results containing:
                - frame_count (int): Total number of frames processed
                - elapsed_time (float): Total elapsed time in seconds
                - avg_frame_time (float): Average processing time per frame in seconds
                - avg_model_times (dict): Average processing times per model (seconds)
                - frame_times (list[float]): Per-frame processing times in seconds
                - model_times_per_frame (list[dict]): Per-frame model times. Each dict contains:
                    {'sign_model': float, 'signal_model': float, 'anomaly_model': float, 'risk_congestion_model': float}
                - metrics (dict): Detection metrics if labels provided
                - risk_level_metrics (dict): Risk level metrics if available
        """
        print(f"\nProcessing video: {video_path}")
        
        # Extract ground truth risk level from filename (matching riskest_vid_analysis.py)
        video_name_upper = Path(video_path).stem.upper()
        risk_level_ground_truth = None
        if video_name_upper.startswith('LOW'):
            risk_level_ground_truth = 'LOW'
        elif video_name_upper.startswith('MODERATE'):
            risk_level_ground_truth = 'MODERATE'
        elif video_name_upper.startswith('HIGH'):
            risk_level_ground_truth = 'HIGH'
        elif video_name_upper.startswith('CRITICAL'):
            risk_level_ground_truth = 'CRITICAL'
        
        # Track raw and smoothed risk labels for metrics calculation
        raw_risk_labels = []
        smoothed_risk_labels = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS ({total_frames} frames)")
        
        # Update risk_congestion detector with FPS for proper window size calculation
        if self.risk_congestion_detector and fps > 0:
            self.risk_congestion_fps = fps
            # Reinitialize with FPS to set correct window size
            self.risk_congestion_detector.set_smoothing_params(
                window_size=int(fps * 2),  # Match original: window_size = int(ip_fps * 2)
                decay_beta=0.001
            )
        
        # Setup video writer if saving
        writer = None
        final_output_path = None
        if output_path:
            output_path_obj = Path(output_path)
            
            # Determine if output_path is a directory or file path
            # If it's an existing directory, or has no extension and doesn't exist as a file, treat as directory
            is_directory = (
                output_path_obj.is_dir() or 
                (not output_path_obj.suffix and not output_path_obj.is_file())
            )
            
            if is_directory:
                # Create directory if it doesn't exist
                output_path_obj.mkdir(parents=True, exist_ok=True)
                # Generate output filename based on input filename
                input_path = Path(video_path)
                output_filename = f"annotated_{input_path.stem}{input_path.suffix}"
                final_output_path = output_path_obj / output_filename
            else:
                # It's a file path, create parent directory if needed
                final_output_path = output_path_obj
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(final_output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Will save output to: {final_output_path}")
            else:
                print(f"Warning: Failed to initialize video writer for: {final_output_path}")
                writer = None
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        frame_times = []  # Track processing time for each frame
        model_times_list = []  # Track individual model times for each frame
        
        # Metrics tracking (if labels available)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        model_stats = {
            'traffic_sign': {'tp': 0, 'fp': 0, 'fn': 0},
            'traffic_signal': {'tp': 0, 'fp': 0, 'fn': 0},
            'road_anomaly': {'tp': 0, 'fp': 0, 'fn': 0},
            'risk_congestion': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        labels_path = Path(labels_dir) if labels_dir and Path(labels_dir).exists() else None
        video_name = Path(video_path).stem
        
        # Detection statistics (available without ground truth)
        total_detections = 0
        detection_counts = {
            'traffic_sign': 0,
            'traffic_signal': 0,
            'road_anomaly': 0,
            'risk_congestion': 0
        }
        confidence_scores = {
            'traffic_sign': [],
            'traffic_signal': [],
            'road_anomaly': [],
            'risk_congestion': []
        }
        class_counts = {}  # Track counts per class name
        
        # Individual model output tracking
        signal_detections_by_class = {}  # class_name -> count
        sign_detections_by_class = {}  # class_name -> count
        anomaly_detections_by_class = {}  # class_name -> count
        risk_congestion_detections_by_class = {}  # class_name -> count
        all_detections_for_csv = []  # List of all detections for CSV output
        
        # Track risk and congestion metrics for original format CSV (matching demoidd_video_riskest_cong.py)
        risk_congestion_frame_data = []  # List of dicts with frame-by-frame risk/congestion data
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            results = self.process_frame(frame, frame_id=frame_count)
            
            # Track frame processing time
            frame_times.append(results['processing_time'])
            
            # Track individual model times
            if results.get('model_times'):
                model_times_list.append(results['model_times'])
            
            # Track detection statistics (available without ground truth)
            detections = results.get('detections', [])
            total_detections += len(detections)
            
            for det in detections:
                model_type = det.get('model_type', 'unknown')
                class_name = det.get('class_name', 'unknown')
                confidence = det.get('confidence', 0.0)
                
                # Count by model type
                if model_type in detection_counts:
                    detection_counts[model_type] += 1
                    confidence_scores[model_type].append(confidence)
                
                # Count by class name
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                # Track individual model outputs
                if model_type == 'traffic_signal':
                    if class_name not in signal_detections_by_class:
                        signal_detections_by_class[class_name] = 0
                    signal_detections_by_class[class_name] += 1
                
                elif model_type == 'traffic_sign':
                    if class_name not in sign_detections_by_class:
                        sign_detections_by_class[class_name] = 0
                    sign_detections_by_class[class_name] += 1
                
                elif model_type == 'road_anomaly':
                    if class_name not in anomaly_detections_by_class:
                        anomaly_detections_by_class[class_name] = 0
                    anomaly_detections_by_class[class_name] += 1
                
                elif model_type == 'risk_congestion':
                    if class_name not in risk_congestion_detections_by_class:
                        risk_congestion_detections_by_class[class_name] = 0
                    risk_congestion_detections_by_class[class_name] += 1
                
                # Collect all detections for CSV output
                video_name = Path(video_path).stem
                all_detections_for_csv.append({
                    'detection': det,
                    'source_name': video_name,
                    'frame_number': frame_count
                })
            
            # Calculate metrics if ground truth is available
            if labels_path:
                # Try to find corresponding label file (frame_XXXXX.txt or similar naming)
                # Common patterns: frame_00001.txt, 00001.txt, etc.
                label_file = None
                for pattern in [f"{video_name}_frame_{frame_count:05d}.txt", 
                               f"frame_{frame_count:05d}.txt",
                               f"{frame_count:05d}.txt",
                               f"{video_name}_{frame_count:05d}.txt"]:
                    potential_label = labels_path / pattern
                    if potential_label.exists():
                        label_file = potential_label
                        break
                
                if label_file and label_file.exists():
                    h, w = frame.shape[:2]
                    gt_boxes = self._load_yolo_labels(label_file, w, h)
                    pred_detections = [d for d in results['detections'] 
                                     if d['confidence'] >= conf_threshold]
                    
                    # Convert predictions to format for matching
                    pred_boxes = []
                    for det in pred_detections:
                        x1, y1, x2, y2 = det['bbox']
                        pred_boxes.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': det['class_name'],
                            'model_type': det['model_type'],
                            'confidence': det['confidence']
                        })
                    
                    # Match predictions to ground truth
                    matches, unmatched_preds, unmatched_gts = self._match_detections(
                        pred_boxes, gt_boxes, iou_threshold
                    )
                    
                    # Update statistics
                    total_tp += len(matches)
                    total_fp += len(unmatched_preds)
                    total_fn += len(unmatched_gts)
                    
                    # Update per-model statistics
                    for match in matches:
                        model_type = match['pred']['model_type']
                        if model_type in model_stats:
                            model_stats[model_type]['tp'] += 1
                    
                    for unmatched_pred in unmatched_preds:
                        model_type = unmatched_pred['model_type']
                        if model_type in model_stats:
                            model_stats[model_type]['fp'] += 1
                    
                    for unmatched_gt in unmatched_gts:
                        model_type = self._infer_model_type_from_gt(unmatched_gt)
                    if model_type in model_stats:
                        model_stats[model_type]['fn'] += 1
            
            # Track risk labels if available
            risk_metrics = results.get('risk_metrics')
            congestion_metrics = results.get('congestion_metrics')
            
            if risk_metrics or congestion_metrics:
                # Store frame data for original format CSV
                frame_data = {'Frame Number': frame_count}
                
                if risk_metrics:
                    raw_label = risk_metrics.get('risk_label')
                    smoothed_label = risk_metrics.get('smoothed_label')
                    if raw_label:
                        raw_risk_labels.append(raw_label)
                        # Always append smoothed label (can be None during warm-up)
                        smoothed_risk_labels.append(smoothed_label if smoothed_label else None)
                    
                    # Add risk metrics to frame data
                    frame_data.update({
                        'c0': risk_metrics.get('counts', [0]*5)[0],
                        'c1': risk_metrics.get('counts', [0]*5)[1],
                        'c2': risk_metrics.get('counts', [0]*5)[2],
                        'c3': risk_metrics.get('counts', [0]*5)[3],
                        'c4': risk_metrics.get('counts', [0]*5)[4],
                        'a0': risk_metrics.get('areas', [0]*5)[0],
                        'a1': risk_metrics.get('areas', [0]*5)[1],
                        'a2': risk_metrics.get('areas', [0]*5)[2],
                        'a3': risk_metrics.get('areas', [0]*5)[3],
                        'a4': risk_metrics.get('areas', [0]*5)[4],
                        'd0': risk_metrics.get('x_disps', [0]*5)[0],
                        'd1': risk_metrics.get('x_disps', [0]*5)[1],
                        'd2': risk_metrics.get('x_disps', [0]*5)[2],
                        'd3': risk_metrics.get('x_disps', [0]*5)[3],
                        'd4': risk_metrics.get('x_disps', [0]*5)[4],
                        'b0': risk_metrics.get('combined', [0]*5)[0],
                        'b1': risk_metrics.get('combined', [0]*5)[1],
                        'b2': risk_metrics.get('combined', [0]*5)[2],
                        'b3': risk_metrics.get('combined', [0]*5)[3],
                        'b4': risk_metrics.get('combined', [0]*5)[4],
                        'B_sum': risk_metrics.get('B_sum', 0.0),
                        'Raw Label': raw_label if raw_label else '',
                        'Smoothed Label': smoothed_label if smoothed_label else ''
                    })
                else:
                    # Fill with zeros if no risk metrics
                    frame_data.update({
                        'c0': 0, 'c1': 0, 'c2': 0, 'c3': 0, 'c4': 0,
                        'a0': 0, 'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0,
                        'd0': 0, 'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0,
                        'b0': 0, 'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0,
                        'B_sum': 0.0,
                        'Raw Label': '',
                        'Smoothed Label': ''
                    })
                
                if congestion_metrics:
                    # Add PCU metrics to frame data
                    pcu_counts = congestion_metrics.get('counts', [0]*5)
                    pcu_values = congestion_metrics.get('pcu_per_class', [0]*5)
                    frame_data.update({
                        'pc_c0': pcu_counts[0],
                        'pc_c1': pcu_counts[1],
                        'pc_c2': pcu_counts[2],
                        'pc_c3': pcu_counts[3],
                        'pc_c4': pcu_counts[4],
                        'pc_p0': pcu_values[0],
                        'pc_p1': pcu_values[1],
                        'pc_p2': pcu_values[2],
                        'pc_p3': pcu_values[3],
                        'pc_p4': pcu_values[4],
                        'PCU_Sum': congestion_metrics.get('pcu_sum', 0.0)
                    })
                else:
                    # Fill with zeros if no congestion metrics
                    frame_data.update({
                        'pc_c0': 0, 'pc_c1': 0, 'pc_c2': 0, 'pc_c3': 0, 'pc_c4': 0,
                        'pc_p0': 0.0, 'pc_p1': 0.0, 'pc_p2': 0.0, 'pc_p3': 0.0, 'pc_p4': 0.0,
                        'PCU_Sum': 0.0
                    })
                
                risk_congestion_frame_data.append(frame_data)
            
            # Show progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                avg_frame_time = sum(frame_times) / len(frame_times)
                print(f"Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} FPS, avg: {avg_frame_time*1000:.2f} ms/frame)")
            
            # Save frame if writing
            if writer:
                writer.write(results['annotated_frame'])
            
            # Show frame if requested
            if show:
                cv2.imshow('Integrated Traffic Perception', results['annotated_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user (press 'q')")
                    break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            if final_output_path:
                print(f"\nAnnotated video saved to: {final_output_path}")
        if show:
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        
        # Calculate average model times
        avg_model_times = {
            'sign_model': 0.0,
            'signal_model': 0.0,
            'anomaly_model': 0.0
        }
        if model_times_list:
            for model_name in avg_model_times.keys():
                times = [mt.get(model_name, 0.0) for mt in model_times_list if model_name in mt]
                if times:
                    avg_model_times[model_name] = sum(times) / len(times)
        
        # Calculate risk level metrics if ground truth is available
        total = frame_count
        raw_accuracy = 0.0
        smooth_accuracy = 0.0
        raw_correct = 0
        smooth_correct = 0
        raw_total = 0
        smooth_total = 0
        raw_errors = 0
        smooth_errors = 0
        b = 0  # Frames improved by smoothing
        c = 0  # Frames worsened by smoothing
        mcnemar_stat = np.nan
        p_value = np.nan
        interpretation = "No ground truth available"
        raw_prediction_counts = {}
        smooth_prediction_counts = {}
        raw_macro_precision = 0.0
        raw_macro_recall = 0.0
        raw_macro_f1 = 0.0
        raw_micro_precision = 0.0
        raw_micro_recall = 0.0
        raw_micro_f1 = 0.0
        raw_per_class_metrics = {}
        smooth_macro_precision = 0.0
        smooth_macro_recall = 0.0
        smooth_macro_f1 = 0.0
        smooth_micro_precision = 0.0
        smooth_micro_recall = 0.0
        smooth_micro_f1 = 0.0
        smooth_per_class_metrics = {}
        
        if risk_level_ground_truth and raw_risk_labels:
            # Filter out None values for smoothed labels
            valid_indices = [i for i, lbl in enumerate(smoothed_risk_labels) if lbl is not None]
            
            # Calculate raw metrics
            raw_total = len(raw_risk_labels)
            raw_correct = sum(1 for lbl in raw_risk_labels if lbl == risk_level_ground_truth)
            raw_accuracy = raw_correct / raw_total if raw_total > 0 else 0.0
            raw_errors = raw_total - raw_correct
            
            # Calculate smoothed metrics (only for valid frames)
            smooth_total = len(valid_indices)
            if smooth_total > 0:
                smooth_labels_valid = [smoothed_risk_labels[i] for i in valid_indices]
                raw_labels_valid = [raw_risk_labels[i] for i in valid_indices]
                
                smooth_correct = sum(1 for lbl in smooth_labels_valid if lbl == risk_level_ground_truth)
                smooth_accuracy = smooth_correct / smooth_total if smooth_total > 0 else 0.0
                smooth_errors = smooth_total - smooth_correct
                
                # Calculate b and c (McNemar test components)
                b = sum(1 for i in range(len(valid_indices)) 
                       if raw_labels_valid[i] != risk_level_ground_truth 
                       and smooth_labels_valid[i] == risk_level_ground_truth)
                c = sum(1 for i in range(len(valid_indices))
                       if raw_labels_valid[i] == risk_level_ground_truth
                       and smooth_labels_valid[i] != risk_level_ground_truth)
                
                # McNemar test
                if (b + c) > 0:
                    mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
                    p_value = chi2.sf(mcnemar_stat, df=1)
                    
                    # Interpretation
                    if np.isnan(p_value):
                        interpretation = "No test (b+c=0)"
                    elif p_value < 0.05:
                        if b > c:
                            interpretation = "Improved significantly"
                        elif c > b:
                            interpretation = "Worsened significantly"
                        else:
                            interpretation = "Changed significantly (tie)"
                    else:
                        interpretation = "No significant change"
                else:
                    interpretation = "No test (b+c=0)"
            else:
                smooth_accuracy = 0.0
                smooth_correct = 0
                smooth_errors = 0
                interpretation = "No smoothed labels available"
            
            # Count predictions
            for lbl in raw_risk_labels:
                raw_prediction_counts[lbl] = raw_prediction_counts.get(lbl, 0) + 1
            for lbl in smoothed_risk_labels:
                if lbl is not None:
                    smooth_prediction_counts[lbl] = smooth_prediction_counts.get(lbl, 0) + 1
        
        # Only create risk_level_metrics if we have ground truth
        risk_level_metrics = None
        if risk_level_ground_truth:
            risk_level_metrics = {
                'ground_truth': risk_level_ground_truth,
                'frames': total,
                'raw_accuracy': raw_accuracy,
                'smooth_accuracy': smooth_accuracy,
                'raw_correct': raw_correct,
                'smooth_correct': smooth_correct,
                'raw_total': raw_total,
                'smooth_total': smooth_total,
                'raw_errors': raw_errors,
                'smooth_errors': smooth_errors,
                'errors_before': raw_errors,  # Matches riskest_vid_analysis.py naming
                'errors_after': smooth_errors,  # Matches riskest_vid_analysis.py naming
                'b': b,  # Frames improved by smoothing
                'c': c,  # Frames worsened by smoothing
                'mcnemar_stat': mcnemar_stat,
                'p_value': p_value,
                'interpretation': interpretation,
                'raw_prediction_counts': raw_prediction_counts,
                'smooth_prediction_counts': smooth_prediction_counts,
                'raw_most_common': max(raw_prediction_counts.items(), key=lambda x: x[1])[0] if raw_prediction_counts else None,
                'smooth_most_common': max(smooth_prediction_counts.items(), key=lambda x: x[1])[0] if smooth_prediction_counts else None,
                # Precision, Recall, F1 metrics (simplified - would need confusion matrix for proper calculation)
                'raw_macro_precision': raw_macro_precision,
                'raw_macro_recall': raw_macro_recall,
                'raw_macro_f1': raw_macro_f1,
                'raw_micro_precision': raw_micro_precision,
                'raw_micro_recall': raw_micro_recall,
                'raw_micro_f1': raw_micro_f1,
                'raw_per_class_metrics': raw_per_class_metrics,
                'smooth_macro_precision': smooth_macro_precision,
                'smooth_macro_recall': smooth_macro_recall,
                'smooth_macro_f1': smooth_macro_f1,
                'smooth_micro_precision': smooth_micro_precision,
                'smooth_micro_recall': smooth_micro_recall,
                'smooth_micro_f1': smooth_micro_f1,
                'smooth_per_class_metrics': smooth_per_class_metrics
            }
        
        # Calculate metrics if ground truth was available
        metrics = None
        if labels_path:
            # Calculate metrics even if no detections were made (to show 0 values)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
            
            per_model_metrics = {}
            for model_type, stats in model_stats.items():
                tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                per_model_metrics[model_type] = {
                    'precision': p,
                    'recall': r,
                    'f1_score': f1,
                    'accuracy': acc,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'per_model_metrics': per_model_metrics
            }
        
        print(f"\nProcessing complete!")
        print(f"  Frames processed: {frame_count}")
        print(f"  Time elapsed: {elapsed:.2f} seconds")
        print(f"  Average FPS: {frame_count / elapsed:.2f}")
        print(f"  Average time per frame: {avg_frame_time:.4f} seconds ({avg_frame_time*1000:.2f} ms)")
        
        # Final Results Summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS - VIDEO PROCESSING")
        print(f"{'='*70}")
        print(f"\nTIMING METRICS (Videos):")
        print(f"  Integrated Model - Average time per frame: {avg_frame_time:.4f} seconds ({avg_frame_time*1000:.2f} ms)")
        print(f"  Sign Model - Average time per frame: {avg_model_times['sign_model']:.4f} seconds ({avg_model_times['sign_model']*1000:.2f} ms)")
        print(f"  Signal Model - Average time per frame: {avg_model_times['signal_model']:.4f} seconds ({avg_model_times['signal_model']*1000:.2f} ms)")
        print(f"  Anomaly Model - Average time per frame: {avg_model_times['anomaly_model']:.4f} seconds ({avg_model_times['anomaly_model']*1000:.2f} ms)")
        if 'risk_congestion_model' in avg_model_times:
            print(f"  Risk & Congestion Model - Average time per frame: {avg_model_times['risk_congestion_model']:.4f} seconds ({avg_model_times['risk_congestion_model']*1000:.2f} ms)")
        # Performance metrics are suppressed here - they will be aggregated and printed
        # at the end of process_directory instead
        # (Metrics are still calculated and returned in the return dictionary)
        
        print(f"\nTotal frames processed: {frame_count}")
        print(f"Total processing time: {elapsed:.2f} seconds")
        print(f"Average processing FPS: {frame_count / elapsed:.2f}")
        
        # Print individual model test outputs
        self._print_individual_model_outputs(
            signal_detections_by_class,
            sign_detections_by_class,
            anomaly_detections_by_class,
            risk_congestion_detections_by_class,
            frame_count
        )
        
        # Save all detections to CSV
        if all_detections_for_csv and output_path:
            try:
                # Determine output CSV path
                output_path_obj = Path(output_path)
                if output_path_obj.is_dir():
                    csv_path = output_path_obj / "detections.csv"
                else:
                    csv_path = output_path_obj.parent / "detections.csv"
                
                # Ensure parent directory exists
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Prepare CSV data
                csv_data = []
                for item in all_detections_for_csv:
                    det = item['detection']
                    x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
                    
                    # Get timing for this frame
                    frame_idx = item['frame_number'] - 1  # Convert to 0-indexed
                    frame_time = frame_times[frame_idx] if frame_idx < len(frame_times) and frame_idx >= 0 else 0.0
                    model_times = model_times_list[frame_idx] if frame_idx < len(model_times_list) and frame_idx >= 0 else {}
                    
                    row = {
                        'Source': item['source_name'],
                        'Frame': item['frame_number'],
                        'Processing Time (s)': f"{frame_time:.6f}",
                        'Processing Time (ms)': f"{frame_time * 1000:.3f}",
                        'Sign Model Time (ms)': f"{model_times.get('sign_model', 0.0) * 1000:.3f}",
                        'Signal Model Time (ms)': f"{model_times.get('signal_model', 0.0) * 1000:.3f}",
                        'Anomaly Model Time (ms)': f"{model_times.get('anomaly_model', 0.0) * 1000:.3f}",
                        'Risk Congestion Model Time (ms)': f"{model_times.get('risk_congestion_model', 0.0) * 1000:.3f}",
                        'Model Type': det.get('model_type', 'unknown'),
                        'Class Name': det.get('class_name', 'unknown'),
                        'Confidence': f"{det.get('confidence', 0.0):.4f}",
                        'Bbox X1': int(x1),
                        'Bbox Y1': int(y1),
                        'Bbox X2': int(x2),
                        'Bbox Y2': int(y2)
                    }
                    csv_data.append(row)
                
                # Write to CSV
                fieldnames = ['Source', 'Frame', 'Processing Time (s)', 'Processing Time (ms)',
                             'Sign Model Time (ms)', 'Signal Model Time (ms)',
                             'Anomaly Model Time (ms)', 'Risk Congestion Model Time (ms)',
                             'Model Type', 'Class Name', 'Confidence', 
                             'Bbox X1', 'Bbox Y1', 'Bbox X2', 'Bbox Y2']
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)
                
                print(f"Saved integrated format detections CSV to: {csv_path}")
            except Exception as e:
                print(f"Warning: Failed to save integrated format CSV: {e}")
        
        # Save risk & congestion CSV in original format (matching demoidd_video_riskest_cong.py)
        if risk_congestion_frame_data and output_path:
            try:
                # Determine output CSV path for original format
                output_path_obj = Path(output_path)
                if output_path_obj.is_dir():
                    video_name = Path(video_path).stem
                    csv_path_original = output_path_obj / f"{video_name}_result.csv"
                else:
                    video_name = Path(video_path).stem
                    csv_path_original = output_path_obj.parent / f"{video_name}_result.csv"
                
                # Ensure parent directory exists
                csv_path_original.parent.mkdir(parents=True, exist_ok=True)
                
                # Create DataFrame matching original format
                df = pd.DataFrame(risk_congestion_frame_data)
                
                # Ensure columns are in the correct order (matching original)
                column_order = [
                    'Frame Number',
                    'c0', 'c1', 'c2', 'c3', 'c4',
                    'a0', 'a1', 'a2', 'a3', 'a4',
                    'd0', 'd1', 'd2', 'd3', 'd4',
                    'b0', 'b1', 'b2', 'b3', 'b4',
                    'pc_c0', 'pc_c1', 'pc_c2', 'pc_c3', 'pc_c4',
                    'pc_p0', 'pc_p1', 'pc_p2', 'pc_p3', 'pc_p4',
                    'B_sum', 'PCU_Sum', 'Raw Label', 'Smoothed Label'
                ]
                
                # Reorder columns (only include columns that exist)
                existing_columns = [col for col in column_order if col in df.columns]
                df = df[existing_columns]
                
                # Save to CSV
                df.to_csv(csv_path_original, index=False)
                print(f"Saved original format risk/congestion CSV to: {csv_path_original}")
            except Exception as e:
                print(f"Warning: Failed to save original format CSV: {e}")
                
                print(f"\n{'='*70}")
                print(f"Saved all detections to CSV: {csv_path}")
                print(f"  Total detections: {len(all_detections_for_csv)}")
                print(f"    - Traffic Signs: {sum(1 for item in all_detections_for_csv if item['detection']['model_type'] == 'traffic_sign')}")
                print(f"    - Traffic Signals: {sum(1 for item in all_detections_for_csv if item['detection']['model_type'] == 'traffic_signal')}")
                print(f"    - Road Anomalies: {sum(1 for item in all_detections_for_csv if item['detection']['model_type'] == 'road_anomaly')}")
                print(f"{'='*70}")
            except Exception as e:
                print(f"\nWarning: Failed to save detections CSV file: {e}")
        
        print(f"{'='*70}\n")
        
        return {
            'frame_count': frame_count,
            'elapsed_time': elapsed,
            'avg_frame_time': avg_frame_time,
            'avg_model_times': avg_model_times,
            'frame_times': frame_times,  # Per-frame processing times (list of floats in seconds)
            'model_times_per_frame': model_times_list,  # Per-frame model times (list of dicts)
            'metrics': metrics,
            'risk_level_metrics': risk_level_metrics
        }
    
    def process_directory(self, images_dir: str, output_dir: str = None, labels_dir: str = None,
                         show: bool = False, conf_threshold: float = 0.4, iou_threshold: float = 0.5):
        """
        Process a directory of images or videos.
        
        Args:
            images_dir: Directory containing input images or videos
            output_dir: Directory to save annotated images/videos (optional)
            labels_dir: Directory containing label files for evaluation (optional)
            show: Whether to display results
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        images_path = Path(images_dir)
        
        # Resolve relative paths
        if not images_path.is_absolute():
            cwd_path = Path.cwd() / images_path
            if cwd_path.exists():
                images_path = cwd_path
            else:
                script_dir = Path(__file__).parent.parent
                project_path = script_dir / images_path
                if project_path.exists():
                    images_path = project_path
        
        if not images_path.exists() or not images_path.is_dir():
            print(f"Error: Directory not found: {images_dir}")
            return None
        
        # Get all image and video files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        image_files = [f for f in images_path.rglob('*') 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        video_files = [f for f in images_path.rglob('*') 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        # Determine what to process
        if image_files and video_files:
            print(f"Warning: Found both images ({len(image_files)}) and videos ({len(video_files)}) in directory.")
            print(f"Processing images only. To process videos, specify a directory containing only videos.")
            files_to_process = image_files
            is_video = False
        elif image_files:
            files_to_process = image_files
            is_video = False
        elif video_files:
            files_to_process = video_files
            is_video = True
        else:
            print(f"Error: No image or video files found in {images_path}")
            print(f"Supported image formats: {image_extensions}")
            print(f"Supported video formats: {video_extensions}")
            return None
        
        if is_video:
            print(f"\nFound {len(files_to_process)} videos to process")
        else:
            print(f"\nFound {len(files_to_process)} images to process")
        
        # Setup output directory
        output_path = None
        if output_dir:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                cwd_path = Path.cwd() / output_path
                script_dir = Path(__file__).parent.parent
                project_path = script_dir / output_path
                output_path = cwd_path if cwd_path.exists() or not project_path.exists() else project_path
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_path}")
        
        # Setup labels directory
        labels_path = None
        if labels_dir:
            labels_path = Path(labels_dir)
            if not labels_path.is_absolute():
                cwd_path = Path.cwd() / labels_path
                script_dir = Path(__file__).parent.parent
                project_path = script_dir / labels_path
                labels_path = cwd_path if cwd_path.exists() or not project_path.exists() else project_path
            if not labels_path.exists():
                print(f"Warning: Labels directory not found: {labels_path}")
                labels_path = None
        
        # Process each file
        processed = 0
        failed_files = []
        
        # Collect metrics from all videos for aggregation
        all_video_metrics = []  # List of metrics dicts from each video
        all_per_model_metrics = {}  # Aggregated per-model metrics
        
        for file in files_to_process:
            print(f"\nProcessing: {file.name}")
            
            # Determine output path
            output_file = None
            if output_path:
                relative_path = file.relative_to(images_path)
                if is_video:
                    # For videos, keep the same extension
                    output_file = output_path / f"annotated_{relative_path.name}"
                else:
                    # For images, add annotated prefix
                    output_file = output_path / f"annotated_{relative_path.name}"
                output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if is_video:
                    # Process video
                    result = self.process_video(
                        str(file),
                        output_path=str(output_file) if output_file else None,
                        show=show,
                        labels_dir=str(labels_path) if labels_path else None,
                        iou_threshold=iou_threshold,
                        conf_threshold=conf_threshold
                    )
                    
                    if output_file:
                        print(f"  ✓ Saved to: {output_file}")
                    if result and 'metrics' in result:
                        print(f"  Frames processed: {result.get('frame_count', 'N/A')}")
                        # Collect metrics for aggregation
                        if result.get('metrics'):
                            all_video_metrics.append(result['metrics'])
                else:
                    # Find corresponding label file if labels directory provided
                    label_file = None
                    if labels_path:
                        label_file = labels_path / f"{file.stem}.txt"
                        if not label_file.exists():
                            # Try other naming conventions
                            for pattern in [f"{file.stem}.lines.txt", f"{file.name}.txt"]:
                                potential = labels_path / pattern
                                if potential.exists():
                                    label_file = potential
                                    break
                            else:
                                label_file = None
                    
                    # Process image
                    results, metrics = self.process_image(
                        str(file),
                        output_path=str(output_file) if output_file else None,
                        show=show,
                        label_path=str(label_file) if label_file else None,
                        iou_threshold=iou_threshold,
                        conf_threshold=conf_threshold
                    )
                    
                    if output_file:
                        print(f"  ✓ Saved to: {output_file}")
                    print(f"  Detections: {len(results['detections'])}")
                
                processed += 1
                print(f"  ✓ Successfully processed {file.name}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n  ✗ ERROR processing {file.name}: {error_msg}")
                failed_files.append((file.name, error_msg))
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"Processing complete!")
        file_type = "videos" if is_video else "images"
        print(f"  Successfully processed: {processed}/{len(files_to_process)} {file_type}")
        if failed_files:
            print(f"\n  Failed {file_type} ({len(failed_files)}):")
            for failed_name, error in failed_files:
                print(f"    - {failed_name}: {error}")
        if output_path:
            print(f"  Output directory: {output_path}")
        
        # Aggregate and print overall performance metrics for videos
        # Initialize variables for return statement
        total_tp_all = 0
        total_fp_all = 0
        total_fn_all = 0
        overall_precision = None
        overall_recall = None
        overall_f1 = None
        overall_accuracy = None
        
        if is_video and all_video_metrics:
            print(f"\n{'='*70}")
            print("OVERALL PERFORMANCE METRICS (Aggregated across all videos)")
            print(f"{'='*70}")
            
            # Aggregate TP, FP, FN across all videos
            total_tp_all = sum(m.get('total_tp', 0) for m in all_video_metrics)
            total_fp_all = sum(m.get('total_fp', 0) for m in all_video_metrics)
            total_fn_all = sum(m.get('total_fn', 0) for m in all_video_metrics)
            
            # Calculate overall metrics
            overall_precision = total_tp_all / (total_tp_all + total_fp_all) if (total_tp_all + total_fp_all) > 0 else 0.0
            overall_recall = total_tp_all / (total_tp_all + total_fn_all) if (total_tp_all + total_fn_all) > 0 else 0.0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
            overall_accuracy = total_tp_all / (total_tp_all + total_fp_all + total_fn_all) if (total_tp_all + total_fp_all + total_fn_all) > 0 else 0.0
            
            print(f"\n  Integrated Model (All models combined across all videos):")
            print(f"    Precision: {overall_precision:.4f}")
            print(f"    Recall:    {overall_recall:.4f}")
            print(f"    F1 Score:  {overall_f1:.4f}")
            print(f"    Accuracy:  {overall_accuracy:.4f}")
            print(f"    TP: {total_tp_all}, FP: {total_fp_all}, FN: {total_fn_all}")
            
            # Aggregate per-model metrics
            per_model_aggregated = {}
            for video_metrics in all_video_metrics:
                if 'per_model_metrics' in video_metrics:
                    for model_type, model_metrics in video_metrics['per_model_metrics'].items():
                        if model_type not in per_model_aggregated:
                            per_model_aggregated[model_type] = {'tp': 0, 'fp': 0, 'fn': 0}
                        per_model_aggregated[model_type]['tp'] += model_metrics.get('tp', 0)
                        per_model_aggregated[model_type]['fp'] += model_metrics.get('fp', 0)
                        per_model_aggregated[model_type]['fn'] += model_metrics.get('fn', 0)
            
            if per_model_aggregated:
                print(f"\n  Individual Models (Aggregated across all videos):")
                for model_type, stats in per_model_aggregated.items():
                    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                    print(f"    {model_type}:")
                    print(f"      Precision: {p:.4f}")
                    print(f"      Recall:    {r:.4f}")
                    print(f"      F1 Score:  {f1:.4f}")
                    print(f"      Accuracy:  {acc:.4f}")
                    print(f"      TP: {tp}, FP: {fp}, FN: {fn}")
        
        elif is_video and labels_path:
            print(f"\n{'='*70}")
            print("PERFORMANCE METRICS:")
            print(f"  (No ground truth labels found - precision/recall/F1/accuracy not calculated)")
            print(f"{'='*70}")
        
        print(f"{'='*70}")
        
        return {
            'processed': processed,
            'total': len(files_to_process),
            'failed': len(failed_files),
            'failed_files': failed_files,
            'output_dir': str(output_path) if output_path else None,
            'overall_metrics': {
                'total_tp': total_tp_all,
                'total_fp': total_fp_all,
                'total_fn': total_fn_all,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'accuracy': overall_accuracy
            } if (is_video and all_video_metrics and overall_precision is not None) else None
        }
    
    def process_webcam(self, camera_id: int = 0):
        """
        Process live video from webcam.
        
        Args:
            camera_id (int): Camera device ID (default: 0)
        """
        print(f"\nStarting webcam capture (camera {camera_id})")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Process frame
            results = self.process_frame(frame)
            
            # Show FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                results['annotated_frame'],
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display
            cv2.imshow('Integrated Traffic Perception', results['annotated_frame'])
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam processing stopped")
    
    def evaluate(self, test_dir: str = None, iou_threshold: float = 0.5, conf_threshold: float = 0.4, 
                 save_images: bool = False, output_dir: str = None):
        """
        Evaluate the integrated model on test dataset.
        
        This method:
        1. Loads images and labels from the test directory
        2. Runs predictions on each image
        3. Matches predictions with ground truth using IoU
        4. Calculates precision, recall, and F1 score
        5. Tracks timing statistics
        6. Optionally saves annotated images
        
        Args:
            test_dir (str): Path to test directory containing 'images' and 'labels' subdirectories.
                          Can be absolute or relative to project root.
                          If None, defaults to 'test' directory relative to this file
            iou_threshold (float): IoU threshold for matching predictions to ground truth (default: 0.5)
            conf_threshold (float): Confidence threshold for predictions (default: 0.4)
            save_images (bool): Whether to save annotated output images (default: False)
            output_dir (str): Directory to save annotated images (default: test_dir/eval_output)
        
        Returns:
            dict: Evaluation metrics containing:
                - precision: Overall precision (integrated model)
                - recall: Overall recall (integrated model)
                - f1_score: Overall F1 score (integrated model)
                - per_model_metrics: Metrics per individual model type
                - per_class_metrics: Metrics per class
                - avg_frame_time: Average processing time per frame (integrated model)
                - avg_model_times: Average processing time per frame for each individual model
        """
        # Default to 'test' directory relative to this file if not specified
        if test_dir is None:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            test_dir = str(script_dir / 'test')
        
        # Resolve the test directory path
        test_path = Path(test_dir)
        
        # Check if path is a file (user might have passed a video/image file by mistake)
        if test_path.is_file():
            raise ValueError(
                f"Test directory path points to a file, not a directory: {test_path}\n"
                f"Please provide a directory path containing 'images' and 'labels' subdirectories.\n"
                f"Example: --test-dir 'sign_det/sign_det/Indian_detection'"
            )
        
        # If path doesn't exist, try resolving relative to project root
        if not test_path.exists():
            # Try resolving relative to project root (parent of Integrated folder)
            project_root = Path(__file__).parent.parent
            alt_path = project_root / test_dir
            if alt_path.exists() and alt_path.is_dir():
                test_path = alt_path
                print(f"Resolved test directory relative to project root: {test_path}")
            else:
                raise ValueError(
                    f"Test directory not found: {test_path}\n"
                    f"Also tried: {alt_path}\n"
                    f"Please provide a valid directory path containing 'images' and 'labels' subdirectories."
                )
        
        # Check for images and labels subdirectories
        images_dir = test_path / 'images'
        labels_dir = test_path / 'labels'
        
        # If images/labels not found directly, check if test_path itself contains images
        # (some datasets have test/images structure)
        if not images_dir.exists():
            # Check if test_path has image files directly
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            direct_images = [f for f in test_path.iterdir() 
                           if f.is_file() and f.suffix.lower() in image_extensions]
            if direct_images:
                raise ValueError(
                    f"Found image files directly in {test_path}, but expected 'images' subdirectory.\n"
                    f"Please ensure your test directory has this structure:\n"
                    f"  test_dir/\n"
                    f"    images/\n"
                    f"      *.jpg\n"
                    f"    labels/\n"
                    f"      *.txt"
                )
            else:
                raise ValueError(
                    f"Test images directory not found: {images_dir}\n"
                    f"Expected directory structure:\n"
                    f"  {test_path}/\n"
                    f"    images/  (containing image files)\n"
                    f"    labels/  (containing label files)\n"
                    f"\nCurrent directory contents: {list(test_path.iterdir()) if test_path.exists() else 'N/A'}"
                )
        
        if not labels_dir.exists():
            # Labels directory is optional (some tests might not have ground truth)
            print(f"Warning: Test labels directory not found: {labels_dir}")
            print("Evaluation will proceed without ground truth comparison (inference-only mode)")
            labels_dir = None
        
        print(f"\n{'='*70}")
        print("Evaluating Integrated Traffic Perception Model")
        print(f"{'='*70}")
        print(f"Test directory: {test_dir}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"Confidence threshold: {conf_threshold}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        print(f"\nFound {len(image_files)} test images")
        
        # Setup output directory for saving images if requested
        if save_images:
            if output_dir is None:
                output_dir = test_path / 'eval_output'
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving annotated images to: {output_dir}")
        
        # Statistics
        total_tp = 0  # True Positives
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives
        
        # Per-model statistics
        model_stats = {
            'traffic_sign': {'tp': 0, 'fp': 0, 'fn': 0},
            'traffic_signal': {'tp': 0, 'fp': 0, 'fn': 0},
            'road_anomaly': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        
        # Per-class statistics
        class_stats = {}
        
        # Timing statistics
        frame_times = []  # Track processing time for each frame
        model_times_list = []  # Track individual model times for each frame
        
        # Process each image
        processed = 0
        for img_file in image_files:
            # Load image
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"Warning: Could not load image {img_file}")
                continue
            
            h, w = frame.shape[:2]
            
            # Load ground truth labels
            gt_boxes = []
            if labels_dir is not None:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    gt_boxes = self._load_yolo_labels(label_file, w, h)
            
            # Run predictions
            results = self.process_frame(frame, conf_threshold=conf_threshold)
            
            # Track timing statistics
            frame_times.append(results.get('processing_time', 0.0))
            if results.get('model_times'):
                model_times_list.append(results['model_times'])
            
            pred_detections = [d for d in results['detections'] 
                             if d['confidence'] >= conf_threshold]
            
            # Convert predictions to format for matching
            pred_boxes = []
            for det in pred_detections:
                x1, y1, x2, y2 = det['bbox']
                pred_boxes.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': det['class_name'],
                    'model_type': det['model_type'],
                    'confidence': det['confidence']
                })
            
            # Match predictions to ground truth
            matches, unmatched_preds, unmatched_gts = self._match_detections(
                pred_boxes, gt_boxes, iou_threshold
            )
            
            # Update statistics
            total_tp += len(matches)
            total_fp += len(unmatched_preds)
            total_fn += len(unmatched_gts)
            
            # Update per-model statistics
            for match in matches:
                model_type = match['pred']['model_type']
                if model_type in model_stats:
                    model_stats[model_type]['tp'] += 1
            
            for unmatched_pred in unmatched_preds:
                model_type = unmatched_pred['model_type']
                if model_type in model_stats:
                    model_stats[model_type]['fp'] += 1
            
            for unmatched_gt in unmatched_gts:
                # Try to infer model type from class_id or class_name
                model_type = self._infer_model_type_from_gt(unmatched_gt)
                if model_type in model_stats:
                    model_stats[model_type]['fn'] += 1
            
            # Update per-class statistics
            for match in matches:
                class_name = match['pred']['class']
                if class_name not in class_stats:
                    class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_stats[class_name]['tp'] += 1
            
            for unmatched_pred in unmatched_preds:
                class_name = unmatched_pred['class']
                if class_name not in class_stats:
                    class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_stats[class_name]['fp'] += 1
            
            for unmatched_gt in unmatched_gts:
                class_name = unmatched_gt.get('class_name', 'unknown')
                if class_name not in class_stats:
                    class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_stats[class_name]['fn'] += 1
            
            # Save annotated image if requested
            if save_images:
                annotated_frame = results['annotated_frame']
                if annotated_frame is not None and annotated_frame.size > 0:
                    annotated_frame = annotated_frame.copy()
                    
                    # Optionally draw ground truth boxes for comparison
                    # (You can uncomment this to visualize GT boxes)
                    # for gt_box in gt_boxes:
                    #     x1, y1, x2, y2 = gt_box['bbox']
                    #     cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                    #     cv2.putText(annotated_frame, 'GT', (int(x1), int(y1) - 5), 
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    output_path = output_dir / f"eval_{img_file.name}"
                    output_path_str = str(output_path).replace('\\', '/')
                    success = cv2.imwrite(output_path_str, annotated_frame)
                    if not success:
                        print(f"Warning: Failed to save image {output_path}")
                else:
                    print(f"Warning: Annotated frame is empty for {img_file.name}")
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(image_files)} images...")
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        
        # Calculate per-model metrics
        per_model_metrics = {}
        for model_type, stats in model_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            per_model_metrics[model_type] = {
                'precision': p,
                'recall': r,
                'f1_score': f1,
                'accuracy': acc,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for class_name, stats in class_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            per_class_metrics[class_name] = {
                'precision': p,
                'recall': r,
                'f1_score': f1,
                'accuracy': acc,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Calculate average timing statistics
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0.0
        avg_model_times = {
            'sign_model': 0.0,
            'signal_model': 0.0,
            'anomaly_model': 0.0
        }
        if model_times_list:
            for model_name in avg_model_times.keys():
                times = [mt.get(model_name, 0.0) for mt in model_times_list if model_name in mt]
                if times:
                    avg_model_times[model_name] = sum(times) / len(times)
        
        # Print results
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Print timing metrics
        print(f"\nTIMING METRICS (Images):")
        print(f"  Integrated Model - Average time per frame: {avg_frame_time:.4f} seconds ({avg_frame_time*1000:.2f} ms)")
        print(f"  Total frames processed: {len(frame_times)}")
        print(f"  Sign Model - Average time per frame: {avg_model_times['sign_model']:.4f} seconds ({avg_model_times['sign_model']*1000:.2f} ms)")
        print(f"  Signal Model - Average time per frame: {avg_model_times['signal_model']:.4f} seconds ({avg_model_times['signal_model']*1000:.2f} ms)")
        print(f"  Anomaly Model - Average time per frame: {avg_model_times['anomaly_model']:.4f} seconds ({avg_model_times['anomaly_model']*1000:.2f} ms)")
        if 'risk_congestion_model' in avg_model_times:
            print(f"  Risk & Congestion Model - Average time per frame: {avg_model_times['risk_congestion_model']:.4f} seconds ({avg_model_times['risk_congestion_model']*1000:.2f} ms)")
        
        # Print performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print(f"\n  Integrated Model (All models combined):")
        print(f"    Precision: {precision:.4f} ({total_tp}/{total_tp + total_fp})")
        print(f"    Recall:    {recall:.4f} ({total_tp}/{total_tp + total_fn})")
        print(f"    F1 Score:  {f1_score:.4f}")
        print(f"    Accuracy:  {accuracy:.4f} ({total_tp}/{total_tp + total_fp + total_fn})")
        print(f"    TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        
        print(f"\n  Individual Models:")
        for model_type, metrics in per_model_metrics.items():
            print(f"    {model_type}:")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall:    {metrics['recall']:.4f}")
            print(f"      F1 Score:  {metrics['f1_score']:.4f}")
            print(f"      Accuracy:  {metrics['accuracy']:.4f}")
            print(f"      TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        
        print(f"\nPer-Class Metrics (top 10 by F1 score):")
        sorted_classes = sorted(per_class_metrics.items(), 
                              key=lambda x: x[1]['f1_score'], 
                              reverse=True)
        for class_name, metrics in sorted_classes[:10]:
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1 Score:  {metrics['f1_score']:.4f}")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        
        if save_images:
            print(f"\nAnnotated images saved to: {output_dir}")
        
        print(f"\n{'='*70}\n")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'per_model_metrics': per_model_metrics,
            'per_class_metrics': per_class_metrics,
            'avg_frame_time': avg_frame_time,
            'avg_model_times': avg_model_times,
            'frame_times': frame_times
        }
    
    def _load_yolo_labels(self, label_file: Path, img_width: int, img_height: int) -> List[Dict]:
        """
        Load YOLO format labels and convert to absolute coordinates.
        
        YOLO format: class_id x_center y_center width height (normalized)
        
        Args:
            label_file: Path to label file
            img_width: Image width
            img_height: Image height
        
        Returns:
            List of ground truth boxes with absolute coordinates (x1, y1, x2, y2)
        """
        gt_boxes = []
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to x1, y1, x2, y2
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    gt_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'class_name': self._class_id_to_name(class_id)
                    })
        except Exception as e:
            print(f"Warning: Error loading label file {label_file}: {e}")
        
        return gt_boxes
    
    def _class_id_to_name(self, class_id: int) -> str:
        """
        Convert class ID to class name based on common mappings.
        
        This is a heuristic - you may need to adjust based on your dataset.
        Common mappings:
        - 0: pothole (for detection datasets)
        - 1: traffic signal (traffic_signal)
        - 2+: various traffic signs (traffic_sign)
        
        Note: The actual mapping depends on your dataset.
        """
        # This is a default mapping - adjust based on your dataset
        # You might want to load this from a config file
        if class_id == 0:
            return 'pothole'  # Default for object detection datasets
        elif class_id == 1:
            return 'traffic_signal'  # Generic signal
        else:
            return f'class_{class_id}'
    
    def _infer_model_type_from_gt(self, gt_box: Dict) -> str:
        """Infer model type from ground truth box."""
        class_id = gt_box.get('class_id', -1)
        class_name = gt_box.get('class_name', '').lower()
        
        # Heuristic: class 0 is usually pothole (for object detection datasets)
        if 'pothole' in class_name or 'speedbump' in class_name or class_id == 0:
            return 'road_anomaly'
        elif class_id == 1 or 'signal' in class_name or 'light' in class_name:
            return 'traffic_signal'
        else:
            return 'traffic_sign'
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
        
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _match_detections(self, pred_boxes: List[Dict], gt_boxes: List[Dict], 
                         iou_threshold: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Match predictions to ground truth boxes using IoU and class matching.
        
        Args:
            pred_boxes: List of prediction dictionaries
            gt_boxes: List of ground truth dictionaries
            iou_threshold: Minimum IoU for a match
        
        Returns:
            Tuple of (matches, unmatched_predictions, unmatched_ground_truths)
        """
        matches = []
        used_pred_indices = set()
        used_gt_indices = set()
        
        # Sort predictions by confidence (highest first)
        sorted_preds_with_idx = sorted(enumerate(pred_boxes), 
                                      key=lambda x: x[1]['confidence'], 
                                      reverse=True)
        
        for pred_idx, pred in sorted_preds_with_idx:
            if pred_idx in used_pred_indices:
                continue
                
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in used_gt_indices:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                
                # Check if classes are compatible
                # We allow matches if model types are compatible
                pred_model = pred['model_type']
                gt_model = self._infer_model_type_from_gt(gt)
                
                # For class matching, we're more lenient - just check if IoU is high enough
                # In a real scenario, you'd want stricter class matching
                if iou > best_iou and pred_model == gt_model:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matches.append({
                    'pred': pred,
                    'gt': gt_boxes[best_gt_idx],
                    'iou': best_iou
                })
                used_pred_indices.add(pred_idx)
                used_gt_indices.add(best_gt_idx)
        
        # Get unmatched predictions and ground truths
        unmatched_preds = [pred for idx, pred in enumerate(pred_boxes) 
                          if idx not in used_pred_indices]
        unmatched_gts = [gt for idx, gt in enumerate(gt_boxes) 
                        if idx not in used_gt_indices]
        
        return matches, unmatched_preds, unmatched_gts
    
    def _find_label_file(self, img_file: Path, labels_dir: Path, label_extensions=None):
        """
        Find corresponding label file for an image.
        Tries multiple naming conventions and extensions.
        
        Args:
            img_file: Path to image file
            labels_dir: Directory containing label files
            label_extensions: List of label extensions to try
        
        Returns:
            Path to label file or None if not found
        """
        if label_extensions is None:
            label_extensions = ['.txt', '.json', '.png', '.jpg', '.jpeg']
        
        img_stem = img_file.stem
        img_name = img_file.name
        
        labels_dir = Path(labels_dir)
        
        # Try different naming conventions
        patterns = [
            img_stem,  # Same name as image (e.g., image.jpg -> image.txt)
            img_name,  # Full image name
            f"{img_stem}.lines",  # Alternative naming format
            f"{img_stem}_label",  # With _label suffix
            f"{img_stem}_gt",  # With _gt suffix
        ]
        
        for pattern in patterns:
            for ext in label_extensions:
                # Try exact match
                label_file = labels_dir / f"{pattern}{ext}"
                if label_file.exists():
                    return label_file
                
                # Try in subdirectories (recursive search)
                found = list(labels_dir.rglob(f"{pattern}{ext}"))
                if found:
                    return found[0]
        
        return None


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Integrated Traffic Perception System - Combines sign, signal, and road anomaly detection'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='webcam',
        help='Input source: path to image/video, or "webcam" for live camera (default: webcam)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for annotated image/video (optional)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display results (useful when only saving)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        metavar='MODEL_LIST',
        help='Select which models to run (default: all models). '
             'Available models: sign (traffic signs), signal (traffic lights), '
             'anomaly (road anomalies). '
             'You can use comma-separated or space-separated format. '
             'Examples: --models sign,signal or --models "sign signal anomaly" or --models sign signal anomaly'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to run models on: cpu, cuda/gpu, or auto (default: auto)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing of models (run sequentially instead)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model on test dataset (requires test/images and test/labels directories)'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Path to test directory containing images and labels subdirectories. '
             'Can be absolute or relative to project root. '
             'Example: sign_det/sign_det/Indian_detection or signal_det/signal_det/training & testing data 1/test '
             '(default: Integrated/test)'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching predictions to ground truth (default: 0.5)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.4,
        help='Confidence threshold for predictions (default: 0.4)'
    )
    
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Save annotated images during evaluation (default: False)'
    )
    
    parser.add_argument(
        '--eval-output-dir',
        type=str,
        default=None,
        help='Directory to save annotated images during evaluation (default: test_dir/eval_output)'
    )
    
    parser.add_argument(
        '--label-path',
        type=str,
        default=None,
        help='Path to ground truth label file (YOLO format) for single image evaluation (optional)'
    )
    
    parser.add_argument(
        '--labels-dir',
        '--label-dir',  # Alias for convenience
        type=str,
        default=None,
        dest='labels_dir',
        help='Path to directory containing ground truth label files (YOLO format) for video evaluation (optional)'
    )
    
    args = parser.parse_args()
    
    def parse_models_arg(models_arg) -> List[str]:
        """
        Parse models argument into validated list, warn on unknown.
        
        Supports both comma-separated and space-separated input.
        
        Args:
            models_arg: Can be None, a string (comma or space separated), or a list
            
        Returns:
            List of valid model names from VALID_MODELS
        """
        if not models_arg:
            return VALID_MODELS
        
        selected = []
        unknown = []
        
        # Handle different input types
        if isinstance(models_arg, list):
            model_list = models_arg
        elif isinstance(models_arg, str):
            # Support both comma-separated and space-separated
            if ',' in models_arg:
                model_list = models_arg.split(',')
            else:
                # Split by spaces, handling quoted strings
                model_list = models_arg.split()
        else:
            print(f"Warning: Invalid models argument type: {type(models_arg)}. Using all models.")
            return VALID_MODELS
        
        # Process each model name
        for part in model_list:
            name = part.strip().lower()
            if not name:
                continue
            if name in VALID_MODELS:
                if name not in selected:
                    selected.append(name)
            else:
                unknown.append(name)
        
        if unknown:
            print(f"Warning: Unknown models ignored: {', '.join(unknown)}")
            print(f"Valid models are: {', '.join(VALID_MODELS)}")
        
        if not selected:
            print("No valid models specified; defaulting to all models.")
            return VALID_MODELS
        
        print(f"Selected models: {', '.join(selected)}")
        return selected
    
    def parse_device_arg(device_arg: str) -> str:
        """Normalize device argument to values accepted by detectors."""
        if not device_arg or device_arg.lower() == 'auto':
            return ''
        name = device_arg.lower()
        if name == 'gpu':
            return 'cuda'
        if name in ('cuda', 'cpu'):
            return name
        print(f"Warning: Unknown device '{device_arg}' - defaulting to auto.")
        return ''
    
    # Initialize pipeline
    selected_models = parse_models_arg(args.models)
    device = parse_device_arg(args.device)
    use_parallel = not args.no_parallel
    pipeline = IntegratedTrafficPerception(
        selected_models=selected_models,
        device=device,
        use_parallel=use_parallel
    )
    
    # If evaluation mode, run evaluation and exit
    if args.evaluate:
        try:
            metrics = pipeline.evaluate(
                test_dir=args.test_dir,
                iou_threshold=args.iou_threshold,
                conf_threshold=args.conf_threshold,
                save_images=args.save_images,
                output_dir=args.eval_output_dir
            )
            # Return metrics for potential script usage
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Determine source type and process
    source = args.source.lower()
    show = not args.no_show
    
    if source == 'webcam':
        # Process webcam
        pipeline.process_webcam()
    else:
        # Check if file exists
        source_path = Path(args.source)
        
        # If path doesn't exist, try resolving relative to script directory
        if not source_path.exists():
            script_dir = Path(__file__).parent
            alt_path = script_dir / args.source
            if alt_path.exists():
                source_path = alt_path
                print(f"Resolved source path relative to script directory: {source_path}")
            else:
                # Try relative to project root
                project_root = Path(__file__).parent.parent
                alt_path2 = project_root / args.source
                if alt_path2.exists():
                    source_path = alt_path2
                    print(f"Resolved source path relative to project root: {source_path}")
                else:
                    print(f"Error: Source file not found: {args.source}")
                    print(f"Tried locations:")
                    print(f"  1. {Path(args.source).absolute()}")
                    print(f"  2. {alt_path.absolute()}")
                    print(f"  3. {alt_path2.absolute()}")
                    return
        
        # Check if source is a directory
        if source_path.is_dir():
            # Process directory of images or videos
            pipeline.process_directory(
                images_dir=str(source_path),
                output_dir=args.output,
                labels_dir=args.labels_dir,
                show=show,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold
            )
        else:
            # Determine if image or video
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            
            ext = source_path.suffix.lower()
            
            if ext in image_extensions:
                # Process image
                pipeline.process_image(
                    str(source_path), 
                    args.output, 
                    show,
                    label_path=args.label_path,
                    iou_threshold=args.iou_threshold,
                    conf_threshold=args.conf_threshold
                )
            elif ext in video_extensions:
                # Process video
                pipeline.process_video(
                    str(source_path), 
                    args.output, 
                    show,
                    labels_dir=args.labels_dir,
                    iou_threshold=args.iou_threshold,
                    conf_threshold=args.conf_threshold
                )
            else:
                print(f"Error: Unsupported file type: {ext}")
                print(f"Supported image formats: {image_extensions}")
                print(f"Supported video formats: {video_extensions}")
                print(f"Or provide a directory path to process all images in the directory")


if __name__ == "__main__":
    main()

