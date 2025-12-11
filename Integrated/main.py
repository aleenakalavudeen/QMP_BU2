"""
Integrated Traffic Perception Pipeline

This is the main pipeline that combines all three YOLO models:
1. Traffic Sign Detection & Recognition
2. Traffic Signal (Light) Detection
3. Road Anomaly Detection

For each frame, it produces:
- List of all detections
- Confidence summaries per model
- Fused feature vector (for quantum module)
- Annotated frame with bounding boxes

Author: Integrated Traffic Perception System
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Import our wrapper modules
from sign_module import SignDetector
from signal_module import SignalDetector
from road_module import RoadAnomalyDetector

# Import feature fusion functions
from features import (
    extract_model_features,
    fuse_features,
    get_feature_summary
)


VALID_MODELS = ['sign', 'signal', 'anomaly']


class IntegratedTrafficPerception:
    """
    Main class that integrates all three detection models.
    
    This class coordinates:
    - Loading all three models
    - Processing frames through all models
    - Combining results
    - Creating fused feature vectors
    - Drawing annotations
    """
    
    def __init__(self, selected_models: List[str] = None, device: str = ''):
        """
        Initialize detection models based on selected models.
        
        Args:
            selected_models (list[str] | None): Models to load. Valid entries:
                'sign', 'signal', 'anomaly'. Defaults to all.
            device (str): Device to run models on ('cpu', 'cuda', or '' for auto)
        """
        self.device = device
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
        if 'signal' in self.selected_models:
            print("\n[2/3] Loading signal detection model...")
            self.signal_detector = SignalDetector(device=self.device)
        else:
            print("\n[2/3] Skipping signal detection model (disabled)")
            self.signal_detector = None
        
        # Load road anomaly model if enabled
        if 'anomaly' in self.selected_models:
            print("\n[3/3] Loading road anomaly detection model...")
            self.road_detector = RoadAnomalyDetector(device=self.device)
        else:
            print("\n[3/3] Skipping road anomaly detection model (disabled)")
            self.road_detector = None
        
        print("\n" + "=" * 70)
        print("All models loaded successfully! ✓")
        print("=" * 70)
        
        # Color scheme for drawing bounding boxes
        self.colors = {
            'traffic_sign': (0, 255, 0),      # Green
            'traffic_signal': (0, 0, 255),    # Red
            'road_anomaly': (255, 0, 0)       # Blue
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through all three models.
        
        This is the main processing function. It:
        1. Runs all three models on the frame
        2. Collects all detections
        3. Extracts features and confidences
        4. Creates fused feature vector
        5. Draws annotations
        
        Args:
            frame (numpy.ndarray): BGR image frame
        
        Returns:
            dict: Complete frame analysis containing:
                - detections: List of all detection dictionaries
                - confidence_summaries: Dict with confidence lists per model
                - fused_feature_vector: Combined feature vector (256 dims)
                - annotated_frame: Frame with bounding boxes drawn
        """
        # Step 1: Run enabled models
        sign_detections = self.sign_detector.detect(frame) if self.sign_detector else []
        signal_detections = self.signal_detector.detect(frame) if self.signal_detector else []
        anomaly_detections = self.road_detector.detect(frame) if self.road_detector else []
        
        # Step 2: Combine all detections into a single list
        all_detections = sign_detections + signal_detections + anomaly_detections
        
        # Step 3: Extract features and confidences from each model
        sign_features, sign_confidences = extract_model_features(sign_detections)
        signal_features, signal_confidences = extract_model_features(signal_detections)
        anomaly_features, anomaly_confidences = extract_model_features(anomaly_detections)
        
        # Step 4: Create confidence summaries
        confidence_summaries = {
            'sign_model': sign_confidences,
            'signal_model': signal_confidences,
            'anomaly_model': anomaly_confidences
        }
        
        # Step 5: Fuse all features into a single vector
        fused_feature_vector = fuse_features(
            sign_features=sign_features,
            signal_features=signal_features,
            anomaly_features=anomaly_features,
            sign_confidences=sign_confidences,
            signal_confidences=signal_confidences,
            anomaly_confidences=anomaly_confidences,
            include_confidence=True,
            target_size=256
        )
        
        # Step 6: Create annotated frame
        annotated_frame = self._draw_annotations(frame.copy(), all_detections)
        
        # Step 7: Compile results
        results = {
            'detections': all_detections,
            'confidence_summaries': confidence_summaries,
            'fused_feature_vector': fused_feature_vector,
            'annotated_frame': annotated_frame
        }
        
        return results
    
    def _draw_annotations(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (numpy.ndarray): Original frame
            detections (list): List of detection dictionaries
        
        Returns:
            numpy.ndarray: Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = det['bbox']
            
            # Get color based on model type
            model_type = det['model_type']
            color = self.colors.get(model_type, (255, 255, 255))  # Default white
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            class_name = det['class_name']
            confidence = det['confidence']
            label = f"{model_type}: {class_name} ({confidence:.2f})"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
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
        
        return annotated
    
    def process_image(self, image_path: str, output_path: str = None, show: bool = True):
        """
        Process a single image file.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image (optional). Can be a directory or full file path.
            show (bool): Whether to display the result
        """
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        # Process frame
        results = self.process_frame(frame)
        
        # Print summary
        print(f"\nDetection Summary:")
        print(f"  Traffic Signs: {len([d for d in results['detections'] if d['model_type'] == 'traffic_sign'])}")
        print(f"  Traffic Signals: {len([d for d in results['detections'] if d['model_type'] == 'traffic_signal'])}")
        print(f"  Road Anomalies: {len([d for d in results['detections'] if d['model_type'] == 'road_anomaly'])}")
        print(f"  Total Detections: {len(results['detections'])}")
        print(f"\nFused Feature Vector:")
        summary = get_feature_summary(results['fused_feature_vector'])
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save if requested
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
            success = cv2.imwrite(str(final_output_path), results['annotated_frame'])
            if success:
                print(f"\nAnnotated image saved to: {final_output_path}")
            else:
                print(f"\nError: Failed to save image to: {final_output_path}")
        
        # Show if requested
        if show:
            cv2.imshow('Integrated Traffic Perception', results['annotated_frame'])
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self, video_path: str, output_path: str = None, show: bool = True):
        """
        Process a video file frame by frame.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save annotated video (optional). Can be a directory or full file path.
            show (bool): Whether to display the result
        """
        print(f"\nProcessing video: {video_path}")
        
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            results = self.process_frame(frame)
            
            # Show progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} FPS)")
            
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
        print(f"\nProcessing complete!")
        print(f"  Frames processed: {frame_count}")
        print(f"  Time elapsed: {elapsed:.2f} seconds")
        print(f"  Average FPS: {frame_count / elapsed:.2f}")
    
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
        help='Comma-separated list of models to run: sign, signal, anomaly (default: all)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to run models on: cpu, cuda/gpu, or auto (default: auto)'
    )
    
    args = parser.parse_args()
    
    def parse_models_arg(models_arg: str) -> List[str]:
        """Parse models argument into validated list, warn on unknown."""
        if not models_arg:
            return VALID_MODELS
        
        selected = []
        unknown = []
        
        for part in models_arg.split(','):
            name = part.strip().lower()
            if not name:
                continue
            if name in VALID_MODELS and name not in selected:
                selected.append(name)
            elif name not in VALID_MODELS:
                unknown.append(name)
        
        if unknown:
            print(f"Warning: Unknown models ignored: {', '.join(unknown)}")
        
        if not selected:
            print("No valid models specified; defaulting to all models.")
            return VALID_MODELS
        
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
    pipeline = IntegratedTrafficPerception(selected_models=selected_models, device=device)
    
    # Determine source type and process
    source = args.source.lower()
    show = not args.no_show
    
    if source == 'webcam':
        # Process webcam
        pipeline.process_webcam()
    else:
        # Check if file exists
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: Source file not found: {args.source}")
            return
        
        # Determine if image or video
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        ext = source_path.suffix.lower()
        
        if ext in image_extensions:
            # Process image
            pipeline.process_image(str(source_path), args.output, show)
        elif ext in video_extensions:
            # Process video
            pipeline.process_video(str(source_path), args.output, show)
        else:
            print(f"Error: Unsupported file type: {ext}")
            print(f"Supported image formats: {image_extensions}")
            print(f"Supported video formats: {video_extensions}")


if __name__ == "__main__":
    main()

