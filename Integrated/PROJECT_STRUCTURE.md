# Integrated Traffic Perception System - Project Structure

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [File Structure](#file-structure)
6. [Module Details](#module-details)
7. [API Reference](#api-reference)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Dependencies](#dependencies)

---

## Overview

The **Integrated Traffic Perception System** is a comprehensive computer vision pipeline that combines three specialized detection models to provide complete traffic scene understanding. The system processes images and videos to detect:

- **Traffic Signs** (12 classes: speed limits, Stop, Yield, etc.)
- **Traffic Signals** (3 classes: red, green, yellow lights)
- **Road Anomalies** (potholes, speedbumps)

The system is designed with modularity and extensibility in mind, producing standardized detection outputs ready for downstream processing.

### Key Features

- ✅ **Parallel Processing**: Thread-safe parallel inference across all models
- ✅ **Model Selection**: Enable/disable specific models via command-line flags
- ✅ **Comprehensive Evaluation**: Full metrics with ground truth support (precision, recall, F1, accuracy)
- ✅ **Flexible I/O**: Support for images, videos, webcam, and directory processing
- ✅ **Performance Monitoring**: Detailed timing statistics per model
- ✅ **Standardized Output**: Consistent detection format across all models

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Source                              │
│  (Image/Video/Webcam/Directory)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         IntegratedTrafficPerception Pipeline                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Sign         │  │ Signal       │  │ Road         │     │
│  │ Detector     │  │ Detector     │  │ Anomaly      │     │
│  │              │  │              │  │ Detector     │     │
│  │ YOLOv5       │  │ YOLOv5       │  │ YOLOv8/v5    │     │
│  │ + YOLOv8     │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output                                    │
│  - Detections (standardized format)                         │
│  - Confidence summaries                                      │
│  - Annotated frame with bounding boxes                      │
│  - Performance metrics                                       │
└─────────────────────────────────────────────────────────────┘
```

### Processing Flow

1. **Initialization**: Load all selected models (once at startup)
2. **Frame Processing**: Run all enabled models in parallel (thread-safe)
3. **Annotation**: Draw bounding boxes and labels on frame
4. **Output**: Return standardized results dictionary

---

## Core Components

### 1. Main Pipeline (`main.py`)

**Class**: `IntegratedTrafficPerception`

**Purpose**: Orchestrates all detection models and provides unified interface

**Key Methods**:

- `__init__(selected_models, device, use_parallel)`: Initialize pipeline
- `process_frame(frame, use_parallel)`: Process single frame through all models
- `process_image(image_path, output_path, show, label_path)`: Process single image
- `process_video(video_path, output_path, show, labels_dir)`: Process video file
- `process_directory(images_dir, output_dir, labels_dir)`: Process directory of images
- `process_webcam(camera_id)`: Process live webcam feed
- `evaluate(test_dir, iou_threshold, conf_threshold)`: Evaluate on test dataset

**Output Format**:

```python
{
    'detections': List[Dict],              # All detections from all models
    'confidence_summaries': Dict,          # Confidence lists per model
    'annotated_frame': np.ndarray,         # Frame with bounding boxes
    'processing_time': float,              # Total processing time (seconds)
    'model_times': Dict                    # Per-model timing statistics
}
```

### 2. Detection Modules

All detection modules follow a standard interface:

#### Standard Interface

```python
class Detector:
    def __init__(self, weights_path=None, device=''):
        """Initialize detector with model weights"""
        pass
    
    def detect(self, frame, conf_threshold=0.4):
        """
        Detect objects in frame.
        
        Returns:
            List[Dict]: Detection dictionaries with standard format
        """
        pass
```

#### Standard Detection Format

Each detection dictionary contains:

```python
{
    'model_type': str,           # "traffic_sign", "traffic_signal", "road_anomaly"
    'class_name': str,          # e.g., "Stop", "red", "pothole", "person"
    'confidence': float,        # 0.0 to 1.0
    'bbox': tuple              # (x1, y1, x2, y2) in pixels
}
```

---

## Data Flow

### Detailed Processing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Input Frame (BGR Image)                            │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Parallel Model Inference (Thread-Safe)                │
│                                                               │
│  Thread 1: SignDetector.detect(frame)                        │
│    ├─→ YOLOv5: Detect sign regions                           │
│    ├─→ For each detected region:                             │
│    │   └─→ YOLOv8: Recognize sign type                      │
│    └─→ Return: List[detection_dict]                         │
│                                                               │
│  Thread 2: SignalDetector.detect(frame)                      │
│    └─→ YOLOv5: Detect traffic lights (red/green/yellow)     │
│    └─→ Return: List[detection_dict]                         │
│                                                               │
│  Thread 3: RoadAnomalyDetector.detect(frame)                │
│    └─→ YOLOv8/YOLOv5: Detect potholes/speedbumps             │
│    └─→ Return: List[detection_dict]                         │
│                                                               │
│    └─→ SqueezeDet: Detect vehicles/pedestrians              │
│    ├─→ Compute risk metrics (LOW/MODERATE/HIGH/CRITICAL)   │
│    └─→ Compute congestion metrics (LIGHT/MEDIUM/HEAVY/JAM) │
│    └─→ Return: List[detection_dict]                         │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 6: Annotation                                           │
│                                                               │
│  For each detection:                                         │
│    ├─→ Draw bounding box (color-coded by model type)         │
│    ├─→ Draw label (class_name + confidence)                  │
│    └─→ Add background rectangle for label                    │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 7: Output Dictionary                                    │
│                                                               │
│  {                                                           │
│    'detections': [...],                                      │
│    'confidence_summaries': {...},                            │
│    'annotated_frame': np.array(H, W, 3),                     │
│    'processing_time': float,                                 │
│    'model_times': {...}                                      │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Integrated/
│
├── main.py                          # Main pipeline (2061 lines)
│   └── IntegratedTrafficPerception  # Core orchestration class
│
├── sign_module.py                   # Traffic sign detection (323 lines)
│   └── SignDetector                 # Two-stage: YOLOv5 + YOLOv8
│
├── signal_module.py                 # Traffic signal detection (415 lines)
│   └── SignalDetector               # YOLOv5 for traffic lights
│
├── road_module.py                   # Road anomaly detection (428 lines)
│   └── RoadAnomalyDetector          # YOLOv8 (preferred) or YOLOv5
│
│
├── test_pipeline.py                 # Integration test script
├── test_wrappers.py                 # Module loading test
├── debug_evaluation.py              # Evaluation debugging tool
│
├── README.md                        # Project overview
├── HOW_IT_WORKS.md                  # Detailed flow explanation
├── QUICK_START.md                   # Quick start guide
├── QUICK_START_CMD.md               # CMD-specific guide
├── PROJECT_STRUCTURE.md             # This file
│
├── test/                            # Test dataset
│   ├── images/                      # Test images
│   ├── labels/                      # Ground truth labels (YOLO format)
│   └── eval_output/                 # Evaluation output images
│
├── test image/                      # Sample test images
├── test video/                      # Sample test videos
├── output/                          # Output directory for results
│
├── run_test.bat                     # Windows batch script
├── run_test.ps1                     # PowerShell script
└── fix_torch_cache.bat             # Torch cache fix utility
```

---

## Module Details

### 1. Sign Detection Module (`sign_module.py`)

**Purpose**: Detect and recognize traffic signs

**Architecture**: Two-stage pipeline
- **Stage 1**: YOLOv5 detects sign regions
- **Stage 2**: YOLOv8 recognizes specific sign type

**Classes Detected** (12 classes):
- Speed limits: `20kmhr`, `30kmhr`, `40kmhr`, `50kmhr`
- Regulatory: `Stop`, `Yield`, `No Entry`, `NoParking`
- Warning: `Intersection`, `Men At Work`, `Narrow Road`, `PedCross`

**Features**:
- Red color segmentation for filtering
- Combined confidence (detection × recognition)

**Model Weights**:
- Detection: `../sign_det/sign_det/weights/detect_weights.pt`
- Recognition: `../sign_det/sign_det/weights/recog_weights.pt`

### 2. Signal Detection Module (`signal_module.py`)

**Purpose**: Detect traffic lights and classify color

**Architecture**: Single-stage YOLOv5

**Classes Detected** (3 classes):
- `red`: Red traffic light
- `green`: Green traffic light
- `yellow`: Yellow traffic light

**Features**:
- Multiple loading methods (DetectMultiBackend, torch.hub, ultralytics)
- Windows compatibility fixes

**Model Weights**:
- `../signal_det/signal_det/yolov5 trained model.pt`

### 3. Road Anomaly Detection Module (`road_module.py`)

**Purpose**: Detect road anomalies (potholes, speedbumps)

**Architecture**: YOLOv8 (preferred) or YOLOv5 (fallback)

**Classes Detected** (2 classes):
- `pothole`: Road potholes
- `speedbump`: Speed bumps

**Features**:
- Automatic fallback to YOLOv5 if YOLOv8 unavailable
- Windows PosixPath compatibility fixes

**Model Weights**:
- Preferred: `../speedbump_pothole_det/speedbump_pothole_det/yolov8_trained_model.pt`
- Fallback: `../speedbump_pothole_det/speedbump_pothole_det/weights_yolov5/best.pt`

---

## API Reference

### IntegratedTrafficPerception Class

#### Initialization

```python
pipeline = IntegratedTrafficPerception(
    selected_models=['sign', 'signal', 'anomaly'],  # Optional
    device='',  # 'cpu', 'cuda', or '' for auto
    use_parallel=True  # Enable parallel processing
)
```

#### Process Single Frame

```python
results = pipeline.process_frame(
    frame,              # np.ndarray: BGR image (H, W, 3)
    use_parallel=None   # Optional: override instance setting
)

# Returns:
{
    'detections': List[Dict],              # All detections
    'confidence_summaries': Dict,          # Per-model confidences
    'annotated_frame': np.ndarray,         # Annotated image
    'processing_time': float,              # Seconds
    'model_times': {                       # Per-model timing
        'sign_model': float,
        'signal_model': float,
        'anomaly_model': float
    }
}
```

#### Process Image

```python
results, metrics = pipeline.process_image(
    image_path,         # str: Path to image
    output_path=None,   # Optional: Save annotated image
    show=True,          # Display result
    label_path=None,    # Optional: Ground truth for metrics
    iou_threshold=0.5,  # IoU threshold for matching
    conf_threshold=0.4  # Confidence threshold
)
```

#### Process Video

```python
stats = pipeline.process_video(
    video_path,         # str: Path to video
    output_path=None,   # Optional: Save annotated video
    show=True,          # Display result
    labels_dir=None,    # Optional: Ground truth labels directory
    iou_threshold=0.5,  # IoU threshold
    conf_threshold=0.4  # Confidence threshold
)
```

#### Process Directory

```python
stats = pipeline.process_directory(
    images_dir,         # str: Directory with images
    output_dir=None,    # Optional: Save annotated images
    labels_dir=None,    # Optional: Ground truth labels
    show=False,         # Display results
    conf_threshold=0.4, # Confidence threshold
    iou_threshold=0.5   # IoU threshold
)
```

#### Process Webcam

```python
pipeline.process_webcam(
    camera_id=0  # Camera device ID
)
```

#### Evaluate on Test Dataset

```python
metrics = pipeline.evaluate(
    test_dir=None,           # Test directory (default: 'test')
    iou_threshold=0.5,       # IoU threshold
    conf_threshold=0.4,       # Confidence threshold
    save_images=False,        # Save annotated images
    output_dir=None           # Output directory
)

# Returns:
{
    'precision': float,
    'recall': float,
    'f1_score': float,
    'accuracy': float,
    'total_tp': int,
    'total_fp': int,
    'total_fn': int,
    'per_model_metrics': Dict,
    'per_class_metrics': Dict,
    'avg_frame_time': float,
    'avg_model_times': Dict
}
```

---

## Usage Examples

### Command-Line Interface

#### Process Single Image

```bash
# Basic usage
python main.py --source "test image/original.jpg" --output "output/result.jpg"

# With ground truth for metrics
python main.py --source "test image/original.jpg" \
               --output "output/result.jpg" \
               --label-path "test/labels/original.txt"

# Select specific models only
python main.py --source "image.jpg" --models sign,signal

# Disable parallel processing
python main.py --source "image.jpg" --no-parallel

# Specify device
python main.py --source "image.jpg" --device cuda
```

#### Process Video

```bash
# Process video with output
python main.py --source "test video/video.mp4" \
               --output "output/annotated_video.mp4"

# Process without displaying
python main.py --source "video.mp4" \
               --output "output.mp4" \
               --no-show

# With ground truth labels
python main.py --source "video.mp4" \
               --output "output.mp4" \
               --labels-dir "test/labels"
```

#### Process Webcam

```bash
# Use default camera (0)
python main.py --source webcam

# Process and save (if supported)
python main.py --source webcam --output "webcam_output.mp4"
```

#### Process Directory

```bash
# Process all images in directory
python main.py --source "test image/" \
               --output "output/" \
               --labels-dir "test/labels"
```

#### Evaluate on Test Dataset

```bash
# Basic evaluation
python main.py --evaluate --test-dir "test/"

# With custom thresholds
python main.py --evaluate \
               --test-dir "test/" \
               --iou-threshold 0.5 \
               --conf-threshold 0.4

# Save annotated images
python main.py --evaluate \
               --test-dir "test/" \
               --save-images \
               --eval-output-dir "test/eval_output"
```

### Python API Usage

#### Basic Example

```python
import cv2
from main import IntegratedTrafficPerception

# Initialize pipeline
pipeline = IntegratedTrafficPerception()

# Load image
frame = cv2.imread("test image/original.jpg")

# Process frame
results = pipeline.process_frame(frame)

# Access results
print(f"Found {len(results['detections'])} detections")
print(f"Processing time: {results['processing_time']:.4f}s")

# Display annotated frame
cv2.imshow("Result", results['annotated_frame'])
cv2.waitKey(0)
```

#### Selective Model Loading

```python
# Load only sign and signal models
pipeline = IntegratedTrafficPerception(
    selected_models=['sign', 'signal']
)

# Process frame (only sign and signal will run)
results = pipeline.process_frame(frame)
```

#### Custom Device

```python
# Force CPU usage
pipeline = IntegratedTrafficPerception(device='cpu')

# Force CUDA (if available)
pipeline = IntegratedTrafficPerception(device='cuda')
```

#### Evaluation Example

```python
# Evaluate on test dataset
metrics = pipeline.evaluate(
    test_dir="test/",
    iou_threshold=0.5,
    conf_threshold=0.4,
    save_images=True
)

# Print results
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Per-model metrics
for model_type, model_metrics in metrics['per_model_metrics'].items():
    print(f"{model_type}:")
    print(f"  Precision: {model_metrics['precision']:.4f}")
    print(f"  Recall: {model_metrics['recall']:.4f}")
```

---

## Configuration

### Model Selection

Available models:
- `sign`: Traffic sign detection and recognition
- `signal`: Traffic signal (light) detection
- `anomaly`: Road anomaly detection (potholes, speedbumps)

### Device Configuration

- `auto` or `''`: Auto-detect (CUDA if available, else CPU)
- `cpu`: Force CPU usage
- `cuda` or `gpu`: Force CUDA/GPU usage

### Processing Options

- `use_parallel`: Enable/disable parallel processing (default: True)
- `conf_threshold`: Confidence threshold for detections (default: 0.4)
- `iou_threshold`: IoU threshold for evaluation matching (default: 0.5)

### Model Weight Paths

Default weight file locations (relative to project root):

- **Sign Detection**: `sign_det/sign_det/weights/detect_weights.pt`
- **Sign Recognition**: `sign_det/sign_det/weights/recog_weights.pt`
- **Signal Detection**: `signal_det/signal_det/yolov5 trained model.pt`
- **Road Anomaly (YOLOv8)**: `speedbump_pothole_det/speedbump_pothole_det/yolov8_trained_model.pt`
- **Road Anomaly (YOLOv5)**: `speedbump_pothole_det/speedbump_pothole_det/weights_yolov5/best.pt`

### Custom Weight Paths

You can override default paths when initializing detectors:

```python
from sign_module import SignDetector

detector = SignDetector(
    detect_weights_path="custom/path/to/detect_weights.pt",
    recog_weights_path="custom/path/to/recog_weights.pt"
)
```

---

## Testing

### Module Loading Test

Test if all modules can be loaded:

```bash
python test_wrappers.py
```

This verifies:
- All modules can be imported
- All models can be initialized
- Basic detection can run (with dummy images)

### Pipeline Integration Test

Test the complete pipeline:

```bash
python test_pipeline.py
```

This verifies:
- Pipeline initialization
- Frame processing
- Output format correctness

### Evaluation Test

Test on actual dataset:

```bash
python main.py --evaluate --test-dir "test/"
```

This will:
- Process all images in `test/images/`
- Compare with ground truth in `test/labels/`
- Calculate precision, recall, F1, accuracy
- Generate per-model and per-class metrics

### Expected Test Directory Structure

```
test/
├── images/          # Test images (.jpg, .png, etc.)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/          # Ground truth labels (YOLO format)
    ├── image1.txt
    ├── image2.txt
    └── ...
```

**YOLO Label Format** (per line):
```
class_id x_center y_center width height
```
All values are normalized (0.0 to 1.0).

---

## Dependencies

### Required Packages

```txt
torch>=1.9.0              # PyTorch
opencv-python>=4.5.0      # OpenCV
numpy>=1.19.0              # NumPy
ultralytics>=8.0.0        # YOLOv8 support
```

### Optional Packages

```txt
pillow>=8.0.0             # Image processing
matplotlib>=3.3.0         # Visualization (if needed)
tqdm>=4.60.0              # Progress bars (if needed)
```

### Model-Specific Dependencies

**Sign Module**:
- YOLOv5 (via torch.hub or local repository)
- YOLOv8 (via ultralytics)

**Signal Module**:
- YOLOv5 (via torch.hub or local repository)
- Optional: Local YOLOv5 DetectMultiBackend

**Road Anomaly Module**:
- YOLOv8 (via ultralytics) - preferred
- YOLOv5 (via torch.hub) - fallback

- SqueezeDet implementation
- IDD dataset utilities

### Installation

```bash
# Install core dependencies
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install ultralytics

# For YOLOv5 (if using local repository)
# Clone yolov5 repository to project root
git clone https://github.com/ultralytics/yolov5.git yolov5-master
```

---

## Performance Considerations

### Parallel Processing

The system uses thread-safe parallel processing by default. Each model runs in a separate thread, with locks to prevent race conditions.

**Benefits**:
- Faster processing (models run simultaneously)
- Better GPU utilization (if available)

**Disable if**:
- Memory constraints
- Debugging single model
- Sequential processing needed

### Model Caching

Models are loaded once at initialization and cached in memory. This means:
- First frame may be slower (model loading)
- Subsequent frames are faster
- Memory usage is constant per model

### Timing Statistics

The system tracks:
- Total processing time per frame
- Individual model processing times
- Average times across video/directory processing

Use these metrics to identify bottlenecks and optimize performance.

---

## Troubleshooting

### Common Issues

#### 1. Model Weights Not Found

**Error**: `FileNotFoundError: Detection weights not found`

**Solution**: 
- Check weight file paths
- Verify files exist in expected locations
- Use custom paths if files are elsewhere

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Use CPU: `--device cpu`
- Process smaller batches
- Reduce image resolution

#### 3. Import Errors

**Error**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
pip install ultralytics
```

#### 4. Windows Path Issues

**Error**: `PosixPath` related errors on Windows

**Solution**:
- Use `fix_torch_cache.bat` to clear cache
- Or manually delete: `%USERPROFILE%\.cache\torch\hub`

#### 5. Thread Safety Warnings

**Warning**: Thread safety issues with parallel processing

**Solution**:
- Disable parallel: `--no-parallel`
- Or ensure all models support thread-safe inference

---

## Extension Points

### Adding New Models

To add a new detection model:

1. Create a new module file (e.g., `new_module.py`)
2. Implement the standard detector interface:
   ```python
   class NewDetector:
       def __init__(self, weights_path=None, device=''):
           # Initialize model
           pass
       
       def detect(self, frame, conf_threshold=0.4):
           # Return List[Dict] with standard format
           pass
   ```
3. Add to `main.py`:
   - Import the module
   - Add to `VALID_MODELS`
   - Initialize in `__init__`
   - Call in `process_frame`

---

## License and Credits

This integrated system combines multiple specialized models:
- Traffic sign detection/recognition models
- Traffic signal detection models
- Road anomaly detection models

All models follow their respective licenses and should be used accordingly.

---

## Version History

- **v1.0**: Initial integrated system with 4 models
- Parallel processing support
- Comprehensive evaluation framework

---

## Contact and Support

For issues, questions, or contributions, please refer to the main project repository.

---

**Last Updated**: 2024
**Documentation Version**: 1.0

