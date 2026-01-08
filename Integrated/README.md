# Integrated Traffic Perception System - Wrapper Modules

This directory contains wrapper modules for the three YOLO models used in the integrated traffic perception system.

## Files Created

1. **`sign_module.py`** - Wrapper for traffic sign detection and recognition
   - Uses YOLOv5 for detection (best model)
   - Uses YOLOv8 for recognition (best model)
   - Returns detections with sign class names

2. **`signal_module.py`** - Wrapper for traffic signal (light) detection
   - Uses YOLOv5 (best model)
   - Detects red, green, and yellow traffic lights

3. **`road_module.py`** - Wrapper for road anomaly detection
   - Uses YOLOv8 (best model, preferred)
   - Falls back to YOLOv5 if YOLOv8 weights not available
   - Detects potholes and speedbumps

4. **`test_wrappers.py`** - Test script to verify all modules load correctly

## How Each Module Works

Each wrapper module follows the same pattern:

### Initialization
```python
from sign_module import SignDetector
detector = SignDetector()
```

### Detection
```python
detections = detector.detect(frame)
```

### Output Format
Each detection is a dictionary with:
- `model_type`: "traffic_sign", "traffic_signal", or "road_anomaly"
- `class_name`: Name of the detected class (e.g., "Stop", "red", "pothole")
- `confidence`: Confidence score (0.0 to 1.0)
- `bbox`: Bounding box tuple (x1, y1, x2, y2)

## Testing the Modules

### Quick Test (Load Check)
```bash
cd Integrated
python test_wrappers.py
```

This will verify that all modules can be loaded and initialized.

### Test with Real Image
```bash
# Test sign detection
python sign_module.py path/to/image.jpg

# Test signal detection
python signal_module.py path/to/image.jpg

# Test road anomaly detection
python road_module.py path/to/image.jpg
```

## Expected Weight File Locations

The modules look for weights in these default locations:

- **Sign Detection**: `../sign_det/sign_det/weights/detect_weights.pt`
- **Sign Recognition**: `../sign_det/sign_det/weights/recog_weights.pt`
- **Signal Detection**: `../signal_det/signal_det/yolov5 trained model.pt`
- **Road Anomaly**: 
  - Preferred: `../speedbump_pothole_det/speedbump_pothole_det/yolov8_trained_model.pt`
  - Fallback: `../speedbump_pothole_det/speedbump_pothole_det/weights_yolov5/best.pt`

You can override these paths when initializing the detectors:
```python
detector = SignDetector(
    detect_weights_path="custom/path/to/weights.pt",
    recog_weights_path="custom/path/to/recog_weights.pt"
)
```

## Dependencies

Required packages:
- `numpy`
- `opencv-python` (cv2)
- `torch` (PyTorch)
- `ultralytics` (for YOLOv8)
- YOLOv5 repository (for YOLOv5 models)

## Step 2: Integrated Pipeline

The main pipeline (`main.py`) combines all three models and produces:

### Output Format

For each frame, the pipeline returns a dictionary with:

**A. Detection Results** (`detections`)
- List of all detections from all three models
- Each detection contains:
  - `model_type`: "traffic_sign", "traffic_signal", or "road_anomaly"
  - `class_name`: e.g., "Stop", "red", "pothole"
  - `confidence`: float between 0 and 1
  - `bbox`: (x1, y1, x2, y2) in pixels

**B. Confidence Summaries** (`confidence_summaries`)
- `sign_model`: List of all sign detection confidences
- `signal_model`: List of all signal detection confidences
- `anomaly_model`: List of all anomaly detection confidences

**C. Annotated Frame** (`annotated_frame`)
- Original frame with colored bounding boxes and labels
- Green boxes: Traffic signs
- Red boxes: Traffic signals
- Blue boxes: Road anomalies

### Usage

```bash
# Process an image
python main.py --source path/to/image.jpg --output output.jpg

# Process a video
python main.py --source path/to/video.mp4 --output output.mp4

# Process webcam (live)
python main.py --source webcam

# Process without displaying (save only)
python main.py --source video.mp4 --output output.mp4 --no-show
```

### Testing

```bash
# Test pipeline structure
python test_pipeline.py

# Test with real image
python main.py --source path/to/traffic_image.jpg
```

## Next Steps

1. ✅ **Step 1 Complete**: Wrapper modules created
2. ✅ **Step 2 Complete**: Integrated pipeline
3. **Step 3**: Performance evaluation and optimization

