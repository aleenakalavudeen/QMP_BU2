# Files Used from Individual Models in Integrated System

This document lists all files from the individual model directories (`signal_det`, `sign_det`, and `speedbump_pothole_det`) that are used by the Integrated model.

---

## 1. signal_det

### Model Weights (Required)
- **`signal_det/signal_det/yolov5 trained model.pt`**
  - Used by: `Integrated/signal_module.py`
  - Purpose: YOLOv5 model weights for traffic signal (light) detection
  - Default path: Line 66 in `signal_module.py`

### Configuration Files (Optional, Auto-detected)
- **`signal_det/signal_det/training & testing data 1/data.yaml`**
  - Used by: `Integrated/signal_module.py`
  - Purpose: Dataset configuration file (class count, class names)
  - Auto-detected if available (Line 79 in `signal_module.py`)

- **`signal_det/signal_det/testing data2/data.yaml`**
  - Used by: `Integrated/signal_module.py`
  - Purpose: Alternative dataset configuration file
  - Auto-detected if available (Line 80 in `signal_module.py`)

### Python Modules (Imported via sys.path)
The Integrated model adds `signal_det/signal_det` to `sys.path` to import YOLOv5 utilities:

- **`models/common.py`** → `DetectMultiBackend` class
  - Used by: `Integrated/signal_module.py` (Line 31)
  - Purpose: YOLOv5 model loading backend

- **`utils/torch_utils.py`** → `select_device` function
  - Used by: `Integrated/signal_module.py` (Line 32)
  - Purpose: Device selection for model inference

- **`utils/general.py`** → `non_max_suppression`, `scale_boxes` functions
  - Used by: `Integrated/signal_module.py` (Line 380)
  - Purpose: Post-processing utilities (only if DetectMultiBackend method is used)

### Files NOT Used
- `signal_det/signal_det/detect.py` - Standalone detection script
- `signal_det/signal_det/train.py` - Training script
- `signal_det/signal_det/val.py` - Validation script
- `signal_det/signal_det/distance.ipynb` - Notebook
- `signal_det/signal_det/yolov8 trained model.pt` - Not used (YOLOv5 is preferred)

---

## 2. sign_det

### Model Weights (Required)
- **`sign_det/sign_det/weights/detect_weights.pt`**
  - Used by: `Integrated/sign_module.py`
  - Purpose: YOLOv5 model weights for traffic sign detection
  - Default path: Line 122 in `sign_module.py`

- **`sign_det/sign_det/weights/recog_weights.pt`**
  - Used by: `Integrated/sign_module.py`
  - Purpose: YOLOv8 model weights for traffic sign recognition
  - Default path: Line 124 in `sign_module.py`

### Python Modules (Imported via sys.path)
The Integrated model may use YOLOv5 utilities from `signal_det/signal_det` (shared) or a separate YOLOv5 repository:

- **`models/common.py`** → `DetectMultiBackend`, `AutoShape` classes
  - Used by: `Integrated/sign_module.py` (Line 57)
  - Purpose: YOLOv5 model loading backend

- **`utils/torch_utils.py`** → `select_device` function
  - Used by: `Integrated/sign_module.py` (Line 58)
  - Purpose: Device selection for model inference

### Configuration Files (Potentially Used)
- **`sign_det/sign_det/yaml/data_detect.yaml`**
  - May be used by YOLOv5 detection model (if model requires it)
  - Not explicitly referenced in code, but may be embedded in weights

- **`sign_det/sign_det/yaml/data_recog.yaml`**
  - May be used by YOLOv8 recognition model (if model requires it)
  - Not explicitly referenced in code, but may be embedded in weights

### Files NOT Used
- `sign_det/sign_det/pipeline.py` - Standalone pipeline script
- `sign_det/sign_det/script.ipynb` - Notebook
- `sign_det/sign_det/Indian_detection/` - Test data directory

---

## 3. speedbump_pothole_det

### Model Weights (Required - One of the following)
- **`speedbump_pothole_det/speedbump_pothole_det/yolov8_trained_model.pt`** (Preferred)
  - Used by: `Integrated/road_module.py`
  - Purpose: YOLOv8 model weights for road anomaly detection
  - Default path: Line 112 in `road_module.py`
  - Status: Preferred if available

- **`speedbump_pothole_det/speedbump_pothole_det/weights_yolov5/best.pt`** (Fallback)
  - Used by: `Integrated/road_module.py`
  - Purpose: YOLOv5 model weights as fallback
  - Default path: Line 113 in `road_module.py`
  - Status: Used if YOLOv8 weights not available

### Configuration Files (Potentially Used)
- **`speedbump_pothole_det/speedbump_pothole_det/dataset/data.yaml`**
  - May be used by YOLOv5/YOLOv8 models (if model requires it)
  - Not explicitly referenced in code, but may be embedded in weights

### Files NOT Used
- `speedbump_pothole_det/speedbump_pothole_det/detect.py` - Standalone detection script
- `speedbump_pothole_det/speedbump_pothole_det/train.py` - Training script
- `speedbump_pothole_det/speedbump_pothole_det/val.py` - Validation script
- `speedbump_pothole_det/speedbump_pothole_det/weights_yolov5/last.pt` - Not used (best.pt is preferred)

---

## Summary

### Required Files (Must Exist)
1. **signal_det**: `yolov5 trained model.pt`
2. **sign_det**: `detect_weights.pt`, `recog_weights.pt`
3. **speedbump_pothole_det**: `yolov8_trained_model.pt` OR `weights_yolov5/best.pt`

### Optional Files (Auto-detected if Available)
1. **signal_det**: `training & testing data 1/data.yaml` OR `testing data2/data.yaml`

### Python Modules (Imported at Runtime)
1. **signal_det**: `models/common.py`, `utils/torch_utils.py`, `utils/general.py`
2. **sign_det**: `models/common.py`, `utils/torch_utils.py` (may use signal_det's YOLOv5)

---

## Notes

1. **YOLOv5 Utilities**: The Integrated model uses YOLOv5 utilities from `signal_det/signal_det` directory. If that directory contains a full YOLOv5 repository structure, it will be used. Otherwise, the model falls back to `torch.hub` or `ultralytics` YOLO.

2. **Path Management**: The Integrated model carefully manages `sys.path` to ensure correct module imports.

3. **Configuration**: Most configuration values are hardcoded in the Integrated model modules, matching defaults from the original individual models. The original config files are referenced for documentation purposes only.

4. **Model Weights**: All model weights are loaded from `.pt` or `.pth` files. The Integrated model does not re-train or modify these weights.

5. **Standalone Scripts**: The Integrated model does NOT use the standalone detection/training/evaluation scripts from individual models. It only uses the model weights and core Python modules.

