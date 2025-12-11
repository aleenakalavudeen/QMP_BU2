# How the Detection Models Are Called

This document explains the complete flow of how the wrapper modules call the actual YOLO detection models.

## Overview: The Complete Flow

```
main.py (IntegratedTrafficPerception)
    ↓
    Initializes 3 wrapper modules (once, at startup)
    ↓
    When processing a frame:
    ↓
    Calls wrapper.detect(frame) for each model
    ↓
    Each wrapper calls the actual YOLO model
    ↓
    Returns standardized detection results
```

---

## Step-by-Step: How Each Model is Called

### 1. INITIALIZATION (Happens Once at Startup)

When you create `IntegratedTrafficPerception()`, it initializes all three wrappers:

```python
# In main.py, line 50-63
pipeline = IntegratedTrafficPerception()
    ↓
    [1/3] self.sign_detector = SignDetector()
    [2/3] self.signal_detector = SignalDetector()
    [3/3] self.road_detector = RoadAnomalyDetector()
```

**What happens inside each wrapper's `__init__()`:**

#### Sign Detector Initialization:
```python
# sign_module.py, __init__()
1. Load YOLOv5 detection model:
   self.detect_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path='detect_weights.pt')
   ↓
   This loads the actual YOLOv5 model from weights file
   ↓
   self.detect_model is now a YOLOv5 model object

2. Load YOLOv8 recognition model:
   self.recog_model = YOLO('recog_weights.pt')
   ↓
   This loads the actual YOLOv8 model from weights file
   ↓
   self.recog_model is now a YOLOv8 model object
```

#### Signal Detector Initialization:
```python
# signal_module.py, __init__()
1. Try to load using DetectMultiBackend (local YOLOv5)
2. If fails, try torch.hub
3. If fails, try ultralytics YOLO
   ↓
   self.model = [one of the above]
   ↓
   self.model is now a YOLOv5 model object
```

#### Road Anomaly Detector Initialization:
```python
# road_module.py, __init__()
1. Try ultralytics YOLO (for YOLOv8 weights)
2. If fails, try torch.hub (for YOLOv5 weights)
3. If fails, try Windows-compatible loader
   ↓
   self.model = [one of the above]
   ↓
   self.model is now a YOLOv5 or YOLOv8 model object
```

---

### 2. DETECTION (Happens for Each Frame)

When you call `pipeline.process_frame(frame)`, here's what happens:

```python
# main.py, process_frame() - line 98-100
sign_detections = self.sign_detector.detect(frame)      # ← Calls sign wrapper
signal_detections = self.signal_detector.detect(frame)  # ← Calls signal wrapper
anomaly_detections = self.road_detector.detect(frame)    # ← Calls road wrapper
```

---

### 3. INSIDE SIGN DETECTOR.DETECT()

```python
# sign_module.py, detect() method - line 207-260

def detect(self, frame):
    detections = []
    
    # STEP 1: Call YOLOv5 detection model
    results = self.detect_model(frame)  # ← ACTUAL YOLOv5 MODEL CALL
    #         ^^^^^^^^^^^^^^^^
    #         This is the real YOLOv5 model loaded from weights
    
    # STEP 2: Extract bounding boxes from YOLOv5 results
    bounding_boxes = results.xyxy[0].cpu().numpy()
    # Format: [x1, y1, x2, y2, confidence, class_id]
    
    # STEP 3: For each detected sign box:
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max, confidence, _ = box
        
        # STEP 4: Crop the sign region from the frame
        detected_img = frame[y_min:y_max, x_min:x_max]
        
        # STEP 5: Call YOLOv8 recognition model on cropped sign
        recog_pred = self.recog_model.predict(detected_img)  # ← ACTUAL YOLOv8 MODEL CALL
        #         ^^^^^^^^^^^^^^^^
        #         This is the real YOLOv8 model loaded from weights
        
        # STEP 6: Extract class name and confidence
        class_id = np.argmax(recog_pred[0].probs.data)
        class_name = self.classes[class_id]  # e.g., "Stop", "Yield"
        
        # STEP 7: Create detection dictionary
        detection = {
            'model_type': 'traffic_sign',
            'class_name': class_name,
            'confidence': confidence,
            'bbox': (x_min, y_min, x_max, y_max),
            'feature_vector': [128-dim array]
        }
        detections.append(detection)
    
    return detections  # Return list of detections
```

**Key Model Calls:**
- `self.detect_model(frame)` → Calls YOLOv5 detection model
- `self.recog_model.predict(cropped_img)` → Calls YOLOv8 recognition model

---

### 4. INSIDE SIGNAL DETECTOR.DETECT()

```python
# signal_module.py, detect() method - line 207-237

def detect(self, frame):
    detections = []
    
    # STEP 1: Call the model (format depends on how it was loaded)
    if hasattr(self.model, 'predict'):
        # If it's ultralytics YOLO format:
        results = self.model.predict(frame, conf=0.25)  # ← ACTUAL MODEL CALL
        #         ^^^^^^^^^^^^^^^^
        #         This calls the YOLOv5 model loaded from weights
        
        # STEP 2: Extract detections from results
        boxes = results[0].boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            
            # STEP 3: Get class name
            class_name = self.classes[class_id]  # "red", "green", or "yellow"
            
            # STEP 4: Create detection dictionary
            detection = {
                'model_type': 'traffic_signal',
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'feature_vector': [128-dim array]
            }
            detections.append(detection)
    
    return detections
```

**Key Model Call:**
- `self.model.predict(frame)` → Calls YOLOv5 traffic signal model

---

### 5. INSIDE ROAD ANOMALY DETECTOR.DETECT()

```python
# road_module.py, detect() method - line 211-280

def detect(self, frame):
    detections = []
    
    # STEP 1: Call the model (format depends on how it was loaded)
    if hasattr(self.model, 'predict'):
        # If it's ultralytics YOLO format (YOLOv8):
        results = self.model.predict(frame, conf=0.25)  # ← ACTUAL MODEL CALL
        #         ^^^^^^^^^^^^^^^^
        #         This calls the YOLOv8 or YOLOv5 model loaded from weights
        
        # STEP 2: Extract detections
        boxes = results[0].boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            
            # STEP 3: Get class name
            class_name = self.classes[class_id]  # "pothole" or "speedbump"
            
            # STEP 4: Create detection dictionary
            detection = {
                'model_type': 'road_anomaly',
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'feature_vector': [128-dim array]
            }
            detections.append(detection)
    
    elif hasattr(self.model, 'names'):
        # If it's torch.hub YOLOv5 format:
        results = self.model(frame)  # ← ACTUAL MODEL CALL
        #         ^^^^^^^^^^^^
        #         This calls the YOLOv5 model loaded from weights
        
        # Process results similarly...
    
    return detections
```

**Key Model Calls:**
- `self.model.predict(frame)` → Calls YOLOv8 model (if available)
- `self.model(frame)` → Calls YOLOv5 model (fallback)

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ main.py: IntegratedTrafficPerception                    │
│                                                          │
│  process_frame(frame)                                    │
│    ↓                                                     │
│    ├─→ sign_detector.detect(frame)  ────────────────┐  │
│    │                                                  │  │
│    ├─→ signal_detector.detect(frame) ─────────────┐  │  │
│    │                                               │  │  │
│    └─→ road_detector.detect(frame) ────────────┐  │  │  │
│                                                  │  │  │  │
└──────────────────────────────────────────────────┼──┼──┼──┘
                                                   │  │  │
┌──────────────────────────────────────────────────┼──┼──┼──┐
│ sign_module.py: SignDetector                      │  │  │  │
│                                                   │  │  │  │
│  detect(frame):                                  │  │  │  │
│    ↓                                             │  │  │  │
│    results = self.detect_model(frame)  ←─────────┘  │  │  │
│    # ↑ Calls YOLOv5 detection model                │  │  │
│    ↓                                               │  │  │
│    for each detected box:                          │  │  │
│      cropped = frame[y_min:y_max, x_min:x_max]    │  │  │
│      recog = self.recog_model.predict(cropped) ←──┘  │  │
│      # ↑ Calls YOLOv8 recognition model              │  │
│      ↓                                               │  │
│      return detections                               │  │
└───────────────────────────────────────────────────────┼──┘
                                                       │
┌───────────────────────────────────────────────────────┼──┐
│ signal_module.py: SignalDetector                      │  │
│                                                       │  │
│  detect(frame):                                       │  │
│    ↓                                                  │  │
│    results = self.model.predict(frame)  ←────────────┘  │
│    # ↑ Calls YOLOv5 traffic signal model                │
│    ↓                                                    │
│    extract boxes, classes, confidences                  │
│    ↓                                                    │
│    return detections                                    │
└──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ road_module.py: RoadAnomalyDetector                      │
│                                                          │
│  detect(frame):                                          │
│    ↓                                                     │
│    results = self.model.predict(frame)  ←───────────────┘
│    # ↑ Calls YOLOv8 or YOLOv5 road anomaly model
│    ↓
│    extract boxes, classes, confidences
│    ↓
│    return detections
└──────────────────────────────────────────────────────────┘
```

---

## Summary: The Model Call Chain

1. **You call**: `pipeline.process_frame(frame)`

2. **main.py calls**:
   - `sign_detector.detect(frame)`
   - `signal_detector.detect(frame)`
   - `road_detector.detect(frame)`

3. **Each wrapper calls the actual YOLO model**:
   - **Sign**: `detect_model(frame)` → YOLOv5, then `recog_model.predict(cropped)` → YOLOv8
   - **Signal**: `model.predict(frame)` → YOLOv5
   - **Road**: `model.predict(frame)` → YOLOv8 (or YOLOv5 fallback)

4. **Each wrapper processes the raw YOLO output**:
   - Extracts bounding boxes
   - Gets class names
   - Calculates confidences
   - Creates feature vectors
   - Formats into standard dictionary

5. **Returns standardized detections** to main.py

---

## Key Points

- **Models are loaded ONCE** during initialization (in `__init__`)
- **Models are called ONCE per frame** in the `detect()` method
- **The wrapper handles** all the complexity of:
  - Different model formats (YOLOv5 vs YOLOv8)
  - Different output formats
  - Feature extraction
  - Standardizing the output

- **You just call** `detector.detect(frame)` and get clean, standardized results!

