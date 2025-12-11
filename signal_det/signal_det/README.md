# **Traffic Light Detection and Classification**

## **AIM**  
To detect traffic lights in real-time and classify them into three classes: **Red**, **Green**, and **Yellow**.

---

## **PROCEDURE**  
Two models have been used for this task: **YOLOv5** and **YOLOv8**.

---

## **MODEL DESCRIPTION**

### **YOLOv5**  
- **Epochs:** 50  
- **Batch Size:** 8  
- **Learning Rate:** 0.01  
- **Parameters:** 7,018,216  
- **Model Size:** 14.114 MB  
- **Inference Time:** 16.4 ms  
- **Hardware:** NVIDIA GeForce GTX 1650 (4096 MB VRAM)  

### **YOLOv8**  
- **Epochs:** 10  
- **Batch Size:** 8  
- **Learning Rate:** 0.01  
- **Parameters:** 3,006,233  
- **Model Size:** 6.097 MB  
- **Inference Time:** 7.3 ms  
- **Hardware:** NVIDIA GeForce GTX 1650 (4096 MB VRAM)  

---

## **DATASETS**  
- **Training Dataset:** [Cinta_v2](https://universe.roboflow.com/wawan-pradana/cinta_v2/browse)  
- **Testing Dataset 1:** [Cinta_v2](https://universe.roboflow.com/wawan-pradana/cinta_v2/browse)  
- **Testing Dataset 2:** [Traffic Lights](https://universe.roboflow.com/elec490/traffic_lights-tnrte)  

---

## **HOW TO USE**

### **Training the Model**

**YOLOv5:**
```bash
python train.py --img 640 --batch 8 --epochs 50 --data data.yaml --weights yolov5s.pt --name traffic_yolov5
```
The trained weights will get saved as runs/train/traffic_yolov5/weights/best.pt

**YOLOv8:**
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=10 batch=8 imgsz=640 name=traffic_yolov8
```

The trained weights will get saved as runs/detect/traffic_yolov8/weights/best.pt

---

### **Testing / Evaluation**

**YOLOv5:**
```bash
python val.py --weights runs/train/traffic_yolov5/weights/best.pt --data data.yaml --img 640
```

**YOLOv8:**
```bash
yolo task=detect mode=val model=runs/detect/traffic_yolov8/weights/best.pt data=data.yaml imgsz=640
```

---

### **Run on Image**

**YOLOv5:**
```bash
python detect.py --weights runs/train/traffic_yolov5/weights/best.pt --source path/to/image.jpg --img 640
```

**YOLOv8:**
```bash
yolo task=detect mode=predict model=runs/detect/traffic_yolov8/weights/best.pt source=path/to/image.jpg imgsz=640
```

---

### **Run on Video**

**YOLOv5:**
```bash
python detect.py --weights runs/train/traffic_yolov5/weights/best.pt --source path/to/video.mp4 --img 640
```

**YOLOv8:**
```bash
yolo task=detect mode=predict model=runs/detect/traffic_yolov8/weights/best.pt source=path/to/video.mp4 imgsz=640
```

---

## **TRAINING RESULTS**

| Metric             | YOLOv5 | YOLOv8 |
|--------------------|--------|--------|
| **Precision (P)**  | 0.91   | 0.97   |
| **Recall (R)**     | 0.962  | 0.872  |
| **Accuracy (A)**   | 0.969  | 0.948  |

---

## **TESTING RESULTS 1**

| Metric             | YOLOv5 | YOLOv8 |
|--------------------|--------|--------|
| **Precision (P)**  | 0.92   | 0.967  |
| **Recall (R)**     | 0.902  | 0.907  |
| **Accuracy (A)**   | 0.92   | 0.975  |

---

## **TESTING RESULTS 2**

| Metric             | YOLOv5 | YOLOv8 |
|--------------------|--------|--------|
| **Precision (P)**  | 0.93   | 0.973  |
| **Recall (R)**     | 0.914  | 0.913  |
| **Accuracy (A)**   | 0.951  | 0.981  |

---

## **DISTANCE EVALUATION**  
- Both models successfully detected traffic lights at a distance greater than the **Stopping Sight Distance (SSD)** for speeds up to **35 km/h**.  
- At higher speeds, the models were unable to detect traffic lights before reaching the SSD.  
- Refer to `distance.ipynb` to calculate distance for the videos.

---

## **CONCLUSION**  
- **YOLOv8** achieves higher accuracy overall.  
- **YOLOv5** is preferred because it detects traffic lights earlier, which is critical for real-time applications.
