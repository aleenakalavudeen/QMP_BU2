# **Roadway Characteristics detection and Classification**

To detect roadway characteristics in real-time and classify them into Speed breakers and potholes.

---

## **PROCEDURE**  
Two models have been used for this task: **YOLOv5** and **YOLOv8**.

---

## **MODEL DESCRIPTION**

### **YOLOv5**  
- **Epochs:** 50  
- **Batch Size:** 32  
- **Learning Rate:** 0.01  
- **Parameters:** 7,018,216  
- **Model Size:** 14.114 MB  
- **Inference Time:** 16.4 ms  
- **Hardware:** NVIDIA RTX 

### **YOLOv8**  
- **Epochs:** 50  
- **Batch Size:** 32  
- **Learning Rate:** 0.01  
- **Parameters:** 3,006,233  
- **Model Size:** 6.097 MB  
- **Inference Time:** 7.3 ms  
- **Hardware:** NVIDIA RTX 


## **HOW TO USE**

### **Training the Model**

**YOLOv5:**
```bash
python train.py --img 640 --batch 8 --epochs 50 --data data.yaml --weights yolov5s.pt --name traffic_yolov5
```
The trained weights will get saved as runs/train/yolov5/weights/best.pt

**YOLOv8:**
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=10 batch=8 imgsz=640 name=traffic_yolov8
```

The trained weights will get saved as runs/detect/yolov8/weights/best.pt

---

### **Testing / Evaluation**

**YOLOv5:**
```bash
python val.py --weights runs/train/yolov5/weights/best.pt --data data.yaml --img 640
```

**YOLOv8:**
```bash
yolo task=detect mode=val model=runs/detect/yolov8/weights/best.pt data=data.yaml imgsz=640
```

---

### **Run on Image**

**YOLOv5:**
```bash
python detect.py --weights runs/train/yolov5/weights/best.pt --source path/to/image.jpg --img 640
```

**YOLOv8:**
```bash
yolo task=detect mode=predict model=runs/detect/yolov8/weights/best.pt source=path/to/image.jpg imgsz=640
```

---

### **Run on Video**

**YOLOv5:**
```bash
python detect.py --weights runs/train/yolov5/weights/best.pt --source path/to/video.mp4 --img 640
```

**YOLOv8:**
```bash
yolo task=detect mode=predict model=runs/detect/yolov8/weights/best.pt source=path/to/video.mp4 imgsz=640
```