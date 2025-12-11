import numpy as np
import cv2
import torch
from ultralytics import YOLO
import os

"""
    Use this for running videas, with whole detect + recog pipeline. Add the weight paths for detection and recog model
"""
class TSDRPipeline:
    def __init__(self, device='auto'):
        """
        Initialize the TSD and Recognition pipeline.
        
        Args:
            device (str): Device to run models on. Options: 'auto' (auto-detect GPU), 'gpu', 'cpu'. Default: 'auto'
        """
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        print(f"Running on device: {self.device}")

        self.detect_model = torch.hub.load('../yolov5-master/', 'custom', path='./weights/detect_weights.pt', force_reload=True, source='local')
        self.detect_model.to(self.device)
        
        self.recog_model = YOLO('./weights/recog_weights.pt')
        self.recog_model.to(self.device)
        
        self.classes = ["20kmhr", "30kmhr", "40kmhr", "50kmhr", "Intersection", "Men At Work", "Narrow Road", "No Entry", "NoParking", "PedCross", "Stop", "Yield"]

    def red_color_seg(self):

        image = self.image

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 120, 0])
        upper1 = np.array([10, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,120,0])
        upper2 = np.array([185,255,255])

        lower_mask = cv2.inRange(hsv_image, lower1, upper1)
        upper_mask = cv2.inRange(hsv_image, lower2, upper2)
        red_mask = lower_mask | upper_mask

        kernel = np.ones((5, 5), np.uint8) # dont change 5
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_segmented_image = cv2.bitwise_and(image, image, mask=red_mask)
        gray = cv2.cvtColor(red_segmented_image, cv2.COLOR_BGR2GRAY)

        # Find contours in the opened mask
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_threshold = 100 # why?
        max_area_threshold = 8000 # why?

        # Filter out contours with area less than the threshold and fill them with black color
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold or area > max_area_threshold:
                cv2.drawContours(gray, [contour], 0, (0, 255, 0), -1)

        red_segmented_image = cv2.bitwise_and(image, image, mask=gray)

        return gray
    
    def run(self, image):
        self.image = image

        results = self.detect_model(self.image)
        mask = self.red_color_seg()

        bounding_boxes = results.xyxy[0].cpu().numpy()
        
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max, confidence, _ = box

            if not mask[int(y_min):int(y_max), int(x_min):int(x_max)].any():
                confidence = confidence * 0.5

            if confidence > 0.3:
                detected_img = self.image[int(y_min):int(y_max), int(x_min):int(x_max)]
                recog_pred = self.recog_model.predict(detected_img, conf=0.5)
                
                if len(recog_pred) != 0:
                    preds = recog_pred[0].probs.cpu().numpy()
                    class_id = np.argmax(preds.data)

                    cv2.rectangle(self.image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    org = (int(x_min), int(y_min) - 10)  
                    cv2.putText(self.image, self.classes[int(class_id)] + str("----") + str(frame_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(self.image, str(frame_count), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("frame", self.image)
                    
                

if __name__ == "__main__":

    # add video path
    video_capture = cv2.VideoCapture('../test_images/Night/test_vdo_3.mkv')
    # Use device='gpu' for GPU or device='cpu' for CPU. Default 'auto' auto-detects GPU if available.
    pipeline = TSDRPipeline(device='auto')
    frame_count = 0
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        pipeline.run(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            frame_count += 1
            continue


    video_capture.release()
    cv2.destroyAllWindows()




