import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s')

img_path = 'test.jpg'
results = model(img_path)
results.print()

results.show()
plt.show()

log_file_path = 'detections_log.txt'
log_file = open(log_file_path, 'w')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)

    for det in results.xyxy[0].numpy():
        class_id, conf, x_min, y_min, x_max, y_max = det
        log_line = f"Class: {int(class_id)}, Confidence: {conf}, Box: ({x_min}, {y_min}, {x_max}, {y_max})\n"
        log_file.write(log_line)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

log_file.close()

cap.release()
cv2.destroyAllWindows()
