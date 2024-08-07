import numpy as np
import cv2
from ultralytics import YOLO
import random
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()


# opening the file in read mode
my_file = open("C:\\Users\\harim\\OneDrive\\Desktop\\coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()


# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    detection_colors.append((b,g,r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8") 

# Vals to resize video frames | small frame optimise the run 
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

start_time = time.time()
frame_count = 0
current_confidence = 0.45  # Initial confidence threshold
confidence_step = 0.05     # Step for adjusting the confidence threshold
