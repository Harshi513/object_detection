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
