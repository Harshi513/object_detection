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


# Mapping class IDs to their respective labels 
class_labels = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    detect_params = model.predict(source=[frame], conf=current_confidence, save=False)

    DP = detect_params[0].numpy()

    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            frame,
            class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )

        # Voice alert for all detected object classes
        if conf > 0.8:
            engine.say(f"{class_labels[int(clsID)]} detected with high confidence!")
            engine.runAndWait()

# Display current confidence threshold
    cv2.putText(
        frame,
        f"Confidence: {round(current_confidence, 2)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(
        frame,
        f"FPS: {round(fps, 2)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow('ObjectDetection', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('u'):  # Increase confidence threshold
        current_confidence += confidence_step
    elif key == ord('d'):  # Decrease confidence threshold
        current_confidence -= confidence_step

cap.release()
cv2.destroyAllWindows()


# completed