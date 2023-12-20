import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
import time

model = YOLO('best.pt')

# Initialize the time and frames
start_time = time.time()
num_frames = 0

# Set the desired frame rate
desired_frame_rate = 20
frame_duration = 1.0 / desired_frame_rate

while True:
    frame_start_time = time.time()
    
    model.predict(source = '0', show = True, conf=0.20)
    
    # Increment the number of frames
    num_frames += 1
    
    # Calculate and display the frame rate
    elapsed_time = time.time() - start_time
    frame_rate = num_frames / elapsed_time
    print("Frame rate: ", frame_rate)
    
    # Delay to maintain the desired frame rate
    frame_end_time = time.time()
    actual_frame_duration = frame_end_time - frame_start_time
    if actual_frame_duration < frame_duration:
        time.sleep(frame_duration - actual_frame_duration)
