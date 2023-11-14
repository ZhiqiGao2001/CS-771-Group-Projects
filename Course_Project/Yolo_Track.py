from ultralytics import YOLO
import cv2
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define folder and output paths
folder_path = "Dataset/UA-DETRAC/hazy/train/MVI_20011_229_0.03"
folder_path = "Dataset/UA-DETRAC/dehaze_DarkChannel/train/MVI_20011_229_0.03"
output_path = "Test_Output"
output_video_path = "Test_Output/output_video.mp4"

# Get the first image path for frame size information
first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
first_frame = cv2.imread(first_image_path)
height, width, _ = first_frame.shape

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 5.0, (width, height))

# Process each image in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg"):
        img_path = os.path.join(folder_path, file_name)
        print(img_path)
        frame = cv2.imread(img_path)  # BGR
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the video file
        out.write(annotated_frame)

# Release the VideoWriter object
out.release()

print("Video has been saved successfully.")
