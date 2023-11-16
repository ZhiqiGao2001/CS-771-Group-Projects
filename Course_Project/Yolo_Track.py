from ultralytics import YOLO
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def process_folder(folder_path, output_path, output_type='video'):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Get the first image path for frame size information
    first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape

    # Create VideoWriter object if output_type is video
    if output_type == 'video':
        output_video_path = os.path.join(output_path, 'output_video.mp4')
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

            if output_type == 'video':
                # Write the frame to the video file
                out.write(annotated_frame)
            elif output_type == 'images':
                # Save each frame as a separate image file
                output_image_path = os.path.join(output_path, f'{file_name[:-4]}.jpg')
                cv2.imwrite(output_image_path, annotated_frame)

    # Release the VideoWriter object if output_type is video
    if output_type == 'video':
        out.release()
        print("Video has been saved successfully.")
    elif output_type == 'images':
        print("Images have been saved successfully.")


# Example usage:
folder_path = "Dataset/UA-DETRAC/dehaze_DarkChannel/train/MVI_20011_229_0.03"
output_path_video = "Test_Output"
output_path_images = "Test_Output"

# # Process the folder and store the results as a video
# process_folder(folder_path, output_path_video, output_type='video')

# Process the same folder and store the results as separate image files
process_folder(folder_path, output_path_images, output_type='images')

