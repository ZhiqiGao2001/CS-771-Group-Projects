from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip, concatenate_videoclips
import cv2
import os
import numpy as np


# def create_video(input_folder, output_file):
#     image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
#     image = cv2.imread(image_path)
#     image_info = image.shape
#     height = image_info[0]
#     width = image_info[1]
#     size = (height, width)
#     print(size)
#
#     fps = 25/10
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
#
#     for file_name in os.listdir(input_folder):
#         if file_name.endswith(".png"):
#             img_path = os.path.join(input_folder, file_name)
#             image = cv2.imread(img_path)
#             video.write(image)
#
#     video.release()
#
# # Example usage:
# folder_path = "Test_Output"
# output_file = 'annotated_video.mp4'
# create_video(folder_path, output_file)

def merge_videos(video1_path, video2_path, output_path):
    # Load the video clips
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)

    # Resize clips to have the same height
    min_height = min(clip1.size[1], clip2.size[1])
    clip1 = clip1.resize(height=min_height)
    clip2 = clip2.resize(height=min_height)

    # Create a video clip with side-by-side arrangement
    final_clip = clips_array([[clip1, clip2]])

    # Write the output video file
    final_clip.write_videofile(output_path, codec="libx264", fps=24)


if __name__ == "__main__":
    # Replace these paths with the paths to your input videos
    video1_path = "Test_Output_original/output_video1.mp4"
    video2_path = "Test_Output_trained/output_video1.mp4"
    annotation_path = "annotated_video.mp4"

    # Replace this path with the desired output path
    output_path = "merged_video.mp4"

    merge_videos(video1_path, video2_path, output_path)
