from moviepy.editor import VideoFileClip, clips_array

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

    # Replace this path with the desired output path
    output_path = "Test_Output/merged_video.mp4"

    merge_videos(video1_path, video2_path, output_path)