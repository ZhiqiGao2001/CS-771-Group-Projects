import cv2
import os
import numpy as np

# video = cv2.VideoWriter('ss.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #创建视频流对象-格式二

"""
参数1 即将保存的文件路径
参数2 VideoWriter_fourcc为视频编解码器
    fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
    cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv 
    cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    文件名后缀为.mp4
参数3 为帧播放速率
参数4 (width,height)为视频帧大小

"""

def create_video(input_folder, output_file, fps=25/7):
    image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
    image = cv2.imread(image_path)
    image_info = image.shape
    height = image_info[0]
    width = image_info[1]
    size = (height, width)
    print(size)

    fps = 25/7
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(input_folder, file_name)
            image = cv2.imread(img_path)
            video.write(image)

    video.release()

# Example usage:
folder_path = "Dataset/UA-DETRAC/dehaze_DarkChannel/test/MVI_39031_241_0.03"
output_file = 'Test_Output/test_video.mp4'
create_video(folder_path, output_file)

