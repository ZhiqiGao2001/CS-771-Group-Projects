import os

from ultralytics import YOLO
import yaml
import sys
import logging


# Load a model
# model = YOLO("models/yolov8m.pt")  # load a pretrained model (recommended for training)
model = YOLO("C:/Users/ZhiQi/OneDrive/桌面/yolov5/runs/detect/yolov8m_gt/train/weights/best.pt")  # or initialize a model from .pt file


if __name__ == '__main__':
    # Use the model
    # epochs = 10
    # model.train(data="Dataset/detrac_train.yaml", epochs=epochs)  # train the model
    # metrics = model.val(data="Dataset/trained_test.yaml")  # evaluate model performance on the validation set

    test_yaml = "Dataset/trained_test.yaml"

    with open(test_yaml, 'r') as file:
        yaml_content = file.read()
    data = yaml.safe_load(yaml_content)
    path_value = data.get('path', '')
    name = path_value.split('\\')[-1]  # Adjust the separator based on your platform
    print(name)
    text_path = "C:/Users/ZhiQi/OneDrive/桌面/yolov5/runs/detect/"+name+'/'+name+".txt"
    metrics = model.val(data=test_yaml, name=name)
    open(text_path, 'w')
    os.startfile(text_path)
