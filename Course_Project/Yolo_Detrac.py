from ultralytics import YOLO
# Load a model
model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Use the model
    model.train(data="Dataset/detrac.yaml", epochs=2)  # train the model
    metrics = model.val(data="Dataset/test.yaml")  # evaluate model performance on the validation set
