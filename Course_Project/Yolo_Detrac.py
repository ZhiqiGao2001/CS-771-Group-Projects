from ultralytics import YOLO
# Load a model
model = YOLO("models/yolov8m.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Use the model
    epochs = 10
    model.train(data="Dataset/detrac_train.yaml", epochs=epochs)  # train the model
    metrics = model.val(data="Dataset/trained_test.yaml")  # evaluate model performance on the validation set
