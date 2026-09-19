from ultralytics import YOLO

# Load pre-trained YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')

# Train on the newly created 'dataset' folder
results = model.train(
    data='dataset',
    epochs=20,
    imgsz=224,
    project='defect_classifier',
    name='model_v1'
)

print("Training finished! Weights saved in runs/classify/defect_classifier/model_v1/weights/best.pt")