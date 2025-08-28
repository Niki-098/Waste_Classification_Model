from ultralytics import YOLO

# Load pre-trained YOLOv8 nano
model = YOLO('yolov8n.pt')  
device = 0
batch_size = 16

results = model.train(
    data='/content/data.yaml',
    epochs=50,
    imgsz=640,
    batch=batch_size,
    device=device,
    name='scrap_detector'
)

