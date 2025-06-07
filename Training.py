from pathlib import Path
from ultralytics import YOLO

yaml_path = Path("data.yaml")

model = YOLO('yolov8n.pt')

# Treinamento
model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=16,
    project='asl-yolov8',
    name='sign-language-model',
    exist_ok=True
)
