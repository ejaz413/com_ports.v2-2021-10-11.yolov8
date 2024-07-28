import os
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

ROOT_DIR = '/home/ejaz/com_ports.v2-2021-10-11.yolov8'

results = model.train(data = os.path.join(ROOT_DIR, 'data.yaml'), epochs=100)
