from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('yolov8n.pt')

# results = model.train(data='coco128.yaml', epochs=3)
#
# results = model.val()

im1 = Image.open("test_image.png")
# success = model.export(format='onnx')
results = model.predict(source=im1)
print(results)