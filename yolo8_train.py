from ultralytics import YOLO
import onnx

model = YOLO('yolov8n.pt')

results = model.train(data='coco128.yaml', epochs=3)
#
# results = model.val()


success = model.export(format='onnx')