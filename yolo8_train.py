from ultralytics import YOLO
from PIL import Image
import cv2

# model = YOLO('yolov8n.pt')
#
# # results = model.train(data='coco128.yaml', epochs=3)
# #
# # results = model.val()
#
# im1 = Image.open("test_image.png")
# # success = model.export(format='onnx')
# results = model.predict(source=im1)
# print(results)

from roboflow import Roboflow
from credentials import api_key, workspace,project


# yolo task=detect \
# mode=train \
# model=yolov8s.pt \
# data={dataset.location}/data.yaml \
# epochs=100 \
# imgsz=640

rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project)
dataset = project.version(1).download("yolov8")


from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')

# Training.
results = model.train(
    data=r'C:\Users\ariai\Documents\DATA SCIENCE\PROJECTS\PeopleCounterApp\PeopleCounter.v1i.yolov8\data.yaml',
    imgsz=640,
    epochs=10,
    batch=8,
    name='yolov8n_custom')

print(results)

# yolo detect train data=/PeopleCounter-1/data.yaml model=yolov8n.pt epochs=100 imgsz=640