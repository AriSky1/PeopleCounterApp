from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
from dash import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash import ctx
from dash.dependencies import Input, Output, State
import pafy
import cv2
from flask import Flask, render_template, Response
from datetime import datetime
import pytz
import time
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression


def gen_frames_yolo(url, model):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    prev_frame_time = 0
    new_frame_time = 0

    while (cap.isOpened()):
        grabbed, frame = cap.read()
        frame = imutils.resize(frame, width=800)
        image = frame

        def zoom_at(img, zoom=1.2, angle=0, coord=None):
            cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
            rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        image = zoom_at(image)
        results = model.predict(image)
        results = results[0].boxes.boxes
        results = results.numpy()
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        count = 0
        for res in results:
            # print(res)
            x1, y1, x2, y2, score, label = int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4] * 100), int(
                res[5])
            names = model.model.names
            if label in names:
                label = names[label]
                if label == 'person':
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # cv2.putText(image, f'{label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'{score} %', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    count += 1

        cv2.putText(img=image, text=str(count), org=(600, 160), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4.0,
                    color=(125, 246, 55), thickness=7)
        # cv2.putText(img=image, text="YOLOv8", org=(300, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2,
        #             color=(125, 246, 55), thickness=2)
        # cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")), (900, 30),
        #             cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2, cv2.LINE_AA)
        cv2.putText(img=image, text=(str(fps) + ' fps'), org=(700, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,
                    color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


def gen_frames_hog(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)

    # HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    prev_frame_time = 0
    new_frame_time = 0
    while (cap.isOpened()):
        ret, frame = cap.read()  # import image
        frame = imutils.resize(frame, width=800)
        image = cv2.resize(frame, (0, 0), None, 1, 1)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

        # pedestrians, weights = hog.detectMultiScale(frame, winStride=(4,4),padding=(4, 4), scale=1.5)
        pedestrians, weights = hog.detectMultiScale(frame)
        pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])

        count = 0
        pedestrians = non_max_suppression(pedestrians, probs=None, overlapThresh=0.8)
        for x, y, w, h in pedestrians:
            cv2.rectangle(image, (x, y), (w, h), (0, 0, 100), 2)
            cv2.rectangle(image, (x, y - 20), (w, y), (0, 0, 255), -1)
            cv2.putText(image, f'P{count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            count += 1

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(img=image, text=str(count), org=(600, 160), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4.0,
                    color=(125, 246, 55), thickness=7)
        cv2.putText(img=image, text=str(count), org=(950, 160), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=5.0,
                    color=(125, 246, 55), thickness=9)
        # cv2.putText(img=image, text="Detection model : HOG", org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #             fontScale=1.2, color=(125, 246, 55), thickness=3)
        # cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")),
        #             (900, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2, cv2.LINE_AA)
        cv2.putText(img=image, text=(str(fps) + ' fps'), org=(700, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,
                    color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break
