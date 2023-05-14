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
from gen_frames import gen_frames


style_flex={"display":"flex", "align-items":"flex-end", "gap":"20px",}
style_dd={'width':'200px','background-color': '#F9F8F9', 'height':'40px', 'font_family': 'Tahoma'}
style_btn = {'background-color': '#7FFF00','color': 'black','font-weight': 'bold', 'width':'200px', 'height':'40px'}
style_text={'color': 'grey','fontSize': 18,'textAlign': 'left','font_family': 'Segoe UI', 'padding-bottom':'20px'}
style_title={'color': 'grey','fontSize': 30,'textAlign': 'left','font_family': 'Tahoma', 'letter-spacing':'2px'}

server = Flask(__name__)
app = Dash(__name__, server=server)


# model = YOLO('yolov8n.pt')
# # url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' #disctricts walking live
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# cap = cv2.VideoCapture(best.url)
#
# def gen_frames():
#     prev_frame_time = 0
#     new_frame_time = 0
#
#     while(cap.isOpened()):
#         grabbed, frame = cap.read()
#         frame = imutils.resize(frame, width=800)
#         image = frame
#         def zoom_at(img, zoom=1.2, angle=0, coord=None):
#             cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
#             rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
#             result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
#             return result
#         image=zoom_at(image)
#         results = model.predict(image)
#         results=results[0].boxes.boxes
#         results = results.numpy()
#         new_frame_time = time.time()
#         fps = 1 / (new_frame_time - prev_frame_time)
#         prev_frame_time = new_frame_time
#         fps = int(fps)
#         fps = str(fps)
#         count = 0
#         for res in results:
#             # print(res)
#             x1, y1, x2, y2, score,label=int(res[0]),int(res[1]),int(res[2]),int(res[3]),int(res[4]*100),int(res[5])
#             names = model.model.names
#             if label in names:
#                 label = names[label]
#                 if label == 'person':
#                     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#                     # cv2.putText(image, f'{label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                     cv2.putText(image, f'{score} %', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                     count += 1
#
#
#         cv2.putText(img=image, text=str(count), org = (950, 160),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5.0,color=(125, 246, 55),thickness = 9)
#         cv2.putText(img=image, text="YOLOv8", org=(300, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2,color=(125, 246, 55), thickness=2)
#         cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")), (900, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2,cv2.LINE_AA)
#         cv2.putText(img=image, text=(str(fps)+' fps'), org=(700, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,color=(125, 246, 55), thickness=2)
#
#         frame = cv2.imencode('.jpg', image)[1].tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         key = cv2.waitKey(20)
#         if key == 27:
#             break














app.layout = html.Div(children=[
    html.H1(children='People Counter', style=style_title),
html.Div(children='''
        Try live streams VS computer vision models.
    ''', style=style_text),
    html.Div([
              html.Div([dcc.Dropdown(['Yolo8', 'MOG2', 'HOG'], 'Yolo8', id='model_dropdown'),], style=style_dd),
    html.Div([dcc.Dropdown(['Shibuya static', 'Street walk', 'Street static'], 'Street walk', id='video_dropdown'),], style=style_dd),
    html.Div([html.Button('Count',id='submit_btn',n_clicks=0,style=style_btn),], ),],

             style=style_flex),
    html.Br(),
    html.Div(id='container'),


    # html.Img(id='stream',src="/video_feed"),




])

# yolo8 + Shibuya static
@server.route('/video_feed')
def video_feed():
    url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya static
    model = YOLO('yolov8n.pt')
    return Response(gen_frames(url, model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# yolo8 + Street walk
@server.route('/video_feed2')
def video_feed2():
    url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA'  # street walk
    model = YOLO('yolov8n.pt')
    return Response(gen_frames(url, model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# yolo8 + Street static
@server.route('/video_feed3')
def video_feed3():
    url = 'https://www.youtube.com/watch?v=3kPH7kTphnE'  # street static
    model = YOLO('yolov8n.pt')
    return Response(gen_frames(url, model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# CALLBACKS

@app.callback(Output('container', 'children'),
              Input('submit_btn', 'n_clicks'),
              State('model_dropdown', 'value'),
              State('video_dropdown', 'value'))

def display_stream(n_clicks,cvmodel, video):
    # global url
    # global model
    #
    # if cvmodel == 'Yolo8':
    #     model = YOLO('yolov8n.pt')
    # if cvmodel == "MOG2" :
    #     model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    # if video == 'Shibuya static':
    #     url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc'
    # if video == 'Street walk':
    #     url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' #disctricts walking live

    if n_clicks > 0 and cvmodel == 'Yolo8' and video == 'Shibuya static':
        return html.Div([html.Img(id='stream', src="/video_feed")])
    if n_clicks > 0 and cvmodel == 'Yolo8' and video == 'Street walk':
        return html.Div([html.Img(id='stream', src="/video_feed2")])
    if n_clicks > 0 and cvmodel == 'Yolo8' and video == 'Street static':
        return html.Div([html.Img(id='stream', src="/video_feed3")])










if __name__ == '__main__':
    app.run_server(debug=True)