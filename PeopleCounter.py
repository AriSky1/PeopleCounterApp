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
from gen_frames import gen_frames_yolo, gen_frames_hog, gen_frames_mog2
import dash_bootstrap_components as dbc

external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Wix+Madefor+Display:wght@600&display=swap',
     dbc.themes.BOOTSTRAP
]

style_flex={"display":"flex", "align-items":"flex-end", "gap":"20px", 'padding-left': '20px'}
style_dd={'width':'200px','background-color': '#F9F8F9', 'height':'40px', 'font_family': 'Tahoma'}
style_btn = {'background-color': '#7FFF00','color': 'black','font-weight': 'bold', 'width':'200px', 'height':'40px'}
style_text={'color': 'grey','fontSize': 18,'textAlign': 'left','font_family': 'Segoe UI', 'padding-bottom':'20px','padding-left': '20px'}
style_title={'color': 'grey','fontSize': 30,'textAlign': 'left', 'letter-spacing':'2px', 'padding-left': '20px','padding-top': '20px'}

server = Flask(__name__)
app = Dash(__name__, server=server,external_stylesheets=external_stylesheets)


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
    html.Div(id='container', style={'padding-left': '20px'}),


    # html.Img(id='stream',src="/video_feed"),




])

# yolo8 + Shibuya static
@server.route('/yolo8_1')
def yolo8_1():
    url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya static
    model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url, model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# yolo8 + Street walk
@server.route('/yolo8_2')
def yolo8_2():
    url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA'  # street walk
    model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url, model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# yolo8 + Street static
@server.route('/yolo8_3')
def yolo8_3():
    url = 'https://www.youtube.com/watch?v=3kPH7kTphnE'  # street static
    model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url, model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# HOG + Shibuya static
@server.route('/hog_1')
def hog_1():
    url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya static
    return Response(gen_frames_hog(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# HOG + Street walk
@server.route('/hog_2')
def hog_2():
    url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' # street walk
    return Response(gen_frames_hog(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# HOG + Street static
@server.route('/hog_3')
def hog_3():
    url = 'https://www.youtube.com/watch?v=3kPH7kTphnE' # street static
    return Response(gen_frames_hog(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# MOG2 + Shibuya static
@server.route('/mog2_1')
def mog2_1():
    url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya static
    return Response(gen_frames_mog2(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# MOG2 + Street static
@server.route('/mog2_2')
def mog2_2():
    url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' #street walk
    return Response(gen_frames_mog2(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# MOG2 + Street static
@server.route('/mog2_3')
def mog2_3():
    url = 'https://www.youtube.com/watch?v=3kPH7kTphnE' #street static
    return Response(gen_frames_mog2(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# CALLBACKS

@app.callback(Output('container', 'children'),
              Input('submit_btn', 'n_clicks'),
              State('model_dropdown', 'value'),
              State('video_dropdown', 'value'))

def display_stream(n_clicks,cvmodel, video):

    if n_clicks > 0 and cvmodel == 'Yolo8' and video == 'Shibuya static':
        return html.Div([html.Img(id='stream', src="/yolo8_1")])
    if n_clicks > 0 and cvmodel == 'Yolo8' and video == 'Street walk':
        return html.Div([html.Img(id='stream', src="/yolo8_2")])
    if n_clicks > 0 and cvmodel == 'Yolo8' and video == 'Street static':
        return html.Div([html.Img(id='stream', src="/yolo8_3")])
    if n_clicks > 0 and cvmodel == 'HOG' and video == 'Shibuya static':
        return html.Div([html.Img(id='stream', src="/hog_1")])
    if n_clicks > 0 and cvmodel == 'HOG' and video == 'Street walk':
        return html.Div([html.Img(id='stream', src="/hog_2")])
    if n_clicks > 0 and cvmodel == 'HOG' and video == 'Street static':
        return html.Div([html.Img(id='stream', src="/hog_3")])
    if n_clicks > 0 and cvmodel == 'MOG2' and video == 'Shibuya static':
        return html.Div([html.Img(id='stream', src="/mog2_1")])
    if n_clicks > 0 and cvmodel == 'MOG2' and video == 'Street walk':
        return html.Div([html.Img(id='stream', src="/mog2_2")])
    if n_clicks > 0 and cvmodel == 'MOG2' and video == 'Street static':
        return html.Div([html.Img(id='stream', src="/mog2_3")])










if __name__ == '__main__':
    app.run_server(debug=True)