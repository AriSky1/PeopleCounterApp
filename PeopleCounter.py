from dash import Dash, html, dcc
from ultralytics import YOLO
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from flask import Flask, render_template, Response
from gen_frames import gen_frames_yolo, gen_frames_hog, gen_frames_mog2
import dash_bootstrap_components as dbc

modelyolo = YOLO('yolov8n.pt')



external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Wix+Madefor+Display:wght@600&display=swap',
     dbc.themes.CYBORG
]

style_flex={"display":"inline-flex", "align-items":"flex-end", "gap":"20px", 'padding-left': '20px'}
style_dd={'width':'200px','background-color': '#696969', 'height':'20px', 'font_family': 'Tahoma','padding-bottom':'50px', }
style_btn = {'background-color': '#7FFF00','color': 'black','font-weight': 'bold', 'width':'200px', 'height':'40px',}
style_text={'color': 'grey','fontSize': 18,'textAlign': 'center','font_family': 'Segoe UI', 'padding-bottom':'20px','padding-left': '20px'}
style_title={'color': 'grey','fontSize': 30,'textAlign': 'center', 'letter-spacing':'2px', 'padding-left': '20px','padding-top': '20px'}
style_input={'width':'200px', 'height':'40px', 'font_family': 'Tahoma',}
style_menu={'padding-left': '20px', 'width':'250px'}

server = Flask(__name__)
app = Dash(__name__, server=server,external_stylesheets=external_stylesheets)
app.css.append_css({'external_url': '/static/styles.css'})
app.server.static_folder = 'static'

app.layout = html.Div(children=[
    html.H1(children='People Counter', style=style_title),
    html.Div(children='''
        Try the fastest computer vision models 
        on real-time streams or 
        YouTube videos.
    ''', style=style_text),

    html.Div([

    html.Div([


    html.Plaintext('Chose a model : '),
    html.Div([dcc.Dropdown(['YOLOv8', 'MOG2'], 'YOLOv8', id='model_dropdown',style=style_dd),], ),
    html.Br(),html.Br(),
    html.Plaintext('Chose a stream : '),
    html.Div([dcc.Dropdown([ 'Street walk', 'Street static','Shibuya static',], 'Street walk', id='video_dropdown',style=style_dd),], ),
html.Br(),html.Br(),
    html.Plaintext('or  ', style={'text-align':'center'}),
    html.Div([dcc.Input(id='input', type='text', placeholder='Paste YouTube link',)],style=style_input ),
    html.Br(),html.Br(),
    html.Div([html.Button('Count',id='submit_btn',n_clicks=0,style=style_btn),], ),], style=style_menu,

             ),

    html.Div(id='container', ),], style={'display':'inline-flex'}),


], style={'backgroundColor': '#111111', 'height':'700px'})



# yolo8 + Shibuya static
@server.route('/yolo8_1')
def yolo8_1():
    url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya static
    # model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url, modelyolo),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# yolo8 + Street walk
@server.route('/yolo8_2')
def yolo8_2():
    url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA'  # street walk
    # model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url, modelyolo),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# yolo8 + Street static
@server.route('/yolo8_3')
def yolo8_3():
    url = 'https://www.youtube.com/watch?v=3kPH7kTphnE'  # street static
    # model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url, modelyolo),
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



# MOG2 + inputs
@server.route('/mog2_4')
def mog2_4():
    url = input
    return Response(gen_frames_mog2(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Yolo8 + input
@server.route('/yolo8_4')
def yolo8_4():
    url = input
    model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(url,model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# HOG + input
@server.route('/hog_4')
def hog_4():
    url = input
    return Response(gen_frames_hog(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# CALLBACKS

@app.callback(Output('container', 'children'),
              Input('submit_btn', 'n_clicks'),
              Input("input", "value"),
              State('model_dropdown', 'value'),
              State('video_dropdown', 'value'))

def display_stream(n_clicks,user_input,cvmodel, video):
    global input


    if user_input is not None and cvmodel == 'YOLOv8':
        input=user_input
        return html.Div([html.Img(id='stream', src="/yolo8_4")])
    # if user_input is not None and cvmodel == 'HOG':
    #     input=user_input
    #     return html.Div([html.Img(id='stream', src="/hog_4")])
    if user_input is not None and cvmodel == 'MOG2':
        input=user_input
        return html.Div([html.Img(id='stream', src="/mog2_4")])


    if n_clicks > 0 and cvmodel == 'YOLOv8' and video == 'Shibuya static':
        return html.Div([html.Img(id='stream', src="/yolo8_1")])
    if n_clicks > 0 and cvmodel == 'YOLOv8' and video == 'Street walk':
        return html.Div([html.Img(id='stream', src="/yolo8_2")])
    if n_clicks > 0 and cvmodel == 'YOLOv8' and video == 'Street static':
        return html.Div([html.Img(id='stream', src="/yolo8_3")])
    # if n_clicks > 0 and cvmodel == 'HOG' and video == 'Shibuya static':
    #     return html.Div([html.Img(id='stream', src="/hog_1")])
    # if n_clicks > 0 and cvmodel == 'HOG' and video == 'Street walk':
    #     return html.Div([html.Img(id='stream', src="/hog_2")])
    # if n_clicks > 0 and cvmodel == 'HOG' and video == 'Street static':
    #     return html.Div([html.Img(id='stream', src="/hog_3")])
    if n_clicks > 0 and cvmodel == 'MOG2' and video == 'Shibuya static':
        return html.Div([html.Img(id='stream', src="/mog2_1")])
    if n_clicks > 0 and cvmodel == 'MOG2' and video == 'Street walk':
        return html.Div([html.Img(id='stream', src="/mog2_2")])
    if n_clicks > 0 and cvmodel == 'MOG2' and video == 'Street static':
        return html.Div([html.Img(id='stream', src="/mog2_3")])










if __name__ == '__main__':
    app.run_server(debug=True)