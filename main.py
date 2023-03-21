from flask import Flask, request, render_template, Response
import cv2
import time
import pafy
from vidgear.gears import CamGear
from cap_from_youtube import cap_from_youtube


# Declare a Flask app
app = Flask(__name__)

# Main function here
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():

    url = "https://www.youtube.com/watch?v=lMOtsTGef38"
    # url = "https://youtu.be/lMOtsTGef38"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)


    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# Running the app
if __name__ == '__main__':
    app.run(debug = True)