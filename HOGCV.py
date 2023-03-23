import imutils
from flask import Flask, request, render_template, Response
import cv2
import time
import pafy
from vidgear.gears import CamGear
from cap_from_youtube import cap_from_youtube
import numpy as np
from imutils.object_detection import non_max_suppression


# Declare a Flask app
app = Flask(__name__)

url = "https://www.youtube.com/watch?v=lMOtsTGef38"
# url = "https://www.youtube.com/watch?v=gC4dJGHWwDU"
url = "https://www.youtube.com/watch?v=Cp2Ku8sUV_4"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)



sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def gen_frames():



    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = frame.shape[1]
        max_width = 600
        if width > max_width:
            frame = imutils.resize(frame, width=max_width)
            print(frame.shape)

        pedestrians, weights = hog.detectMultiScale(frame, winStride=(0, 0),padding=(8, 8), scale=1.05)
        pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])


        count = 0
        #  Draw bounding box over detected pedestrians
        for x, y, w, h in pedestrians:
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 100), 2)
            cv2.rectangle(frame, (x, y - 20), (w, y), (0, 0, 255), -1)
            cv2.putText(frame, f'P{count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            count += 1
        pedestrians = non_max_suppression(pedestrians, probs=None, overlapThresh=0.5)
        print(pedestrians)
        cv2.putText(img=frame, text=str(count), org =(520, 65),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0,color=(0, 0, 255),thickness = 2)


        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        # frame = cv2.imencode('.jpg', fgmask)[1].tobytes()



        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
           break






# Main function here
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# Running the app
if __name__ == '__main__':
    app.run(debug = True)