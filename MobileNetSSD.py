import imutils
from flask import Flask, request, render_template, Response
import cv2
import time
import pafy
from vidgear.gears import CamGear
from cap_from_youtube import cap_from_youtube
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from datetime import datetime, time
import numpy as np
import time as time2



# Declare a Flask app
app = Flask(__name__)


prott1 = r'MobileNetSSD_deploy.prototxt.txt'
prott2 = r'MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prott1, prott2)

CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

url = "https://www.youtube.com/watch?v=lMOtsTGef38"
# url = "https://youtu.be/lMOtsTGef38"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

ret, frame = cap.read()  # import image



def gen_frames():
    while(cap.isOpened()):
        image = cv2.resize(frame, (0, 0), None, 1, 1)
        # print(image.shape) # (720, 1280, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        # print(gray.shape) # (720, 1280)
        sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor
        fgmask = sub.apply(gray)  # subtraction between the current frame and a background model, containing the static part of the scene
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))  # kernel to apply to the morphology
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))

        count = len(contours)

        minarea = 400
        maxarea = 50000


        for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]

                        confidence_level = 0.7

                        if confidence > confidence_level:
                            # extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of
                            # the bounding box for the object
                            idx = int(detections[0, 0, i, 1])
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            # draw the prediction on the frame
                            label = "{}: {:.2f}%".format(CLASSES[idx],
                                                         confidence * 100)
                            cv2.rectangle(frame, (startX, startY), (endX, endY),
                                          COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                            # Start tracker
                            now = datetime.now()
                            if differ == None or differ > 9:
                                tracker = cv2.TrackerGOTURN_create()
                                initBB2 = None
                                tracker.init(frame, initBB2)
                                fps = FPS().start()

                            # check to see if we are currently tracking an object, if so, ignore other boxes
                            # this code is relevant if we want to identify particular persons (section 2 of this tutorial)
                            if initBB2 is not None:

                                # grab the new bounding box coordinates of the object
                                (success, box) = tracker.update(frame)

                                # check to see if the tracking was a success
                                differ = 10
                                if success:
                                    (x, y, w, h) = [int(v) for v in box]
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    differ = abs(initBB2[0] - box[0]) + abs(initBB2[1] - box[1])
                                    i = tracker.update(frame)
                                    if i[0] != True:
                                        time2.sleep(4000)
                                else:
                                    trackeron = 1

                                # update the FPS counter
                                fps.update()
                                fps.stop()

                            # initialize the set of information we'll be displaying on
                            # the frame
                            info = [
                                ("Success", "Yes" if success else "No"),
                                ("FPS", "{:.2f}".format(fps.fps())),
                            ]

                            # loop over the info tuples and draw them on our frame
                            for (i, (k, v)) in enumerate(info):
                                text = "{}: {}".format(k, v)
                                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                                            2)

                            # draw the text and timestamp on the frame
                            now2 = datetime.now()
                            time_passed_seconds = str((now2 - now).seconds)
                            cv2.putText(frame, 'Detecting persons', (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
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