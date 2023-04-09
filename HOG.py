import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from flask import Flask, request, render_template, Response
import cv2
import pafy
from datetime import datetime
import pytz
import time


app = Flask(__name__)


# url = 'https://www.youtube.com/watch?v=T5uyWFmW-vg' #short video japan moving
url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya crossing static
# url = 'https://www.youtube.com/watch?v=3kPH7kTphnE' #street static
# url = 'https://www.youtube.com/watch?v=1-iS7LArMPA' #time square static
# url = 'https://www.youtube.com/watch?v=b3yQXprMj3s' #districts walking record
# url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' #disctricts walking live


video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)



# HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def gen_frames():
    prev_frame_time = 0
    new_frame_time = 0
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        image = cv2.resize(frame, (0, 0), None, 1, 1)
        image = imutils.resize(image, width=1300)
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
        cv2.putText(img=frame, text=str(count), org =(1500, 65),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 8.0,color=(0, 0, 255),thickness = 2)
        cv2.putText(img=image, text=str(count), org = (950, 160),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5.0,color=(125, 246, 55),thickness = 9)
        cv2.putText(img=image, text="Detection model : HOG", org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2,color=(125, 246, 55), thickness=3)
        cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")), (900, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2,cv2.LINE_AA)
        cv2.putText(img=image, text=(str(fps)+' fps'), org=(600, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()



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