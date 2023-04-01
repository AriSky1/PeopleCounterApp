import imutils
from flask import Flask, render_template, Response, request
import cv2
import pafy
from datetime import datetime
import pytz
import time
from Yolo4 import pedestrian_detection

app = Flask(__name__)




# url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya crossing static
# url = 'https://www.youtube.com/watch?v=1-iS7LArMPA' #time square static
# url = 'https://www.youtube.com/watch?v=3kPH7kTphnE' #street static
# url = 'https://www.youtube.com/watch?v=b3yQXprMj3s' #districts walking record
url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' #disctricts walking live


video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)


def gen_frames():
    prev_frame_time = 0
    new_frame_time = 0

    while(cap.isOpened()):
        grabbed, frame = cap.read()

        labelsPath = "coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")

        weights_path = "yolov4-tiny.weights"
        config_path = "yolov4-tiny.cfg"

        model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        '''
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        '''

        layer_name = model.getLayerNames()
        # layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

        writer = None

        image = imutils.resize(frame, width=1300)
        results = pedestrian_detection(image, model, layer_name,personidz=LABELS.index("person"))


        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        count = 0
        for res in results:
            cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
            prob=results[0][0]
            prob *= 100
            prob = round(prob)
            cv2.putText(image, f'{prob} %', (res[1][0], res[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            count += 1


        cv2.putText(img=image, text=str(count), org = (950, 160),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5.0,color=(125, 246, 55),thickness = 9)
        cv2.putText(img=image, text="YOLOv4-tiny", org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2,color=(125, 246, 55), thickness=2)
        cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")), (900, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2,cv2.LINE_AA)
        cv2.putText(img=image, text=(str(fps)+' fps'), org=(700, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break








@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# Main function here
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get['submit_btn'] == 'submit':
            return render_template('index.html')
    else:
        return render_template('base.html')



if __name__ == '__main__':
    app.run(debug=True)