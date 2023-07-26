from ultralytics import YOLO




import numpy as np
import imutils
from flask import Flask, render_template, Response
import cv2
import pafy
from datetime import datetime
import pytz
import time
import tensorflow as tf



app = Flask(__name__)




model = YOLO('../yolov8n.pt')
# model = YOLO(r'C:\Users\ariai\Documents\DATA SCIENCE\PROJECTS\PeopleCounterApp\runs\detect\yolov8n_custom4\weights\best.pt')

# model = model.train(data='coco128.yaml', epochs=3)



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



        image = imutils.resize(frame, width=900)
        # image = frame
        def zoom_at(img, zoom=1.2, angle=0, coord=None):

            cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]

            rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

            return result

        image=zoom_at(image)

        results = model.predict(image)
        # print(results[0])
        # for r in results:
        #     for c in r.boxes.cls:
        #         print(model.names[int(c)])

        results=results[0].boxes.boxes
        results = results.numpy()



        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        count = 0
        for res in results:
            # print(res)
            x1, y1, x2, y2, score,label=int(res[0]),int(res[1]),int(res[2]),int(res[3]),int(res[4]*100),int(res[5])
            names = model.model.names
            if label in names:
                label = names[label]
                if label == 'person':
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # cv2.putText(image, f'{label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'{score} %', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    count += 1


        cv2.putText(img=image, text=str(count), org = (950, 160),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5.0,color=(125, 246, 55),thickness = 9)
        cv2.putText(img=image, text="YOLOv8", org=(300, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2,color=(125, 246, 55), thickness=2)
        cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")), (900, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2,cv2.LINE_AA)
        cv2.putText(img=image, text=(str(fps)+' fps'), org=(700, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,color=(125, 246, 55), thickness=2)

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
    app.run(debug=True)






