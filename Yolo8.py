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




model = YOLO('yolov8n.pt')

# model = model.train(data='coco128.yaml', epochs=3)



# url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya crossing static
# url = 'https://www.youtube.com/watch?v=1-iS7LArMPA' #time square static
# url = 'https://www.youtube.com/watch?v=3kPH7kTphnE' #street static
# url = 'https://www.youtube.com/watch?v=b3yQXprMj3s' #districts walking record
url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA' #disctricts walking live


video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)



NMS_THRESHOLD=0
MIN_CONFIDENCE=0


def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # non-maxima suppression to suppress weak, overlapping bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    # ensure at least one detection exists
    if len(idzs) > 0:
        # loop over the indexes we are keeping
        for i in idzs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    return results


# labelsPath = "coco.names"
# LABELS = open(labelsPath).read().strip().split("\n")
#
# weights_path = "yolov4-tiny.weights"
# config_path = "yolov4-tiny.cfg"
#
# model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# '''
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# '''
#
#
# layer_name = model.getLayerNames()
# # layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
# layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

writer = None




def gen_frames():
    prev_frame_time = 0
    new_frame_time = 0

    while(cap.isOpened()):
        grabbed, frame = cap.read()

        # labelsPath = "coco.names"
        # LABELS = open(labelsPath).read().strip().split("\n")
        #
        # weights_path = "yolov4-tiny.weights"
        # config_path = "yolov4-tiny.cfg"
        #
        # model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        # '''
        # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # '''








        # layer_name = model.getLayerNames()
        # # layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        # layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
        #
        # writer = None

        image = imutils.resize(frame, width=1300)

        results = model.predict(image)
        # print(results[0])
        # for r in results:
        #     for c in r.boxes.cls:
        #         print(model.names[int(c)])

        results=results[0].boxes.boxes
        # print(results)

        # results = pedestrian_detection(image, model, layer_name,personidz=LABELS.index("person"))


        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        count = 0
        for res in results:
            res=res.numpy()
            # print(res)
            x1, y1, x2, y2, score,label=int(res[0]),int(res[1]),int(res[2]),int(res[3]),int(res[4]*100),int(res[5])



            # cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y1), (x2,y2), (0, 255, 0), 2)
            prob=score
            print(score)
            # prob *= 100
            # prob = round(prob)
            cv2.putText(image, f'{prob} %', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            count += 1


        cv2.putText(img=image, text=str(count), org = (950, 160),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5.0,color=(125, 246, 55),thickness = 9)
        cv2.putText(img=image, text="YOLOv8", org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2,color=(125, 246, 55), thickness=2)
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






