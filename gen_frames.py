
import pafy
import cv2
import time
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from datetime import datetime
import pytz


def gen_frames_yolo(url, model):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    # width = 320
    # height = 240
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    FPS = 1 / 7
    FPS_MS = int(FPS * 1000)
    prev_frame_time = 0
    new_frame_time = 0

    while (cap.isOpened()):
        grabbed, frame = cap.read()
        image = imutils.resize(frame, width=900)


        def zoom_at(img, zoom=1.2, angle=0, coord=None):
            cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
            rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        image = zoom_at(image)
        results = model.predict(image)
        results = results[0].boxes.boxes
        results = results.numpy()
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        count = 0
        for res in results:
            # print(res)
            x1, y1, x2, y2, score, label = int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4] * 100), int(
                res[5])
            names = model.model.names
            if label in names:
                label = names[label]
                if label == 'person':
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # cv2.putText(image, f'{label}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'{score} %', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    count += 1

        cv2.putText(img=image, text=str(count), org=(600, 160), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4.0,
                    color=(125, 246, 55), thickness=7)
        cv2.putText(img=image, text=(str(fps) + ' fps'), org=(750, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,
                    color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break

    cv2.waitKey(FPS_MS)


def gen_frames_hog(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)

    # HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    prev_frame_time = 0
    new_frame_time = 0
    while (cap.isOpened()):
        ret, frame = cap.read()  # import image
        image = imutils.resize(frame, width=900)
        # image = cv2.resize(frame, (0, 0), None, 1, 1)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

        # pedestrians, weights = hog.detectMultiScale(frame, winStride=(3,3),padding=(2, 2), scale=2)
        pedestrians, weights = hog.detectMultiScale(frame)
        pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])

        count = 0
        pedestrians = non_max_suppression(pedestrians, probs=None, overlapThresh=0.8)
        for x, y, w, h in pedestrians:
            cv2.rectangle(image, (x, y), (w, h), (125, 246, 55), 2)

            count += 1

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(img=image, text=str(count), org=(600, 160), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4.0,
                    color=(125, 246, 55), thickness=7)
        cv2.putText(img=image, text=(str(fps) + ' fps'), org=(750, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,
                    color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break

def gen_frames_mog2(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    FPS = 1 / 15
    FPS_MS = int(FPS * 1000)
    # background subtractor
    sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)


    sub.setShadowValue(0)
    # Sets shadow

    sub.setVarThresholdGen(100)

    # Sets the variance threshold for the pixel-model match used for new mixture component generation.

    prev_frame_time = 0
    new_frame_time = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # image = cv2.resize(frame, (0, 0), None, 1, 1)
        image = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        curr_img = sub.apply(
            gray)  # subtraction between the current frame and a background model, containing the static part of the scene
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # low quality vid> no kernel!
        # curr_img = cv2.morphologyEx(curr_img, cv2.MORPH_CLOSE, kernel)  # dots inside
        # curr_img = cv2.morphologyEx(curr_img, cv2.MORPH_OPEN, kernel, iterations=1)  # dots outside
        # curr_img = cv2.dilate(curr_img, kernel)

        contours, hierarchy = cv2.findContours(curr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        minarea = 80
        maxarea = 200

        count = 1
        for i in range(len(contours)):  # cycles through all contours in current frame

            if hierarchy[
                0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # x,y is top left corner and w,h is width and height
                    x, y, w, h = cv2.boundingRect(cnt)

                    # cv2.rectangle(curr_img, (x, y), (x + w, y + h), (255, 255, 255), 1) #white
                    cv2.rectangle(image, (x, y), (x + w, y + h), (125, 246, 55), 2)  # green
                    count += 1

        cv2.putText(img=image, text=str(count), org=(600, 160), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4.0,
                    color=(125, 246, 55), thickness=7)
        cv2.putText(img=image, text=(str(fps) + ' fps'), org=(750, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,
                    color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # frame = cv2.imencode('.jpg', curr_img)[1].tobytes() #see preprocessed image

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break
    time.sleep(FPS)
    # cv2.waitKey(FPS_MS)
