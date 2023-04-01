
from flask import Flask, render_template, Response
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


# background subtractor
sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)


# sub.setBackgroundRatio(0.6)
#Sets the "background ratio" parameter of the algorithm.

# sub.setComplexityReductionThreshold(0.01) #default=0.05
#defines the number of samples needed to accept to prove the component exists.
# By setting CT=0 you get an algorithm very similar to the standard Stauffer&Grimson algorithm.

# sub.setVarThreshold(50)
#Sets the variance threshold for the pixel-model match.

# sub.setNMixtures(8)
#Sets the number of gaussian components in the background model.

sub.setShadowValue(0)
#Sets shadow

sub.setVarThresholdGen(50)
#Sets the variance threshold for the pixel-model match used for new mixture component generation.



def gen_frames():
    prev_frame_time = 0
    new_frame_time = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        image = cv2.resize(frame, (0, 0), None, 1, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        curr_img = sub.apply(gray)  # subtraction between the current frame and a background model, containing the static part of the scene
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # low quality vid> no kernel!
        curr_img = cv2.morphologyEx(curr_img, cv2.MORPH_CLOSE, kernel) #dots inside
        curr_img = cv2.morphologyEx(curr_img, cv2.MORPH_OPEN, kernel, iterations=1) # dots outside
        # curr_img = cv2.erode(curr_img, kernel,iterations=5)
        # curr_img = cv2.morphologyEx(curr_img, cv2.MORPH_BLACKHAT, kernel)
        # curr_img = cv2.morphologyEx(curr_img, cv2.MORPH_GRADIENT, kernel) #shaped outlines
        curr_img = cv2.dilate(curr_img, kernel)
        # ret, curr_img = cv2.threshold(curr_img, 100, 140, cv2.THRESH_BINARY)  # removes the shadows
        # curr_img = cv2.adaptiveThreshold(curr_img,200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)  #
        # curr_img = cv2.GaussianBlur(curr_img, (1, 1), 0)
        # ret2, curr_img = cv2.threshold(curr_img, 1000, 2000, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(curr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)



        minarea = 80
        maxarea = 300


        count=1
        for i in range(len(contours)):  # cycles through all contours in current frame

            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

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
                    count+=1

        cv2.putText(img=image, text=str(count), org = (950, 160),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 5.0,color=(125, 246, 55),thickness = 9)
        cv2.putText(img=image, text="BackgroundSubtractorMOG2", org=(20, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,color=(125, 246, 55), thickness=2)
        cv2.putText(image, str(datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")), (900, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (125, 246, 55), 2,cv2.LINE_AA)
        cv2.putText(img=image, text=(str(fps)+' fps'), org=(700, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,color=(125, 246, 55), thickness=2)

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # frame = cv2.imencode('.jpg', curr_img)[1].tobytes() #see preprocessed image



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



if __name__ == '__main__':
    app.run(debug = True)