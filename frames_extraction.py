import glob
import cv2
import os
import time
import re

absolute_path = os.path.dirname(__file__)
vid_path = "videos"
frames_selected_path = 'frames_selected'
vid_path = os.path.join(absolute_path, vid_path)
frames_path = 'frames'
frames_path = os.path.join(absolute_path, frames_path)
frames_selected_path = os.path.join(absolute_path, frames_selected_path)

fl = glob.glob(vid_path + "/*.mp4")
counter = 0
cam = cv2.VideoCapture(fl[counter])
currentframe = 0




while(True):
    ret,frame = cam.read()
    if ret:
        name = 'frames\Frame(' + str(currentframe) + ').jpg'
        if (currentframe % 10 == 0):

            cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()



