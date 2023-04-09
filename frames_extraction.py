import glob
import cv2
import os
import time
import re

absolute_path = os.path.dirname(__file__)
relative_path = "videos"
d = os.path.join(absolute_path, relative_path)

df = r"C:\Users\ariai\Documents\DATA SCIENCE\PROJECTS\PeopleCounterApp\frames"
fl = glob.glob(d + "/*.mp4")
counter = 0
cam = cv2.VideoCapture(fl[counter])
currentframe = 0




while(True):
    ret,frame = cam.read()
    if ret:
        name = 'frames\Frame(' + str(currentframe) + ').jpg'

        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()


images=glob.glob(df+"/*[0]*")
print(images)