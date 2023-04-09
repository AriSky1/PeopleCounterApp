import glob
import cv2



d = r"C:\Users\ariai\Documents\DATA SCIENCE\PROJECTS\PeopleCounterApp\videos"
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