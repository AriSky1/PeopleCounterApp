
import cv2
cam = cv2.VideoCapture("shibuya_raw.mp4")
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