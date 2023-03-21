#pip install pafy
#sudo pip install --upgrade youtube_dl
import yt_dlp as youtube_dl

import cv2, pafy

url= "https://www.youtube.com/watch?v=lMOtsTGef38"
video = pafy.new(url)
best= video.getbest(preftype="mp4")
#documentation: https://pypi.org/project/pafy/

capture = cv2.VideoCapture(best.url)
check, frame = capture.read()
print (check, frame)

cv2.imshow('frame',frame)
cv2.waitKey(1000000)

capture.release()
cv2.destroyAllWindows()