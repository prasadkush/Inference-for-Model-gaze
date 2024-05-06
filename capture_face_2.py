# This code captures eyes (in a predefined rectangular area) and the face

import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

video_capture = cv2.VideoCapture(0)
anterior = 0

x = 200
y = 100
w = 200
h = 200

xe = 220
ye = 125
we = 80
he = 50

while True:
    ret, frame = video_capture.read()
    roi_img = frame[y:y+h, x:x+w, :]    
    roi_eyes = frame[ye:ye+he,xe:xe+we,:]
    cv2.rectangle(roi_img,(xe - x,ye -y),(xe+we -x,ye+he - y),(255,0,0),2)
    cv2.imshow('Video1', roi_img)
    cv2.imshow('Eyes1', roi_eyes)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()

i = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #cv2.imwrite('1.jpg', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    eyes = ()
    roi_img = 0
    roi_gray = 0
    
    roi_img = frame[y:y+h, x:x+w, :]
    roi_gray = gray[y:y+h,x:x+w]

    roi_eyes = frame[ye:ye+he,xe:xe+we,:]


    cv2.imwrite('data/trial 3/just_eyes' + str(i) + '.jpg', roi_eyes)
    cv2.imwrite('data/trial 3/img' + str(i) + '.jpg', roi_img)
    print('i: ', i)
    i = i + 1
 
    cv2.rectangle(roi_img,(xe - 2,ye -2),(xe+we + 2,ye+he + 2),(255,0,0),2)


    # Display the resulting frame
    cv2.imshow('Video', roi_img)
    cv2.imshow('Eyes', roi_eyes)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()