import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

#cascPath = "haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascPath)
#log.basicConfig(filename='webcam.log',level=log.INFO)


cascPath = 'haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyeCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
anterior = 0

x = 200
y = 100
w = 200
h = 200


while True:
    ret, frame = video_capture.read()
    roi_img = frame[y:y+h, x:x+w, :]    
    cv2.imshow('Video1', roi_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()

i = 0

while True:
    print('here')
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cv2.imwrite('1.jpg', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    eyes = ()
    roi_img = 0
    roi_gray = 0
    
    roi_img = frame[y:y+h, x:x+w, :]
    roi_gray = gray[y:y+h,x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 5) 
    
    #cv2.imwrite('1_modified.jpg', roi_img)

    print('eyes: ', eyes)
    for (xe,ye,we,he) in eyes:
        #cv2.rectangle(roi_img,(xe,ye),(xe+we,ye+he),(255,0,0),2)
        roi_eyes = roi_img[ye:ye+he,xe:xe+we,:]
        cv2.imwrite('data/trial 1/just_eyes' + str(i) + '.jpg', roi_eyes)
        cv2.imwrite('data/trial 1/img' + str(i) + '.jpg', roi_img)
        i = i+1


    # Display the resulting frame
    cv2.imshow('Video', roi_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()