import cv2
import numpy as np

face_cascade_name = './son/son_file/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './son/son_file/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

file_name = 'son/face_01.mp4'

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 4)
        faceROI = frame_gray[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
        cv2.imshow("capture - face", frame)

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

    
if not eyes_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)


#2. read the video stream
cap = cv2.VideoCapture(file_name)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('==(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break