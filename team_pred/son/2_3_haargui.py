import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

face_cascade_name = './son_file/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './son_file/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
file_name = './sonfile/marathon_01.jpg'
title_name = 'Haar cascade object dection'
frame_width = 500

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./", title="Select file", filetypes = (("jpeg files","*.jpg"), ("all files", "*.*")))
    print("File name :", file_name)
    read_image = cv2.imread(file_name)
    (height, width) = read_image.shape[:2]
    frameSize = int(sizeSpin.get())
    ratio = frameSize / width
    dimension = (frameSize, int(height * ratio))
    read_image = cv2.resize(read_image, dimension, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=imgae)
    detectAndDisplay(read_image)

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
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image(imgtk)


#main
main = TK()
main.title(title_name)
main.geometry()


#GUI에 의한 이미지 리사이즈
read_img = cv2.imread("./son_file/image/marathon_01.jpg")
(height, width) = img.shape[:2]
ratio = frame_width / width
demension = (frame_width, int(height * ratio))
read_image = (cv2.resize(read_image, dimension, interpolation= cv2.INTER_AREA)) 

image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB) #제대로 된 색표현
image = Image.fromarray(image) #이미지형식을 fromarray형식으로
imgtk = ImageTk.PhotoImage(image=image) #tk이미지에 보여주기위해

face_cascade_name = './son_file/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './son_file/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('___1 error')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('___1 error')
    exit(0)
label = Label(main, text = title_name)
label.config(font =("Courier", 18))
label.grid(row=0, column=0, columnspan=4)
sizeLabel = Label(main, text = 'Frame Width : ')
sizeLabel.grid(row=1, column=0)
sizeVal = Inter(value = frame_width)
sizeSpin = Spinbox(main, textvariable = sizeVal, from_ =0, to = 2000, increment =100, justifi= RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2,columnspan=2)
detection=Label(main, image=imgtk)
detection.grid(row=2, columm=0, columnspan=4)
detectAndDisplay(read_image)


cv2.waitKey(0)
cv2.destroyAllWindows()


