import cv2
import numpy as np

model_name = 'son_file/model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'son_file/model/deploy.prototxt.txt'
min_confidence = 0.7 #최소 신뢰도
file_name = "son_file/image/marathon_01.jpg"

def detectAndDisplay(frame):
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
    #caffe모델을 불러오기

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    #blob형태의 데이터를 불러온다
    #300x300의 크기를 사용하기위해 리사이즈
    #scalefactor: 이미지의 크기비율을 지정 1.0 그대로 사용
    #size: cnn에서 사용할 이미지 크기 300x300
    #mean: rgb색상 채널별로 지정해주는 경험치값 rgb값의 일부를 제외해서 dnn이 분석하기 쉽게 단순화해준다.

    model.setInput(blob) #caffe모델이 blob이미지를 처리하고 detections 에 결과를 저장
    detections = model.forward()

    print(detections[0, 0, 1])
    print(detections.shape[2])

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, -2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int") #좌표값
            print(i, confidence,detections[0,0,i,3], startX, startY, endX, endY)

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10 #사진박스 지정 사진이 맨위에 있을때 박스를 안쪽으로 굵기 지정 
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Face Detection by dnn", frame)

img = cv2.imread(file_name)

(height, width) = img.shape[:2]

cv2.imshow("Original Image", img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()