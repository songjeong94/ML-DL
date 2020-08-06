import cv2
import numpy as np

model_name = 'son_file/model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'son_file/model/deploy.prototxt.txt'
min_confidence = 0.3
file_name = "son_file/image/song_1.jpg"

def detectAndDisplay(frame):
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name) #opencv 의 dnn 사용 카페 모델을 사용하여 모델 객체 생성
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # 리사이즈 300,300 이후 노말라이즈 하는 값을 준후 blob값을 이미지로 보내주게 된다.
    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            print(confidence, startX, startY, endX, endY)

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10 #text의 위치 바운딩 박스 위에 올린다.
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("face detection by dnn", frame)

img = cv2.imread(file_name)

(height, width) = img.shape[:2]

cv2.imshow("Original Image", img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
