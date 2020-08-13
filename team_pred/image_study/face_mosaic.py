import cv2, sys, re

xml = './image_study/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(xml)


# 확인한 부분에 모자이크 걸기
# print(face_list)
color = (0, 0, 255)
for ( x, y, w, h) in face_list:
    # 얼굴 부분 자르기
    face_img = image[x:y+h, x:x+w]
    # 자른 이미지를 지정한 배율로 확대/축소하기
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    # 확대/축소한 그림을 원래 크기로 돌리기
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
    # 원래 이미지에 붙이기
    image[y:y+h, x:x+h] = face_img

# 렌더링 결과를 파일에 출력
cv2.imwrite(output_file, image)
