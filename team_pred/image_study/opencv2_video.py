# import cv2

# cap = cv2.VideoCapture(0) # 비디오 캡쳐 객체 사용 
#                         #2개의 카메를 쓰고싶으면 (1) 생성

# ret, img_color = cap.read() #카메라로부터 이미지1장가져옴
# cv2.imshow("Color", img_color) #캡쳐이미지 출력
# cv2.waitKey(0)

# cap.release()
# cv2.destroyAllWindows()

#############################한장이 아닌 동영상 처럼 보이게 캡쳐하기#######################################
# import cv2

# cap = cv2.VideoCapture(0) # 비디오 캡쳐 객체 사용 
#                         #2개의 카메를 쓰고싶으면 (1) 추가

# #동영상 저장

# fourcc = cv2.VideoWriter_fourcc(* 'XVID') #비디오 코덱 설정
# writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,480))
# # 파일명, 코덱이름, 프레임수, 저장할 영상 크기(캡쳐 이미지 크기와 일치)

# while(True):
#     ret, img_color = cap.read()

#     if ret == False:
#         continue

#     #그레이 이미지로 변환
#     img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

#     cv2.imshow("Color", img_color)
#     cv2.imshow("Gray", img_gray)

#     writer.write(img_color)# 카메라로부터 캡쳐되는 이미지 반복저장하여 동영상 만들기

#     if cv2.waitKey(1)&0xFF == 27:
#         break

# cap.release()
# writer.release()

# cv2.destroyAllWindows()

##################################동영상 파일 재생 코드#######################################
import cv2

filepath = './singing.AVI'
cap = cv2.VideoCapture(filepath) # 비디오 캡쳐 객체 사용 
                        #2개의 카메를 쓰고싶으면 (1) 추가
# 동영상 재생
fourcc = cv2.VideoWriter_fourcc(* 'XVID') #비디오 코덱 설정

import cv2, sys, re

xml = './image_study/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(xml)

while(True):
    ret, img_color = cap.read()

    if ret == False:
        continue

    #그레이 이미지로 변환
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color", img_color)
    cv2.imshow("Gray", img_gray)

    if cv2.waitKey(0)&0xFF == 27:
        break

cap.release()
writer.release()

cv2.destroyAllWindows()
