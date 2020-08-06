# 컨투어

# contours, hierarchy = cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
#image = 바이너리 이미지 , hierarchy = 검출된 컨투어 구조적 저장 offset 좌표구하기
#cv.CHAIN_APPOX_NONE , cv.CHAIN_APPOX_SIMPLE(꼭지점만 컨투어, 메모리 절약)
# mode = 

import cv2 as cv

img_color = cv.imread('test.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img_color, contours, 0, (0, 255, 0), 3) #인덱스 0 컨투어
cv.drawContours(img_color, contours, 1, (255, 0, 0), 3) #인덱스 1 컨투어

cv.imshow("result", img_color)
cv.waitKey(0)

#컨투어 그리기
#image = cv.drawContours( image, contours, contourldx, color[, thickness [, lineType[, hierarchy[, maxLevel[,offset]]]]])
#이미지 = 컨투어 그릴 이미지, 컨투어저장된 컨투어, 이미지그릴 컨투어 인덱스, 컨투어 그릴색상, 컨투어그릴 선굵기

#=================================================
import cv2 as cv

img_color = cv.imread('test.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for cnt in contours:
    for p in cnt:
        cv.circle(img_color, (p[0][0], p[0][1]), 10, (255,0,0), -1)
cv.imshow("result", img_color)
cv.waitKey(0)


#====================컨투어 속의 컨투어=======================#
import cv2 as cv

img_color = cv.imread('test.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# RETR_TREE     : 컨투어 내부의 다른 컨투어가 있을시 계층구조를 만들어준다.
# RETR_LIST     : 모든 컨투어가 같은 계층을 갖는다
# RETR_EXTERNAL : 가장 외곽의 컨투어만 리턴 자식 컨투어는 무시
# RETR_CCOMP    : 모든 컨투어를 2개의 레벨계층으로 구성 외곽 컨투어 , 내각 컨투어


for cnt in contours:
    cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)
print(test)

cv.imshow("result", img_color)
cv.waitKey(0)

#====================컨투어 특징 사용하기==================#

#유튜브 https://www.youtube.com/watch?v=j1OlFuFbRfE
#https://webnautes.tistory.com/1270