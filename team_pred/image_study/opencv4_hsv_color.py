# BGR은 파초빨 (0, 0, 255) 이면 빨간색
# RGB는 빨초파

# RGB88 은 파초빨에 각각 8bit씩사용
# import numpy as np
# import cv2

# color = [255, 0, 0] #파란색 hsv값구해보기
# pixel = np.uint8([[color]]) #cvt컬러함수를 사용하기위해 한필섹로 구성된 이미지로 변환

# hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV) #cvt컬러함수를 사용하여 hsv색공간으로 변환
# hsv = hsv[0][0] #hsv값을 출력하기 위해 픽셀값을 가져온다.

# print("bgr: ", color) # bgr:  [255, 0, 0]
# print("hsv:", hsv) # hsv: [120 255 255] 파란색을 위한 값은 120 +- 10

#===========================파란색만 검출================================
import cv2

img_color = cv2.imread('./image_study/messi_1.jpg') #이미지파일 컬러로 읽기
# height, width = img_color.shape[:2] #높이와 너비 가져오기

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) #hsv이미지로 변환

lower_blue = (120-10, 30, 30) #hsv의 바이너리 생성 범위 하한값
upper_blue = (120+10, 255, 255) #상한값
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) #바이너리 이미지얻기, 범위내의 픽셀은 흰색 나머지는 검은색

img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask) 

cv2.imshow('img_color', img_color)
cv2.imshow('img_mask', img_mask)
cv2.imshow('img_result', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()