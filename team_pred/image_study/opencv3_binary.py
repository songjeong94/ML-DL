# 이미지의 이진화 흰색과 검은색으로 된 바이너리로 변환 (이미지 전처리로 자주사용)
# 배경과 오프젝트를 분리 가능하다.

# retval, dst = cv2.threshold(src, thresh, maxval, type[, dst])

# dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])

#  컬러이미지 (하나의 픽셀은 3개의 채널로 구성되면 각각0~255까지의 값을 가질수 있다.)
# 그레이 이미지 (검은색부터 검은색까지 1개의 채널로 0~255의 값을 가진다.)
# 그레이 이미지를 이진화(임계값을 기준으로 흰색과 검은색 두개로 구별)
# 채널 = 빨강,초록,파랑 의 혼합으로 만든다. 특정채널의 값이 높지않고 일정하면 혼합된값으로 표현
#특정 채널이 높으면 그레이 이미지에서 밝게 표현

# import cv2

# img_color = cv2.imread('red_ball.jpg', cv2.IMREAD_COLOR)

# cv2.imshow('Color', img_color)
# cv2.waitKey(0)

# img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Gray', img_gray)
# cv2.waitKey(0)

# #그레이 이미지 이진화
# ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# # (이진화 대상이미지(그레이로 가능), threshold(이 값을 기준으로 검은색 또는 흰색이 된다.) )
# # 임계값은 조절가능하며 0이면 하얀색 높으면 검은색에 가깝다.
# cv2.imshow("Binary", img_binary)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

######################################### 트랙바 수정하기##############################################

import cv2

def nothing(x):
    pass

cv2.namedWindow('Binary')
cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing) #최소 최대값
cv2.setTrackbarPos('threshold', 'Binary', 127) #초기값

img_color = cv2.imread('./image_study/apple.jpg', cv2.IMREAD_COLOR)

# cv2.imshow('Color', img_color)
# cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Gray', img_gray)
# cv2.waitKey(0)

while(True): # 트랙바 이동의 결과를 바로 확인할수 있게 만들자
    low =cv2.getTrackbarPos('threshold', 'Binary') # 현자의 트랙바 값
    ret,img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY)

    cv2.imshow("Binary", img_binary)

    if cv2.waitKey(1)&0xFF == 27:
        break #esc키를 누르면 루프에서 나올수있도록 조건문

cv2.destroyAllWindows()

#=================================반전 마스크 얻기==============================######################################### 트랙바 수정하기##############################################

import cv2

def nothing(x):
    pass

cv2.namedWindow('Binary')
cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing) #최소 최대값
cv2.setTrackbarPos('threshold', 'Binary', 127) #초기값

img_color = cv2.imread('./image_study/apple.jpg', cv2.IMREAD_COLOR)

# cv2.imshow('Color', img_color)
# cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Gray', img_gray)
# cv2.waitKey(0)

while(True): # 트랙바 이동의 결과를 바로 확인할수 있게 만들자
    low =cv2.getTrackbarPos('threshold', 'Binary') # 현자의 트랙바 값
    ret,img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("Binary", img_binary)

    img_result = cv2.bitwise_and(img_color, img_color, mask = img_binary) #원본이미지와 바이너리 이미지 and연산
    cv2.imshow('Result', img_result) # 결과 확인 (마스크이미지에서 흰색이미지만 원본에 남는다. 오브젝트는 흰색으로 만들자,cv2.THRESH_BINARY_INV반전이미지 만들기)

    if cv2.waitKey(1)&0xFF == 27:
        break #esc키를 누르면 루프에서 나올수있도록 조건문

cv2.destroyAllWindows()