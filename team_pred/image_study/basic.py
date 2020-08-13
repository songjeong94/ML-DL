import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('33messi_1.jpg',1)
# cv2.imread_color: 1
# cv2.imread_grayscale : 0
# cv2.imread_unchanged : -1
#투명도인  alpha 채널을 포함한다.

cv2.imshow('image', img)
#첫번째 인수는 window식별자, 두번째인수는 이미지변수
k = cv2.waitKey(0)
# 인자값이 없거나 0일 경우 시간 제한없이 사용자 키 입력을 기다린다.

if k == ord('s'):
    cv2.imwrite('d:/z.jpg', img)
#입력된 키값이 소문자 s 일 경우 img 객체의 저장된 이미지를 또 다른 파일로 
# 저장하라는  cv2.imwrite 함수입니다.,
cv2.destroyAllWindows()
# 프로그램의 종료를 위해 표시된 모든 window를 종료

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()