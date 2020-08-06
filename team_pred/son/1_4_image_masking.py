import cv2
import numpy as np

img = cv2.imread("iu.jpg")

(height, width) = img.shape[:2]
center = (width // 2, height // 2)

cv2.imshow("image", img)

mask = np.zeros(img.shape[:2], dtype = "uint8")
               # 높이와 넓이에 대해 0칼라 검정을 넣는다.

cv2.circle(mask, center, 100, (255, 255, 255), -1)

    
cv2.imshow("mask", mask)

masked = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow("iu", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()