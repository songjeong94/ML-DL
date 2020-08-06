import cv2
import numpy as np

img = cv2.imread("./iu.jpg")

(height, width) = img.shape[:2]
center = (width // 2, height // 2)

cv2.imshow("image", img)

(Blue, Green, Red) = cv2.split(img)
#==========채널별로 그레이 스케일================
cv2.imshow("Red", Red)
cv2.imshow("Blue", Blue)
cv2.imshow("Green", Green)
cv2.waitKey(0)

#==============각 채널별 색깔입히기============================
# zeros = np.zeros(img.shape[:2], dtype = "uint8")
# cv2.imshow("Red", cv2.merge([zeros, zeros, Red]))
# cv2.imshow("Green", cv2.merge([zeros, Green, zeros]))
# cv2.imshow("Blue", cv2.merge([Blue, zeros, zeros]))
# cv2.waitKey(0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray filter", gray)
# cv2.waitKey(0)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("gray filter", hsv)
# cv2.waitKey(0)

# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow("gray filter", lab)
# cv2.waitKey(0)

BGR = cv2.merge([Blue, Green, Red])
cv2.imshow("BGR", BGR)
cv2.waitKey(0)

cv2.destroyAllWindows()