import cv2
#rgb :빨파초
img = cv2.imread("./iu.jpg")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

cv2.imshow("image", img)

cv2.waitKey(0)
cv2.imwrite("./son/iu1.jpg", img)
cv2.destroyAllWindows()
