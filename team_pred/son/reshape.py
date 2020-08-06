import cv2
#rgb :빨파초

img = cv2.imread("son_file/image/song_test.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.5, fy=0.5)

cv2.waitKey(0)
cv2.imwrite("son_file/image/song_1.jpg", img)
cv2.destroyAllWindows()

# import cv2

# src = cv2.imread("Image/champagne.jpg", cv2.IMREAD_COLOR)

# dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
# dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.imshow("dst2", dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
