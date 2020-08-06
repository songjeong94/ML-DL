import cv2
import numpy as np
#rgb :빨파초
img = cv2.imread("son/image/son/1.jpg")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2] # img.shape 에서 0,1 값을가져온다
center = (width // 2, height // 2) # 높이와 넓이의 중간값 

cv2.imshow("image", img)

#======================이미지 이동=========================
# move = np.float32([[1, 0, -100], [0, 1, -100]]) #[1, 0, 100] =위 아래로 100만큼 다운 , 좌우로 100 -100이여야 올라간다
# moved = cv2.warpAffine(img, move, (width, height)) #width, height  만큼 움직인다.
# cv2.imshow("imag", moved)

#===================이미지 회전=====================
# move = cv2.getRotationMatrix2D(center, -90, 1.0) # 90도만큼돈다, 스케일 얼마나 키울건지 1 = 그대로 사용
# rotated = cv2.warpAffine(img, move, (width, height))
# cv2.imshow("imga", rotated)


#=====================사이즈 줄이기==========================
# ratio = 200.0 / width #가로를 200픽셀로 줄인다.
# dimension = (200, int(height * ratio))

# resized = cv2.resize(img, dimension, interpolation= cv2.INTER_AREA) #인터폴레이션 = 인터 에이리어(영역보호)
# cv2.imshow("Resized", resized) #

#=====================상하좌우대칭===================================
flipped = cv2.flip(img, -1)
cv2.imshow("Flipped Horizontal 1, Vertical 0, both -1", flipped)
#             좌우대칭           , 위아래 대칭  , 둘다, 

cv2.waitKey(0)
cv2.destroyAllWindows()