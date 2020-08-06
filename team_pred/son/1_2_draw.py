import cv2
#rgb :빨파초
img = cv2.imread("./iu.jpg")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

cv2.imshow("image", img)
(b, g, r) = img[0, 0] #opencv내에서는 bgr순 출력은 rgb로 해야한다.
print("Pixel at (0, 0 - Red: {}, Green: {}, Blue: {}".format(r,g,b))

dot = img[50:100, 50:100] # read 이미지에서 50~100번 전까지 사각형만큼 짤른다
# cv2.imshow("Dot", dot)

# img[50:100, 50:100] = (0, 0, 255) #짤린이미지에 r칼라를 채워서 보여준다.

#======================사각형=============================
# cv2.rectangle(img, (150, 50), (200, 100), (0, 255, 0), 5) #  원하는 지점에 사각형 생성 , 5는 사각형 굵기

#=========================원=============================
# cv2.circle(img, (150,75), 25, (0,255,255), -1)#
            # 중심 275  반지름 25 색상  , 선 -1이면 전체가 채워진다.

#=========================선=============================
# cv2.line(img, (350, 100), (400, 100), (255, 0, 0), 5)

#=----------------------텍스트 넣기==========================
cv2.putText(img, 'creApple', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
          #  이미지, 출력 문자, 시작 위치, 폰트 지정, 폰트크기 , 칼라 ,폰트 굵기


# cv2.imshow("iu -dotted", img)
cv2.imshow("iu -draw", img)

cv2.waitKey(0)
cv2.destroyAllWindows()