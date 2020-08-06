# 엣지 검출 알고리즘
# 좋은 엣지란? 1.낮은 에러율 2. 정확한 에지 위치 3.응답 최소화

# img_canny = cv2.Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)



# 첫번째 아규먼트 image는 입력 이미지입니다.

# 두번째, 세번째 아규먼트 threshold1, threshold2는 최소 스레숄드와 최대 스레숄드입니다. 

# 네번째 아규먼트 edges에 Canny 결과를 저장할 변수를 적어줍니다.  파이썬에선 Canny 함수 리턴으로 받을 수 있기 때문에 필요없는 항목입니다. 

# 다섯번째 아규먼트 apertureSize는 이미지 그레디언트를 구할때 사용하는 소벨 커널 크기입니다. 디폴트는 3입니다. 

# 여섯번째 아규먼트 L2gradient가 True이면 그레디언트 크기를 계산할 때 sqrt{(dI/dx)^2 + (dI/dy)^2}를 사용합니다.

#  False라면 근사값인 |dI/dx|+|dI/dy|를 사용합니다.  디폴트값은 False입니다. 

# import cv2

# img_gray = cv2.imread('./image_study/newyork.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Original", img_gray)

# img_canny = cv2.Canny(img_gray, 50, 150)
# cv2.imshow("Canny Edge", img_canny)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#============== 트랙바 사용예제==========================
import cv2

def nothing():
    pass

img_gray = cv2.imread('./image_study/newyork.jpg', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Canny Edge")
cv2.createTrackbar('low threshold', 'Canny Edge', 0, 1000, nothing)
cv2.createTrackbar('high threshold', 'Canny Edge', 0, 1000, nothing)

cv2.setTrackbarPos('low threshold', 'Canny Edge', 50)
cv2.setTrackbarPos('high threshold', 'Canny Edge', 150)

cv2.imshow("Original", img_gray)

while True:

    low = cv2.getTrackbarPos('low threshold', 'Canny Edge')
    high = cv2.getTrackbarPos('high threshold', 'Canny Edge')

    img_canny = cv2.Canny(img_gray, low, high)
    cv2.imshow("Canny Edge", img_canny)

    if cv2.waitKey(1)&0xFF == 27:
        break

cv2.destroyAllWindows()
