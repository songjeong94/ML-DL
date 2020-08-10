import numpy as np 
import cv2

#0
image = cv2.imread("image_study/test.png", cv2.IMREAD_COLOR)
cv2.imshow("result", image)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(image, (5, 5), 0)

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
cv2.imshow("result", gray)
cv2.waitKey(0)
edge = cv2.Canny(gray, 50, 150)
cv2.imshow("result", edge)
cv2.waitKey(0)

# 1
edge = cv2.bitwise_not(edge)
cv2.imshow("result", edge)
cv2.waitKey(0)

contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1)
cv2.imshow("result", edge)
cv2.waitKey(0)


# 2
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge)

for i in range(nlabels):

    if i < 2:
        continue

    area = stats[i, cv2.CC_STAT_AREA]
    center_x = int(centroids[i, 0])
    center_y = int(centroids[i, 1])
    left = stats[i, cv2.CC_STAT_LEFT]
    top = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]

    if area > 50:
        cv2.rectangle(image, (left, top), (left + width, top + height),
                        (0, 0, 255), 1)
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), 1)
        cv2.putText(image, str(i), (left + 20, top+ 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2);

cv2.imshow("result", image)
cv2.waitKey(0)


