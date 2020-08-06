import cv2
import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(42, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(17, 27))
ALL = list(range(0, 68))

predictor_file = 'son_file/model/shape_predictor_68_face_landmarks.dat'
image_file = 'son_file/image/marathon_03.jpg'

detector = dlib.get_frontal_face_detector() #정면 사진을 디덱트하겠다.
predictor = dlib.shape_predictor(predictor_file) #68개의 점

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1) #gray 처리 된 이미지를 몇번?
print("Number of faces detected: {} ".format(len(rects)))
print(rects)

for(i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL]
    print(show_parts)

    for (i, point) in enumerate(show_parts):
        x = point[0, 0]
        y = point[0, 1] #flatten과 같다 
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)
