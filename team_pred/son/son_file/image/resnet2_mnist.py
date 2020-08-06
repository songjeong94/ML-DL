from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import cv2

from keras import layers, models
from keras.applications import ResNet152V2, VGG16, M
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import os
from glob import glob 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
import numpy as np
import dlib
import cv2
import face_recognition


RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(42, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(17, 27))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'son_file/model/shape_predictor_68_face_landmarks.dat' 
dataset_paths = ['son_file/image/song-front/', 'son_file/image/kang-front/']
output_paths = ['son_file/image/song-test/','son_file/image/kang-test/']
MARGIN_RATIO = 1 #랜드마크 사이즈
OUTPUT_SIZE = (400, 400)
number_images = 10
image_type = '.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)


def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    width = width + int(70)
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top() - 50
    endY = rect.bottom() + 50
    return (startX, endX, startY, endY) 

for (i, dataset_path) in enumerate(dataset_paths):
    output_path = output_paths[i]
    
    for idx in range(number_images):
        input_file = dataset_path + str(idx+1) + image_type

        # get RGB image from BGR, OpenCV format
        image = cv2.imread(input_file)
        image_origin = image.copy()

        (image_height, image_width) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = getFaceDimension(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
            show_parts = points[EYES]
            
            right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
            left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")
            print(right_eye_center, left_eye_center)
            
            
            cv2.circle(image, (right_eye_center[0,0], right_eye_center[0,1]), 5, (0, 0, 255), -1)
            cv2.circle(image, (left_eye_center[0,0], left_eye_center[0,1]), 5, (0, 0, 255), -1)
            
            cv2.circle(image, (left_eye_center[0,0], right_eye_center[0,1]), 5, (0, 255, 0), -1)
            cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
            (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 2)
            cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
            (left_eye_center[0,0], right_eye_center[0,1]), (0, 255, 0), 1)
            cv2.line(image, (left_eye_center[0,0], right_eye_center[0,1]),
            (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 1)
            
            eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
            eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
            degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180
            
            eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
            aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
            scale = aligned_eye_distance / eye_distance # 사진을 돌리는 비율
            
            eyes_center = ((left_eye_center[0,0] + right_eye_center[0,0]) // 2,
            (left_eye_center[0,1] + right_eye_center[0,1]) // 2)
            cv2.circle(image, eyes_center, 5, (255, 0, 0), -1)
            
            metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale) #센터지점과 각도와 스케일을 함수를 사용하여 돌린다.
            cv2.putText(image, "{:.5f}".format(degree), (right_eye_center[0,0], right_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
            flags=cv2.INTER_CUBIC) # 이미지 회전
            
            cv2.imshow("warpAffine", warped)
            (startX, endX, startY, endY) = getCropDimension(rect, eyes_center) #사진을 자르는 함수 getCrop
            croped = warped[startY:endY, startX:endX]
            output = cv2.resize(croped, OUTPUT_SIZE)
            cv2.imshow("output", output)
            
            for (i, point) in enumerate(show_parts): #눈의 점을 찍어주는 부분 앞에들어가도된다.
                x = point[0,0]
                y = point[0,1]
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
                
                gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                
            for(i, rect) in enumerate(rects):
                points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
                show_parts = points[ALL]
                print(show_parts)
                
            for (i, point) in enumerate(show_parts):
                x = point[0, 0]
                y = point[0, 1] #flatten과 같다 
                cv2.circle(output, (x, y), 1, (0, 255, 255), -1)
                cv2.putText(output, "{}".format(i + 1), (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
            cv2.imshow("output:", output)
            output_file = output_path + str(idx+1) + image_type
            cv2.imshow(output_file, output)
            cv2.imwrite(output_file, output)

dataset_paths = ['son_file/image/song-test/', 'son_file/image/kang-test/']
names = ['Song', 'Kang']

for (i, dataset_path) in enumerate(dataset_paths):
    name = names[i]
    print("names:", name)
    #파일음 이름 선정
    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type

print(names)
for label in labels:
    for i in label:
        i = i.reshape(-1,400,400,3)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(song, kang, test_size = 0.2, random_state=1, shuffle=True)

input_tensor = Input(shape=(400, 400, 3), dtype='float32', name='input')
image = cv2.imread(file_name)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pre_trained_res = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
pre_trained_res.trainable = False
pre_trained_res.summary()
    
additional_model = models.Sequential()
additional_model.add(pre_trained_res)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(2048, activation='relu'))
additional_model.add(layers.Dense(1024, activation='relu'))
additional_model.add(layers.Dense(1, activation='sigmoid'))

additional_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = additional_model.fit(x_train, y_train, 
                    batch_size=1, 
                    epochs=1, 
                    validation_data=(x_test, y_test))


# print("정확도 : %.4f" % (additional_model.evaluate(x_test, y_test)[1]))

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

# model_json = additional_model.to_json()
# with open("model.json", "w") as json_file : 
#     json_file.write(model_json)

# additional_model.save_weights("model_w.h5")
# print("Saved model to disk")