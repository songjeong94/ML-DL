import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0].shape)
print("y_train: ", y_train[0])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train[0].shape)
#plt.imshow(x_train[0], 'gray')
#plt.imshow(x_train[0])
#plt.show()

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2.정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

model = Sequential()
model.add(Conv2D(80, (2,2), # (2,2) = 픽셀을 2 by 2 씩 잘른다.
         input_shape=(28,28,1))) #(가로,세로,명암 1=흑백, 3=칼라)(행, 열 ,채널수) # batch_size, height, width, channels
model.add(Conv2D(50,(2,2)))
model.add(MaxPooling2D(pool_size=2))    #strides : 높이와 너비를 따라 컨벌루션의 보폭을 지정하는 정수 또는 튜플 / 2 개의 정수 목록입니다. 모든 공간 치수에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다. 모든 보폭 값! = 1을 지정하면 모든 dilation_rate값! = 1 을 지정할 수 없습니다 .
model.add(Conv2D(50,(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(80,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) # 2차원으로 변경
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 20,validation_split =0.18)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)
#과제 : dim 6 에서 5로 변경
#y_pred를 숫자로 바꿔라
