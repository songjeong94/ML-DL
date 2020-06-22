from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()
#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
'''
# 데이터 전처리 2.정규화
x_train = x_train.reshape(50000,1024,3).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000,1024,3).astype('float32') / 255

input1 = Input(shape=(1024,3))
dense1 = LSTM(10, input_shape=(1024, 3))(input1)
dense2 = Dense(100)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(100)(drop1)
dense4 = Dense(100)(dense3)
drop2 = Dropout(0.2)(dense4)
dense5 = Dense(100)(drop2)
dense6 = Dense(100)(dense5)
drop3 = Dropout(0.2)(dense6)

output1 = Dense(100)(drop3)
output2 = Dense(50)(output1)
output3 = Dense(10, activation='softmax')(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 2 ,batch_size=32)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)'''