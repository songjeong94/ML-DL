from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
'''
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()
'''

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(y_train.shape)

# 데이터 전처리 2.정규화
x_train = x_train.reshape(50000,3072,).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000,3072,).astype('float32') / 255

input1 = Input(shape=(3072,))
dense1 = Dense(500)(input1)
drop0 = Dropout(0.2)(dense1)
dense2 = Dense(40)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(40)(drop1)
drop2 = Dropout(0.2)(dense3)
dense3 = Dense(40)(drop2)
drop2_2 = Dropout(0.2)(dense3)
dense4 = Dense(100)(drop2_2)
drop2_3 = Dropout(0.2)(dense4)

output1 = Dense(50)(drop2_3)
drop3 = Dropout(0.3)(output1)
output2 = Dense(50)(drop3)
drop4 = Dropout(0.3)(dense2)
output3 = Dense(10, activation='softmax')(drop4)

model = Model(inputs = input1, outputs = output3)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 20 ,batch_size=32)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)