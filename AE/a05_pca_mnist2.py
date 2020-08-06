# keras56_mnist_DNN.py 떙겨라
# input_dim = 154 로 모델을 만드시오.

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, LSTM
from keras.models import Sequential , Model
from sklearn.decomposition import PCA
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 2.정규화
x_train = x_train.reshape(60000,784,).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 784,).astype('float32') / 255

X = np.append(x_train, x_test, axis=0)
Y = np.append(y_train, y_test, axis=0)

pca = PCA(n_components=154)
pca.fit(X)
X = pca.fit_transform(X)
pca_std = np.std(X)
print(X.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, shuffle = True, train_size=0.8)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(10,input_shape =(154, )))
model.add(Dense(100))
model.add(Dropout(0.2)) 
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 5,batch_size=16)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)