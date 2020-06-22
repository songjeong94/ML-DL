# 과제2 
# sequential형으로 완성하시오.
#하단에 주석으로  acc와 loss 결과 명시하시오.
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Conv2D, Input 
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape) # 50000, 32,32 ,3
print(x_test.shape) # 10000, 32, 32 ,3
print(y_train.shape) # 50000,1
print(y_test.shape) # 10000,1

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(y_train.shape)


x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255 
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(28,28,1)))
model.add(Conv2D(30,(2,2)))
model.add(Conv2D(30,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(30,(2,2)))
model.add(Conv2D(30,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(30,(2,2)))
model.add(Conv2D(30,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 20)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)

#loss :  0.38083849975786016
#acc :  0.8712000250816345