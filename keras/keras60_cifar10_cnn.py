from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Conv2D, Input 
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train[0])
#print('y_train[0] : ', y_train[0])

#print(x_train.shape) # 50000, 32,32 ,3
#print(x_test.shape) # 10000, 32, 32 ,3
#print(y_train.shape) # 50000,1
#print(y_test.shape) # 10000,1

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(y_train.shape)


x_train = x_train.reshape(50000,32,32,3 ).astype('float32') / 255 
x_test = x_test.reshape(10000,32,32,3 ).astype('float32') / 255

input1 = Input(shape=(32,32,3))
dense1 = Conv2D(30,(2,2),input_shape=(32,32,3))(input1)
dense2 = Conv2D(30,(2,2))(dense1)
max1 =(MaxPooling2D(pool_size=2))(dense2)
dense3 = Conv2D(30,(2,2))(max1)
dense4 = Conv2D(30,(2,2))(dense3)
drop3 = Dropout(0.1)(dense4)
dense5 = Conv2D(30,(2,2))(drop3)
max2 =(MaxPooling2D(pool_size=2))(dense5)

fl1 = Flatten()(max2)
output1 = Dense(100)(fl1)
drop1 = Dropout(0.2)(output1) 
output2 = Dense(100)(drop1)
output3 = Dense(100)(output2)
drop2 = Dropout(0.2)(output3) 
output4 = Dense(100)(drop2)
output5 = Dense(10, activation='softmax')(output4)

model = Model(inputs =input1 , outputs = output5)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 20)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)
#과제 : dim 6 에서 5로 변경
#y_pred를 숫자로 바꿔라
