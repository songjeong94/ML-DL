#앙상블 모델로 변경

from numpy import array
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x1= array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]]) #13,3
x2 = array([[10,20,30], [20,30,40],[30,40,50],[40,50,60],
            [50,60,70], [60,70,80], [70,80,90], [80,90,100], [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]]) #13,3

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13, )


x1_predict = array([55, 65 ,75])
x1_predict = x1_predict.reshape(1,3,1)

x2_predict = array([65, 75, 85])
x2_predict = x2_predict.reshape(1, 3, 1)
         

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)


print("x1.shape: ", x1.shape) #(13, 3)
print("x2.shape: ", x2.shape) #(13, 3)
print("y.shape: ", y.shape) #(4, )

#x = x.reshape(13, 3, 1)

#2. 모델 구성
input1 = Input(shape=(3,1))
dense1 = LSTM(18, activation='relu', input_shape=(3, 1))(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)

input2 = Input(shape=(3,1))
dense2_1 = LSTM(18, activation='relu', input_shape=(3, 1))(input2)
dense2_2 = Dense(100, activation='relu')(dense2_1)
dense2_3 = Dense(100, activation='relu')(dense2_2)
dense2_4 = Dense(100, activation='relu')(dense2_3)

from keras.layers.merge import concatenate
merge1 = concatenate([dense4, dense2_4])

middle1 = Dense(100)(merge1)
middle1 = Dense(100)(middle1)
middle1 = Dense(100)(middle1)
middle1 = Dense(100)(middle1)

output1 = Dense(1)(middle1)
output2 = Dense(100)(output1)
output3 = Dense(100)(output2)
output4 = Dense(1)(output3)

model = Model(inputs = [input1, input2], outputs = output1)

model.summary()
'''
#3.실행
model.compile(optimizer='adam', loss = 'mse',metrics =['mse'])
model.fit([x1, x2], y, epochs=93, batch_size=1)

loss, mse = model.evaluate([x1, x2],y, batch_size=1)
print("loss: " ,loss)
print("mse: ", mse)

y_predict = model.predict([x1_predict, x2_predict])
print(y_predict)
y_predict = y_predict.reshape(1,3,1)

'''