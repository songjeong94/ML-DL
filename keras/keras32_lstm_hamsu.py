from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4],[3,4,5],[4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]]) #13,3
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13, )

x_predict = array([50, 60, 70])
x_predict = x_predict.reshape(1,3,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.8)


print("x.shape: ", x.shape) #(13, 3)
print("y.shape: ", y.shape) #(4, )

#x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

#2. 모델 구성

input1 = Input(shape=(3, 1))
dense1 = LSTM(10, activation='relu', input_shape = (3, 1))(input1)
dense2 = Dense(10, activation='relu')(dense1) 
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3) 

output1 = Dense(1)(dense4)
output2 = Dense(10)(output1)
output3 = Dense(1)(output2)

model = Model(inputs = input1, outputs = output3)
model.summary()
'''
#3.실행
model.compile(optimizer='adam', loss = 'mse',)
model.fit(x, y, epochs=93, batch_size=1, validation_split=0.3)


print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
'''