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


print("x.shape: ", x.shape) #(13, 3)
print("y.shape: ", y.shape) #(4, )

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

#2. 모델 구성
input1 = Input(shape=(3, 1))
dense1 = LSTM(10, activation='relu', input_shape = (3, 1), return_sequences =True)(input1)
dense1_1 = LSTM(10, return_sequences =True)(dense1)
dense2 = Dense(5, activation='relu')(dense1_1) 
dense3 = Dense(1, activation='relu')(dense2) 


output1 = Dense(100)(dense2)
output2 = Dense(100)(output1)
output3 = Dense(1)(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()

'''
#3.실행
model.compile(optimizer='adam', loss = 'mse',)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor= 'loss', patience= 2, mode = 'auto')
model.fit(x, y, epochs=11, batch_size=1, validation_split=0.3, callbacks = [early_stopping])

print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
''' 
# 4*3(nm+n^2)

#840 = 4(10+1+10)*10