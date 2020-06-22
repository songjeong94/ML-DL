#keras40_lstm_split.py

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping

#1. 데이터
a = np.array(range(1, 11))
size = 5                   # time_steps = 4

x_predict = np.array([13,14,15,16])
#LSTM  모델을 완성하시오.

def split_x(seq, size):
    aaa = [] # 임심 메모리 리스트
    for i in range(len(seq) - size + 1):#seq - size +1  = 값 행종료값
        subset = seq[i: (i+size)] #열 지정
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)  # (6,5)
print(dataset)
print(dataset.shape)
print(type(dataset))

x = dataset[:, 0:4] #[:]모든행 ,  [0:4] 0~3까지의 인데스 
y = dataset[:, 4]   # 4 인덱스 데이터를 가져온다.

x = np.reshape(x, (6, 4, ))
x_predict = np.reshape(x_predict, (1, 4, ))
#x = x.reshape(6, 4, 1)
print(x.shape)

input1 = Input(shape=(4,))
dense1 = Dense(100)(input1)
dense2 = Dense(100)(dense1)
dense3 = Dense(100)(dense2)
dense4 = Dense(100)(dense3)
dense5 = Dense(100)(dense4)

output1 = Dense(100)(dense5)
output2 = Dense(100)(output1)
output3 = Dense(100)(output2)
output4 = Dense(1)(output3)

model = Model(inputs = input1, outputs = output4)

early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 20)

model.compile(loss ='mse', optimizer = 'adam', metrics=['mse'] )
model.fit(x, y, epochs= 30, batch_size=1, verbose=1, callbacks=[early])

loss, mse = model.evaluate(x, y, batch_size = 1)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x_predict)
print("y예측값:",y_predict)