#keras40_lstm_split.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
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

x = np.reshape(x, (6, 4, 1 ))
x_predict = np.reshape(x_predict, (1, 4, 1 ))
#x = x.reshape(6, 4, 1)
print(x.shape)

model = Sequential()
model.add(LSTM(1, input_shape=(4, 1)))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 30)

model.compile(loss ='mse', optimizer = 'adam', metrics=['mse'] )
model.fit(x, y, epochs= 40, batch_size=1, verbose=1, callbacks=[early])

loss, mse = model.evaluate(x, y, batch_size = 1)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x_predict)
print("y예측값:",y_predict)