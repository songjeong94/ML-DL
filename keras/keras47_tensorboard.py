#keras45_lstm_split.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, History, TensorBoard

#1. 데이터
a = np.array(range(1, 101))
size = 5                   # time_steps = 4
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

x = dataset[:90, 0:4] #[:]모든행 ,  [0:4] 0~3까지의 인데스 
y = dataset[:90, -1:]   # 4 인덱스 데이터를 가져온다.
x_predict = dataset[90:, 0:4]

#print(y)

x = np.reshape(x, (90, 4, 1))
x_predict = np.reshape(x_predict, (6, 4, 1))
#x = x.reshape(6, 4, 1)
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8)

#2. 모델
model = Sequential()
model.add(LSTM(10, input_shape=(4, 1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#model.summary()


early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
#3.  훈련

tb_hist = TensorBoard(log_dir='graph', histogram_freq = 0, write_graph = True, write_images=True)


model.compile(loss ='mse', optimizer = 'adam', metrics=['mse'])
hist = model.fit(x_train, y_train, epochs= 30, batch_size=1, verbose=1, callbacks=[early, tb_hist], validation_split=0.1)

print(hist)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('loss & acc ')
#plt.ylabel('loss, acc')
plt.xlabel('epoch')
#plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
#plt.show()

loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x_predict)
print("y예측값:",y_predict)

