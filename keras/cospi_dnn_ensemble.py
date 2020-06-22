import numpy as np
import pandas as pd

kospi200 = np.load('./data/kospi200.npy', allow_pickle=True)
samsung = np.load('./data/samsung.npy', allow_pickle=True)

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 =split_xy5(kospi200, 5, 1)
print(x2[0,:],"\n", y2[0])
print(x2.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=1, test_size = 0.3)

print(x1_train.shape)
print(x1_test.shape)
print(x2_train.shape)
print(x2_test.shape)

x1_train = x1_train.reshape(294,25)
x1_test = x1_test.reshape(127,25)
x2_train = x2_train.reshape(294,25)
x2_test = x2_test.reshape(127,25)

####데이터 전처리#####
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

print(x1_train_scaled[0, :])


from keras.models import Sequential, Model
from keras.layers import Dense, Input

#모델 구성
input1 = Input(shape=(25,))
dense1 = Dense(64)(input1)
dense2 = Dense(32)(dense1)
dense3 = Dense(32)(dense2)
output1 = Dense(32)(dense3)

input2 = Input(shape=(25,))
dense1 = Dense(64)(input1)
dense2 = Dense(32)(dense1)
dense3 = Dense(32)(dense2)
output2 = Dense(32)(dense3)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs=[input1,input2],
            outputs = output3)



from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, verbose=1,
            batch_size=1, epochs=100, callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test_scaled,x2_test_scaled], y1_test, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)

y_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(5):
    print('종가 : ', y1_test[i],'/ 예측가 :', y_pred[i])