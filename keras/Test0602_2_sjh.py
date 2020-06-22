import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.models import load_model


samsung = np.load('./data/samsung.npy', allow_pickle=True)
hite = np.load('./data/hite.npy', allow_pickle=True)

# print(samsung)
# print(hite)
print(samsung.shape)#508,1
print(hite.shape)   #508,5

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 0:3]
        x.append(tmp_x)

    return np.array(x)
x1 = split_xy5(hite, 1, 1)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 3:5]
        x.append(tmp_x)
    return np.array(x)

x2 = split_xy5(hite, 1, 1)
y = samsung

print(x1)......................................
print(x2)

x1 = np.reshape(x1,
    (x1.shape[0], x1.shape[1] * x1.shape[2]))
x2 = np.reshape(x2,
    (x2.shape[0], x2.shape[1] * x2.shape[2]))

print(x1.shape)
print(x2.shape)
print(y.shape)

pca = PCA(n_components=2)
pca.fit(x1)
x1 = pca.fit_transform(x1)
pca_std = np.std(x1)

pca = PCA(n_components=2)
pca.fit(x2)
x2 = pca.fit_transform(x2)
pca_std = np.std(x2)

x1_train, x1_test, x2_train, x2_test,  = train_test_split(
    x1, x2, random_state=1, train_size = 0.8, shuffle='True')
y_train, y_test = train_test_split(
    y, random_state=1, test_size = 0.2, shuffle='True')

scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

print(x1_train.shape)#402,15
print(x1_test.shape) #101,15
print(x2_train.shape)#402,10
print(x2_test.shape) #101,10
# print(y_train)
# print(y_test)
# print(y_train.shape)
# print(y_test.shape)

x1_train_scaled = x1_train_scaled.reshape(405,2,1)
x1_test_scaled = x1_test_scaled.reshape(102,2,1)
x2_train_scaled = x2_train_scaled.reshape(405,2,1)
x2_test_scaled = x2_test_scaled.reshape(102,2,1)

#모델 구성
input1 = Input(shape=(2,1))
dense1 = LSTM(500, activation='relu')(input1)
dense2 = Dense(1000, activation='relu')(dense1)
drop1  = Dropout(0.2)(dense2)
dense3 = Dense(1000, activation='relu')(drop1)
drop11 = Dropout(0.2)(dense3)
output1= Dense(100, activation='relu')(drop11)

input2 = Input(shape=(2,1))
dense1 = LSTM(500, activation='relu')(input1)
dense2 = Dense(1000, activation='relu')(dense1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(1000, activation='relu')(drop2)
drop2_1 = Dropout(0.2)(dense3)
output2 = Dense(100, activation='relu')(drop2_1)

merge1 = concatenate([output1, output2])

middle1 = Dense(500, activation='relu')(merge1)
drop5 = Dropout(0.3)(middle1)
middle1 = Dense(500, activation='relu')(drop5)
drop3 = Dropout(0.3)(middle1)
middle1 = Dense(20000, activation='relu')(drop3)
drop6 = Dropout(0.3)(middle1)
middle1 = Dense(500, activation='relu')(drop6)

output1 = Dense(1000, activation='relu')(middle1)
drop4 = Dropout(0.3)(output1)
output1_2 = Dense(500, activation='relu')(drop4)
drop4_1 = Dropout(0.3)(output1_2)
output1_3 = Dense(1)(drop4_1)


model = Model(inputs=[input1,input2], outputs = output1_3)

early_stopping = EarlyStopping(monitor= 'mse', patience=10)

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y_train, epochs=50, batch_size=1, validation_split=(0.25), verbose=1)

#4. 평가, 예측

loss, mse = model.evaluate([x1_test, x2_test],y_test, batch_size=1) 

print("loss: ", loss)

y1_predict, y2_predict = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
RMSE1 =  RMSE(y_test, y1_predict)
print("RMSE1 : ", RMSE1)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y1_predict)
print("R2_1 :", r2_y_predict )

