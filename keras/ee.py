import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping

samsung = np.load('./data/samsung.npy', allow_pickle=True)
hite = np.load('./data/hite.npy', allow_pickle=True)

# print(samsung)
# print(hite)
# print(samsung.shape)#508,1
# print(hite.shape)   #508,5

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :5]
        tmp_y = dataset[x_end_number:y_end_number, :-1 ]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(hite, 1, 1)
print(x2[0,:],"\n", y2[0])
print(x1.shape)
print(x2.shape)

x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test = train_test_split(
    x1, y1,x2, y2, random_state=1, train_size = 0.9, shuffle='True')
#x2_train, x2_test, y2_train, y2_test = train_test_split(
    #x2, y2, random_state=1, test_size = 0.2, shuffle='True')

print(x1_train.shape)#204,5,1
print(x1_test.shape) #101,5,1
print(x2_train.shape)#402,5,5
print(x2_test.shape) #101,5,5

    
# print(x1_train.shape)
# print(x1_test.shape)
# print(x2_train.shape)
# print(x2_test.shape)

# ####데이터 전처리#####

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler.fit(x1_train)
# x1_train_scaled = scaler.transform(x1_train)
# x1_test_scaled = scaler.transform(x1_test)
# scaler2 = StandardScaler()
# scaler2.fit(x2_train)
# x2_train_scaled = scaler2.transform(x2_train)
# x2_test_scaled = scaler2.transform(x2_test)

# x1_train_scaled = x1_train_scaled.reshape(228,5,)
# x1_test_scaled = x1_test_scaled.reshape(26,5,)
# x2_train_scaled = x2_train_scaled.reshape(228,5,)
# x2_test_scaled = x2_test_scaled.reshape(26,5,)

# #모델 구성
# modle = Sequential()
# input1 = Input(shape=(5,))
# dense1 = Dense(100, activation='relu')(input1)
# dense2 = Dense(1000, activation='relu')(dense1)
# drop1 = Dropout(0.2)(dense2)
# dense3 = Dense(1000, activation='relu')(drop1)
# drop1_1 = Dropout(0.2)(dense3)
# output1 = Dense(100, activation='relu')(drop1_1)

# input2 = Input(shape=(5,))
# dense1 = Dense(100, activation='relu')(input1)
# dense2 = Dense(1000, activation='relu')(dense1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(1000, activation='relu')(drop2)
# drop2_1 = Dropout(0.2)(dense3)
# output2 = Dense(100, activation='relu')(drop2_1)

# merge1 = concatenate([output1, output2])

# middle1 = Dense(500, activation='relu')(merge1)
# drop5 = Dropout(0.3)(middle1)
# middle1 = Dense(500, activation='relu')(drop5)
# drop3 = Dropout(0.3)(middle1)
# middle1 = Dense(20000, activation='relu')(drop3)
# drop6 = Dropout(0.3)(middle1)
# middle1 = Dense(500, activation='relu')(drop6)

# output1 = Dense(1000, activation='relu')(middle1)
# drop4 = Dropout(0.3)(output1)
# output1_2 = Dense(500, activation='relu')(drop4)
# output1_3 = Dense(1)(output1_2)

# output2 = Dense(1000, activation='relu')(middle1)
# drop5 = Dropout(0.3)(output2)
# output2_2 = Dense(500, activation='relu')(drop5)
# output2_3 = Dense(1)(output2_2)


# model = Model(inputs=[input1,input2], outputs = [output1_3, output2_3])

# early_stopping = EarlyStopping(monitor= 'mse', patience=10)

# model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
# model.fit([x1_train_scaled, x2_train_scaled], [y1_train,y2_train] ,epochs=50, batch_size=1, validation_split=(0.25), verbose=1) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

# #4. 평가, 예측

# mse = model.evaluate([x1_test_scaled, x2_test_scaled],[y1_test, y2_test], batch_size=1) 

# print("loss: ", mse)

# x_predict = 
# y_predict, y2_predict = model.predict([x1_test_scaled, x2_test_scaled])
# #y3_predict3 = datetime+1
# # y3_predict = y3_predict.reshape(1,5,1)
# # print(y3_predict)
# # RMSE 구하기

# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict ):
#     return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# RMSE1 =  RMSE(y1_test, y1_predict)
# RMSE2 =  RMSE(y2_test, y2_predict)
# #print("RMSE1 : ", RMSE1)
# #print("RMSE2 : ", RMSE2)
# print("RMSE : ", (RMSE1 + RMSE2 )/2 )


# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y1_test, y1_predict)
# r2_y2_predict = r2_score(y2_test, y2_predict)

# #print("R2_1 :", r2_y_predict )
# #print("R2_2 :", r2_y2_predict )
# print("R2 :", (r2_y_predict + r2_y2_predict )/2)


