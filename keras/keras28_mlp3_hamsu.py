#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array([range(101, 201), range(711,811), range(100)]) #w  는1 b는100

print(x.shape) #  열우선, 행무시

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.8)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(1, ))
dense1_1 = Dense(30)(input1)
dense1_2 = Dense(30)(dense1_1)
dense1_3 = Dense(30)(dense1_2)
dense1_4 = Dense(30)(dense1_3)

output1 = Dense(3)(dense1_4)
output1_1 = Dense(30)(output1)
output1_2 = Dense(3)(output1_1)

model = Model(inputs = input1, outputs = output1)


#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode = 'auto')
model.fit(x_train, y_train, epochs=57, batch_size=1, validation_split=(0.25), callbacks=[early_stopping])
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("mse : ", mse)

#y_pred = model.predict(x_pred)
#print("y_predict : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
