import numpy as np

x = np.array(range(1, 101)) 
y = np.array(range(1, 101))

x_train =x[:60]
x_val = x[60:80]
x_test = x[:80] #[:80] = 시작부터 80까지 [80:] =80부터 마지막까지
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim =1, activation ='relu'))
model.add(Dense(5, input_shape =(1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()

# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))

mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse:", mse)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # RMSE는 모델의 예측 값과 실제 값의 차이를 하나의 숫자로 표현
   return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)