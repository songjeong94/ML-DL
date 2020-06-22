
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from keras.models import  Model
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import numpy as np

diabets = load_diabetes()

x = diabets.data
y = diabets.target

print(x)
print(y)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 1)

print(x_train.shape)
print(x_test.shape)

input1 = Input(shape=(10,))
dense1 = Dense(10)(input1)
dense2 = Dense(100)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(100)(drop1)
dense4 = Dense(100)(dense3)
drop2 = Dropout(0.2)(dense4)

output1 = Dense(100)(drop2)
output2 = Dense(50)(output1)
output3 = Dense(1)(output2)

model = Model(inputs = input1, outputs = output3)

early_stopping=EarlyStopping(monitor='loss',patience=10,mode='aut')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs =50, batch_size = 2, callbacks=[early_stopping])

y_predict = model.predict(x_test)
print(y_predict)
# RMSE 구하기

from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(y_test, y_predict ))

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('R2는 ', r2)
