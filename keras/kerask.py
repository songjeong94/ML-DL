from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.preprocessing import StandardScaler

boston = load_boston()
x = boston.data
y = boston.target

print(x.shape)
print(y.shape)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8)


model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, batch_size = 1, validation_split=0.25 )

loss, mse = model.evaluate(x_test, y_test)

print("loss: ", loss)
print("mse: ", mse)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)

