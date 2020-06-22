from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

boston = load_boston()
x = boston.data
y = boston.target

print(x.shape)
print(y.shape)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

pca = PCA(n_components=2)
pca.fit(x)
x = pca.fit_transform(x)
pca_std = np.std(x)

print(x.shape) #506,2

x = x.reshape(506,2,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size=0.8)

model = Sequential()
model.add(LSTM(10,input_shape = (2,1), activation = 'relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size = 1, validation_split=0.23)

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
