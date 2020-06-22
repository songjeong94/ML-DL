import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

wine = pd.read_csv("./data/csv/winequality-white.csv", index_col = 0 ,header = 0, encoding='cp949', sep=';')
print(wine)
print(wine.shape) 

wine = wine.values

np.save('./data/wine.npy', arr=wine )

wine = np.load('./data/wine.npy', allow_pickle=True)

x = wine[:, 0:10]
y = wine[:, 10:]
print("x는:",x)
print("y는:",y)
print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 1)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x1_train_scaled = scaler.transform(x_train)
x1_test_scaled = scaler.transform(x_test)
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(3, input_dim=10, activation='softmax'))

#3.실행
model.compile(loss ='categorical_crossentropy',optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,epochs=100, batch_size=1)

#4.평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)
#score = model.score(x_test, y_test)

#acc = accuracy_score(y_test, y_predict)
y_test = np.argmax(y_test, axis=1)
print(acc)
print(y_test)
#print(x_test,"의 예측 결과: ", y_predict )
#print("acc = ", acc)
#print("score:", score) #분류일떄는 acc 회귀일때는 r2
