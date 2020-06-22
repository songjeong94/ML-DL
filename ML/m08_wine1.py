import numpy as np
import pandas as pd
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


#최근날짜를 가장 아래로
#wine= wine.sort_values(['178'], ascending=[True])

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
    x, y, test_size = 0.3, shuffle = True, random_state = 1)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x1_train_scaled = scaler.transform(x_train)
x1_test_scaled = scaler.transform(x_test)

#model = KNeighborsClassifier(n_neighbors=1)
#model = SVC()
#model = LinearSVC()
#model = KNeighborsRegressor()
model = RandomForestClassifier()
#model = RandomForestRegressor()

#3.실행

model.fit(x_train, y_train)

#4.평가 예측

y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

#acc = accuracy_score(y_test, y_predict)

print(x_test,"의 예측 결과: ", y_predict )
#print("acc = ", acc)
print("score:", score) #분류일떄는 acc 회귀일때는 r2
