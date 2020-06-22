from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

#1. 데이터
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])


print(type(x_data))
print(x_data.shape)
#2.모델
model = Sequential()
model.add(Dense(10, input_dim = 2))
model.add(Dense(100, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3.실행
model.compile(loss ='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=1000)

loss, acc = model.evaluate(x_data,y_data)

#4.평가 예측
x_test = np.array([[0, 0], [1, 0], [0, 1], [1,1]])
#y_predict = model.predict(x_test)


#print(x_test,"의 예측 결과: ", y_predict )
print("acc = ", acc)
