import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils


#1. 데이터
x = np.array(range(1, 11))
y = np.array([1,2,3,4,5,1,2,3,4,5])

y = np_utils.to_categorical(y)
print(y)
y = y[:,1:]
print(x.shape)
print(y)
print(y.shape) #(10,5)

#2.모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(5, activation='softmax'))# 다중분류 softmax

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=100, batch_size = 1,)

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y = np.argmax(y, axis=1)+1
print(y)
#과제 : dim 6 에서 5로 변경
#y_pred를 숫자로 바꿔라
