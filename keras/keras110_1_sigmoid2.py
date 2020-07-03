#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout,concatenate

model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3.컴파일
model.compile(loss = ['binary_crossentropy'], optimizer='adam',
                metrics =['acc'])

model.fit(x_train,y_train, epochs=1, batch_size=1)

# 4.평가예측
loss = model.evaluate(x_train,y_train)
print("loss: ", loss)

x_pred = np.array([11,12,13,14])

y_pred = model.predict(x_pred)
print(y_pred)



