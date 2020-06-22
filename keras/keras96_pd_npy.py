#95번을 불러와서 모델을 완성하시오.
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import pandas as pd

iris = np.load('./data/arr.npy')
print(iris)

x = iris[:, 0:4] #[:]모든행 ,  [0:4] 0~3까지의 인데스 
y = iris[:, -1] 

y= np_utils.to_categorical(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8)

model = Sequential()
model.add(Dense(10, input_dim =4, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs= 30, batch_size= 10, validation_split= 0.2)

loss, acc = model.evaluate(x_test, y_test, batch_size=10)

print("loss: ", loss)
print("acc: ", acc)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

print(y_predict)


