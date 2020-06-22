from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv2D,Flatten,MaxPooling2D
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()

x = iris.data
y = iris.target

#y = np_utils.to_categorical(y)

stan = StandardScaler()
x = stan.fit_transform(x)
x = stan.transform(x)

y=np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = False)

print(x_train.shape)
print(x_test.shape)
print(y.shape)

x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model=Sequential()
model.add(Conv2D(10,(2,2),input_shape=(2,2,1),activation='relu',padding="same"))
model.add(Conv2D(50,(2,2),activation='relu',padding="same"))
model.add(Conv2D(45,(2,2),activation='relu',padding="same"))
model.add(Conv2D(40,(2,2),activation='relu',padding="same"))
model.add(Conv2D(35,(2,2),activation='relu',padding="same"))
model.add(Conv2D(30,(2,2),activation='relu',padding="same"))
model.add(Conv2D(25,(2,2),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, validation_split=0.25, batch_size=10)

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss: ", loss)
print("acc: ", acc)

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)

print(y_predict)