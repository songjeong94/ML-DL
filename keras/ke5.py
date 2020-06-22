from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt

import numpy as np


iris = load_iris()

x = iris.data
y = iris.target

y = np_utils.to_categorical(y)

minmax = MinMaxScaler()
#x = minmax.fit_transform(x)
#x = minmax.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 1)

print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(100, input_dim=4, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

model.save('./model/sample/iris/iris_save.h5') # 모델 세이브
#model = load_model('./model/sample/cifar100/cifar100_checkpoint_best.hdf5')


model.save_weights('./model/sample/iris/iris_weight.h5')

modelpath = './model/sample/iris/{epoch:02d} - {val_loss:.4f}.hdf5'     # d : decimal, f : float
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, save_weights_only=False,  mode = 'auto', verbose=1)


model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, validation_split=0.25, batch_size=50, callbacks=[checkpoint])

loss, acc = model.evaluate(x_test, y_test)

print("loss: ", loss)
print("acc: ", acc)

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)

print(y_predict)
