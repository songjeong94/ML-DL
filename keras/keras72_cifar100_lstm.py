import numpy as np
from keras.models import Model, Sequential
from keras.layers import LSTM, Flatten, Dense, Input
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, History

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(50000, 1024, 3).astype('float32') / 255
x_test = x_test.reshape(10000, 1024, 3).astype('float32') / 255

input1 = Input(shape = (1024,3))
dense1  = LSTM(40, input_shape=(1024,3))(input1)
dense2 = Dense(200)(dense1)
dense3 = Dense(190)(dense2)
dense4 = Dense(180)(dense3)
dense5 = Dense(150)(dense4)

output1 = Dense(140)(dense5)
output2 = Dense(130)(output1)
output3 = Dense(100, activation = 'softmax')(output2)

model = Model(inputs = input1, outputs = output3)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
early = EarlyStopping(monitor='loss', patience=10)
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'
mp= ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)

hist = model.fit(x_train, y_train, epochs = 20, batch_size=50, validation_split=0.1, callbacks=[early, mp])


loss, acc = model.evaluate(x_test, y_test)

loss = hist.history('loss')
val_loss= hist.history('val_loss')
acc = hist.history('acc')
val_acc =hist.history('val_acc')

plt.figure(figsize = (12, 6))  

plt.subplot(2, 1 ,1)
plt.plot(hist.history['loss'], marker = '.', c ='black', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'green', label = 'loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'upper right')

plt.subplot(2, 1 ,1)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend([acc, val_acc])
plt.show()






