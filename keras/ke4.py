from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.datasets import cifar100
from keras.models import Model
from keras.models import load_model
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt


(x_train, y_train),(x_test, y_test) = cifar100.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

x_train = x_train.reshape(50000, 3072,).astype('float32') / 255 # 50000,32,32,1
x_test = x_test.reshape(10000, 3072,).astype('float32') / 255

input1 = Input(shape=(3072,))
dense1 = Dense(50)(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(30)(drop1)
dense3 = Dense(30)(dense2)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(30)(drop2)
drop5 = Dropout(0.2)(dense4)
dense5 = Dense(30)(drop5)
drop6 = Dropout(0.2)(dense5)
dense6 = Dense(30)(drop6)

output1 = Dense(30)(dense6)
output2 = Dense(50)(output1)
output3 = Dense(30)(output2)
drop3 = Dropout(0.2)(output3)
output4 = Dense(15)(drop3)
output5 = Dense(10)(output3)
drop4 = Dropout(0.2)(output5)
output6 = Dense(100, activation='softmax')(drop4)

model = Model(inputs = input1 ,outputs = output6)

early = EarlyStopping(monitor='loss', patience=10)

model.save('./model/sample/cifar100/cifar100_save.h5') # 모델 세이브
#model = load_model('./model/sample/cifar100/cifar100_checkpoint_best.hdf5')


model.save_weights('./model/sample/cifar100/cifar100_weight.h5')

modelpath = './model/sample/cifar100/{epoch:02d} - {val_loss:.4f}.hdf5'     # d : decimal, f : float
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, save_weights_only=False,  mode = 'auto', verbose=1)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, validation_split=0.1,callbacks=[early, checkpoint])

loss,acc = model.evaluate(x_test, y_test, batch_size = 10)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10, 6))

plt.subplot(2, 1 , 1)
plt.plot(hist.history['loss'], marker='.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='black', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(['acc', 'val_acc'])
plt.show()











