from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Dense, LSTM, Conv2D, Input 
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(50000,32,32,3 ).astype('float32') / 255 
x_test = x_test.reshape(10000,32,32,3 ).astype('float32') / 255

input1 = Input(shape=(32,32,3))
dense1 = Conv2D(30,(2,2),input_shape=(32,32,3))(input1)
dense2 = Conv2D(30,(2,2))(dense1)
max1 =(MaxPooling2D(pool_size=2))(dense2)
dense3 = Conv2D(30,(2,2))(max1)
dense4 = Conv2D(30,(2,2))(dense3)
drop3 = Dropout(0.1)(dense4)
dense5 = Conv2D(30,(2,2))(drop3)
max2 =(MaxPooling2D(pool_size=2))(dense5)

fl1 = Flatten()(max2)
output1 = Dense(100)(fl1)
drop1 = Dropout(0.2)(output1) 
output2 = Dense(100)(drop1)
output3 = Dense(100)(output2)
drop2 = Dropout(0.2)(output3) 
output4 = Dense(100)(drop2)
output5 = Dense(10, activation='softmax')(output4)

model = Model(inputs =input1 , outputs = output5)

modelpath = './model/sample/cifar10/{epoch:02d} - {val_loss:.4f}.hdf5'     # d : decimal, f : float
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, save_weights_only=False,  mode = 'auto', verbose=1)


model.save('./model/sample/cifar10/cifar10_model_save.h5') # 모델 세이브
#model = load_model('./model/sample/cifar10/cifar10_checkpoint_best.hdf5')


model.save_weights('./model/sample/cifar10/cifar10_weight.h5')
model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 20, callbacks = [checkpoint], validation_split=0.2)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

y_test = np.argmax(y_test, axis=1)
print(y_test)
