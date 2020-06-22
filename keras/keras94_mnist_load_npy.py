#kears  67 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.models import load_model

from keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0].shape)
# print("y_train: ", y_train[0])
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

x_train = np.load('./data/mnist_x_train.npy')
x_test = np.load('./data/mnist_x_test.npy')
y_train = np.load('./data/mnist_y_train.npy')
y_test = np.load('./data/mnist_y_test.npy')

# print(x_train[0].shape)
# #plt.imshow(x_train[0], 'gray')
# #plt.imshow(x_train[0])
# #plt.show()

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2.정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

model = Sequential()
model.add(Conv2D(10, (2,2), 
         input_shape=(28,28,1))) 
model.add(Conv2D(10,(2,2)))
model.add(Conv2D(10,(2,2)))
#model.add(Conv2D(5,(2,2), strides=2))
#model.add(Conv2D(5,(2,2),strides=2, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) # 2차원으로 변경
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))
model.summary()


#model.save('./model/model_test01.h5')
#modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'     # d : decimal, f : float
#checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                    # save_best_only = True, save_weights_only=False,  mode = 'auto', verbose=1)

                     
# 컴파일 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs = 10, validation_split=0.2)

# model = load_model('./model/09 - 0.0711.hdf5')
# model.load_weights('./model/test_weight1.h5')
# #model.save('./model/model_test01.h5')

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc =  hist.history['val_acc']

print('acc: ', acc)
print('val_acc: ', val_acc)


import matplotlib.pyplot as plt
plt.figure(figsize=(10 ,6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #plot 에서 라벨을 지정해줘서 legend 에서 지정하지 않았다.
plt.show()

plt.subplot(2, 1, 2)
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()