#kears  67 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import History

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0].shape)
print("y_train: ", y_train[0])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train[0].shape)
#plt.imshow(x_train[0], 'gray')
#plt.imshow(x_train[0])
#plt.show()

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2.정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255


#model.save('./model/model_test01.h5')
from keras.models import load_model

model = load_model('./model/model_test01.h5')
model.add(Dense(100, name = '1'))
model.add(Dense(100, name = '2'))
model.add(Dense(10, name = '3', activation='softmax'))

model.summary()

#model.save('./model/model_test01.h5')

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc =  hist.history['val_acc']

print('acc: ', acc)
# print('val_acc: ', val_acc)


# import matplotlib.pyplot as plt
# plt.figure(figsize=(10 ,6))

# plt.subplot(2, 1, 1)
# plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
# #plt.plot(hist.history['acc'])
# #plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') #plot 에서 라벨을 지정해줘서 legend 에서 지정하지 않았다.
# plt.show()

# plt.subplot(2, 1, 2)
# #plt.plot(hist.history['loss'])
# #plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])
# plt.show()


#acc:  [0.92791665, 0.9648125, 0.9702917, 0.9730833, 0.97497916, 0.97641665, 0.97702086, 0.97864586, 0.98070836, 0.97945833]
#val_acc:  [0.9635000228881836, 0.9726666808128357, 0.9765833616256714, 0.971750020980835, 0.9698333144187927, 0.9782500267028809, 0.9794999957084656, 0.9779166579246521, 0.9700000286102295, 0.981333315372467]