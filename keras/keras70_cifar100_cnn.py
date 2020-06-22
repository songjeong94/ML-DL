#kears  53 복붙!
from keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from keras.models import Sequential, Model
from keras.callbacks import History, EarlyStopping, ModelCheckpoint

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')

modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'     # d : decimal, f : float
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')

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
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255

input1 = Input(shape=(32,32,3))
dense1 = Conv2D(100, (2,2), input_shape=(32,32,3))(input1)
max1 = MaxPooling2D(pool_size=2)(dense1)
dense2 = Conv2D(90, (2,2),padding='same')(max1)
dense3 = Conv2D(80,(2,2), padding='same')(dense2)
max2 = MaxPooling2D(pool_size=2)(dense3)
dense4 = Conv2D(70,(2,2), padding='same')(max2)
dense5 = Conv2D(60,(2,2), padding='same')(dense4)
dense6 = Conv2D(55,(2,2), padding='same')(dense5)
max3 = MaxPooling2D(pool_size=2)(dense6)
dense7 = Conv2D(50,(2,2), padding='same')(max3)
dense8 = Conv2D(45,(2,2), padding='same')(dense7)
max4 = MaxPooling2D(pool_size=2)(dense8)
dense9 = Conv2D(40,(2,2), padding='same')(max4)

fl1 = Flatten()(dense9)
output1 = Dense(250)(fl1)
drop1 = Dropout(0.3)(output1)
output2 = Dense(200)(drop1)
drop2 = Dropout(0.2)(output2)
output3 = Dense(150)(drop2)
output4 = Dense(150)(output3)
drop3 = Dropout(0.2)(output4)
output5 = Dense(150)(drop3)
output6 = Dense(100, activation='softmax')(output5)

model = Model(inputs = input1, outputs = output6)


# 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs = 30, batch_size = 50, validation_split = 0.01, callbacks = [es, cp])

# 평가 및 예측
loss_acc = model.evaluate(x_test, y_test)
# print("res : ", res)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss_acc : ', loss_acc)


import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))               # 그래프의 크기를 (10, 6) 인치로

plt.subplot(2, 1, 1)                        # 2행 1열의 그래프 중 첫번째 그래프
'''x축은 epoch로 자동 인식하기 때문에 y값만 넣어준다.'''
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')              
plt.plot(hist.history['val_loss'], marker = '.', c = 'black', label = 'val_loss')
plt.grid()                                  # 바탕에 격자무늬 추가
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)                        # 2행 1열의 두번째 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.grid()                                  # 바탕에 격자무늬 추가
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()

