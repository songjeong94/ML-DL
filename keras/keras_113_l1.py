# 20-07-03
#과적합 피하기1.
# L2 : 가중치 각 요소 제곱의 합
# L2 정규화는 아주 큰 값이나 작은 값을 가지는 outlier 모델 가중치에 대해 0에 가깝지만 0은 아닌 값으로 만든다.
# (0으로 수렴하도록 한다)
# L2 정규화는 선형모델의 일반화 능력을 언제나 항상 개선시킨다

# 각 가중치 제곱의 합에 규제 강도(Regularization Strength) 람다(lambda)를 곱한다.
# 그 값을 손실함수에 더한다. 람다를 크게 하면 가중치가 더 많이 감소되고(규제 중시), 람다를 작게하면 가중치가 증가한다(규제 중시 x)
# 가중치를 갱신할 때, 손실함수의 미분값을 이전 가중치에서 빼서 다음 가중치를 계산한다.

##### 데이터 LOAD
from keras.datasets import cifar10
from keras.regularizers import l1,l2,l1_l2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])  # [19]

print(x_train.shape)        # (50000, 32, 32, 3)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)
print(y_test.shape)         # (10000, 1)

# plt.imshow(x_train[0])
# plt.show()


##### 데이터 전처리 1. OneHotEncoding
# from keras.utils import np_utils

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)                 # (50000, 100)

##### 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


##### 2. 모델
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.1)))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.1)))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc'])
                        # 1e-4 = 0.0004
                              # loss= 이것은 원핫인코딩 안했을 때 하면 된다. (개인적인 취향) - 어쨌든 원핫인코딩은 해야한다 (다른방법임)
hist = model.fit(x_train, y_train,
          epochs=30, batch_size=32, verbose=2,
          validation_split=0.3)

loss = model.evaluate(x_test, y_test)


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc 는 ', acc)
print('val_acc 는 ', val_acc)

# evaluate 종속 결과
print('loss, acc 는 ', loss_acc)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# ''' ㅅㅂ
# # loss, acc 는  [3.433340796661377, 0.2328999936580658]
# '''