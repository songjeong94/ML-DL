# a08_ae4_cnn 복붙
# cnn으로 오토인코더 구성하시오
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose
from tensorflow.keras.datasets import mnist

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, strides=(2,2), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=2, strides=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7*7*64, activation='relu'))
    model.add(Reshape(target_shape=(7,7,64)))
    model.add(Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=1, kernel_size=2, strides=(2,2), padding='same', activation='sigmoid'))
    model.summary()
    return model

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_train = x_train/255
x_test = x_test/255

model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='mse',metrics=['acc']) #loss = 0.01 () 0.002
# model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc']) #loss = 0.09 () 0.06


model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

#이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_xlabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("output", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
