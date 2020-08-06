#kaggle : Finging and Measuring Lungs in CT Data

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Activation, Conv2D,Flatten,Dense
from keras.layers import MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau

x_train = np.load('dataset/x_train.npy')
y_train = np.load('dataset/x_train.npy')
x_val = np.load('dataset/x_val.npy')
y_val = np.load('dataset/y_val.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

#maxpooling = downsampling

inputs = Input(shape=(256, 256, 1))
#encoder
net = Conv2D(32, kernel_size=3, activation='reli', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(128, activation='relu')(net)

#decoder
net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, action='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, action='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(1, kernel_size=3, action='sigmoid', padding='same')(net)

model = Model(inputs=inputs, ouputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])
model.summary()

history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
epochs=100, batch_size=32, callbacks=[ 
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1,
    mode='auto', min_lr=1e-05)])

fig,ax = plt.subplots(2, 2, figsize=(10,7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss','r'])
ax[0, 0].set_title('acc')
ax[0, 0].plot(history.history['acc'], 'b')

ax[0, 0].set_title('val_loss')
ax[0, 0].plot(history.history['val_loss','r--'])
ax[0, 0].set_title('val_acc')
ax[0, 0].plot(history.history['val_acc'], 'b--')

preds = model.predict(x_val)

fig, ax = plt.subplots(len(x_val),3, figsize=(10, 100))

for i, pred in enumerate(preds):
    ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(x_val[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')