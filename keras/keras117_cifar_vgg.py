from keras.applications import VGG16
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout,BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.callbacks import History, EarlyStopping, ModelCheckpoint

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

input_tensor = Input(shape=(32,32,3))

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256)(top_model)
top_model = BatchNormalization()
top_model = Activation('relu')
top_model = Dense(10, activation='softmax')

model = Model(inputs=vgg16.input, ouputs=top_model)

for layer in model.layers[:19]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=30,batch_size=64)

loss, acc = (x_test, y_test)

print(acc)
print(loss)

pred = np.agmax(model.predict(x_test[0:10]), axis=1)
print(pred)