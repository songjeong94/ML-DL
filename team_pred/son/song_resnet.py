# from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import cv2

from keras import layers, models
from keras.applications import ResNet152V2, VGG16, MobileNetV2
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from PIL import Image
# from PIL import Image
from tensorflow.keras.datasets import mnist, cifar10

song_path = 'D:\study/team_pred\son\son_file\image/'
categories = ['song-test', "kang-test"]
nb_classes = len(categories)

image_w = 100
image_h = 100

X = []
Y = []

for idx, cate in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = song_path + cate
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

x = np.array(X)
y = np.array(Y)

print(x.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state=1, shuffle=True)

x_train = x_train/255
x_test = x_test/255

input_tensor = Input(shape=(100, 100, 3), dtype='float32', name='input')

pre_trained_res = MobileNetV2(include_top=False, input_tensor=input_tensor)
pre_trained_res.trainable = False
pre_trained_res.summary()


additional_model = models.Sequential()
additional_model.add(pre_trained_res)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(2048, activation='relu'))
additional_model.add(layers.Dense(1024, activation='relu'))
additional_model.add(layers.Dense(2, activation='softmax'))

additional_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# model.save('./model/model_test01.h5')
# additional_model.load_weights('model_w.h5')

history = additional_model.fit(x_train, y_train, 
                    batch_size=1, 
                    epochs=1, 
                    validation_data=(x_test, y_test))

print("정확도 : %.4f" % (additional_model.evaluate(x_test, y_test)[1]))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# model_json = additional_model.to_json()
# with open("model.json", "w") as json_file : 
#     json_file.write(model_json)

# additional_model.save_weights("model_w.h5")
# print("Saved model to disk")

