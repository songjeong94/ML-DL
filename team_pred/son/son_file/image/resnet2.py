from PIL import Image
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
from keras.applications import ResNet152V2
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

caltech_dir = "son_file/image2"
categories = ["song-test","son-test","kang-test"]
nb_classes = len(categories)
labels=os.listdir(caltech_dir)


image_w = 300
image_h = 400

pixels = image_h * image_w * 3

X = []
y = []

labels = ["song-test", "son-test", "kang-test"]
all_label = []

for label in labels:
    print(label)
    picture = [f for f in os.listdir(caltech_dir + '/'+ label) if f.endswith('.jpg')]
    for pic in picture:
        pic = cv2.imread(caltech_dir+ '/' + label + '/')
        all_label.append(label)


for idx, face in enumerate(categories):
    
    #one-hot encoding
    label = [0 for i in range(nb_classes)]
    for label in labels:
        image_dir = caltech_dir + "/" + labels
        files = glob.glob(image_dir+"/*.jpg")
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            
            X.append(data)
            y.append(label)
            
            if i % 700 == 0:
                print(face, " : ", f)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)


#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

history = additional_model.fit(X_train, y_train, 
                    batch_size=1, 
                    epochs=1, 
                    validation_data=(X_test, y_test))


input_tensor = Input(shape=(400, 300, 3), dtype='float32', name='input')

pre_trained_res = ResNet152V2(weights='imagenet', include_top=False, input_shape=(400, 300, 3))
pre_trained_res.trainable = False
pre_trained_res.summary()


additional_model = models.Sequential()
additional_model.add(pre_trained_res)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(4096, activation='relu'))
additional_model.add(layers.Dense(2048, activation='relu'))
additional_model.add(layers.Dense(1024, activation='relu'))
additional_model.add(layers.Dense(3, activation='softmax'))

additional_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = additional_model.fit(X_train, y_train, 
                    batch_size=1, 
                    epochs=1, 
                    validation_data=(X_test, y_test))

print("정확도 : %.4f" % (additional_model.evaluate(X_test, y_test)[1]))

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

model_json = additional_model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)

additional_model.save_weights("model_w.h5")
print("Saved model to disk")