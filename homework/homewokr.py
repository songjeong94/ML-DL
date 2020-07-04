#360p
#퍼셉트론
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2, 0.5]])

#371p
#다중 퍼셉트론 만들기
import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

x_train_full.shape # 60000,28,28
x_train_full.dtype # uint8

x_valid, x_train = x_train_full[:5000]/255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker","Bag","Ankle boot"]

# 일반 모델
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

#리스트 형식의 모델
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28,28]),
keras.layers.Dense(300, activation='relu'),
keras.layers.Dense(100, activation='relu'),
keras.layers.Dense(10, activation='softmax')])

#인덱스나 이름으로 층을 선택하는 모델
# model.layers
# [<tensorflow.python.keras.layers.core.Flatten at 0x132414e48>,
# tensorflow.python.keras.layers.core.Flatten at 0x1324149b0>,
# tensorflow.python.keras.layers.core.Flatten at 0x1356ba8d0>,
# tensorflow.python.keras.layers.core.Flatten at 0x13240d240>]
# hidden1 = model.layers[1]
# hidden1.name # dense
# model.get_layer('dense') is hidden1 # True

#층의 모든 파라미터는 get_weights()와 se_weights() 로 접근할 수 있다. 가중치와 편향 모두
# weights, biases = hidden1.get_weights() #array([[0.02448617, -0.00877795, -0.02189048,....,-0.02766046,]])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
history = model.fit(x_train,y_train, epochs=30, validation_data=(x_valid,y_valid))

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8.5))
plt.grid(True)
plt.gca().get_ylim(0,1)
plt.show()

x_new = x_test[:3]
y_proba = model.predict(x_new)
y_proba.round(2)
array([0])

y_pred = model.predict_classes(x_new)
array([9, 2, 1])
np.array(class_names)[y_pred]
array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')



