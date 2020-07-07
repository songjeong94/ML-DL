from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print(x_train[0])
print(y_train[0])

#가장 긴 문장 출력
print(len(x_train[0]))

#y의 카테고리 개수 출력
category = np.max(y_train) + 1
print(category)
# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

#groupby의 사용법 숙지!
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre', truncating='pre')
# padding  앞에서 채우냐 뒤에서 채우냐 pre앞에서 post 뒤에서
# maxlen  문장의 가장긴 길이
# truncating   얼마나 잘라먹을거냐

# print(len(x_train[0]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
# model.add(Embedding(1000, 128, input_length=100))
model.add(Embedding(1000, 128))
model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker ='.', c= 'red', label ='TestSet loss')
plt.plot(y_loss, marker ='.', c= 'red', label ='TrainSet loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


