from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000)

print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)

#가장 긴 문장 출력
print("가장 긴문장 출력:",len(x_train[0]))

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print("y의 유니크 값:",y_bunpo)

#groupby의 사용법 숙지!
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print("y의 그룹바이 데이터:",bbb)
print("y의 그룹데이터 차원:",bbb.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre', truncating='pre')
# padding  앞에서 채우냐 뒤에서 채우냐 pre앞에서 post 뒤에서
# maxlen  문장의 가장긴 길이
# truncating   얼마나 잘라먹을거냐

print(len(x_train[0]))
print(len(x_train[-1]))

print(x_train.shape, x_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Dropout, Activation,MaxPooling1D, Bidirectional

model = Sequential()
# model.add(Embedding(1000, 128, input_length=111))
model.add(Embedding(1000, 128))
model.add(Conv1D(64,5, padding='valid', activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)

# y_val_loss = history.history['val_loss']
# y_loss = history.history['loss']

# plt.plot(y_val_loss, marker ='.', c= 'red', label ='TestSet loss')
# plt.plot(y_loss, marker ='.', c= 'black', label ='TrainSet loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


#1. imdb 검색해서 데이터 내용 확인.
#2. word_size 전체데이터에서 최상값 확인
#3. 주간과제: groupby()의 사용법 숙지할것.