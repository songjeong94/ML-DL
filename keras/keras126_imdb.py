from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000)

print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)

print(x_train[0])
print(y_train[0])
num_classes = np.max(y_train) + 1
print("카테고리 : {}".format(num_classes))

len_result = [len(s) for s in x_train]

print('리뷰의 최대 길이: {}'.format(np.max(len_result)))
print('리뷰의 평균 길이: {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))

word_to_index = imdb.get_word_index()
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

print('빈도수 상위 1번 단어: {}'.format(index_to_word[1]))
print('빈도수 상위 3941번 단어: {}'.format(index_to_word[3941]))

for index, token in enumerate(("<pad>","<sos>","<unk>")):
    index_to_word[index]=token

print(' '.join([index_to_word[index] for index in x_train[0]]))
#첫번째 훈련용 리뷰의 X_train[0]이 인덱스로 바뀌기 전에 어떤 단어들이었는지 확인해보겠습니다. (물론 인덱스로 바꾸기 전에도 어느정도 전처리가 된 상태라서 제대로 된 문장이 나오지는 않습니다.)

#groupby의 사용법 숙지!
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)
# # padding  앞에서 채우냐 뒤에서 채우냐 pre앞에서 post 뒤에서
# # maxlen  문장의 가장긴 길이
# # truncating   얼마나 잘라먹을거냐

# print(len(x_train[0]))
# print(len(x_train[-1]))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Dropout, Activation,MaxPooling1D

model = Sequential()
# model.add(Embedding(1000, 128, input_length=111))
model.add(Embedding(5000, 128))
model.add(Conv1D(64,5, padding='valid', activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(120))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker ='.', c= 'red', label ='TestSet loss')
plt.plot(y_loss, marker ='.', c= 'black', label ='TrainSet loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


#1. imdb 검색해서 데이터 내용 확인.
#2. word_size 전체데이터에서 최상값 확인
#3. 주간과제: groupby()의 사용법 숙지할것.