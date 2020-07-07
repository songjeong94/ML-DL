from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "최고에요", "참 잘 만든 영화에요",
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요']
# 많이 사용되는 단어순으로 인덱스

# 긍정1 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post',value=0)
print(pad_x)

word_size = len(token.word_index) + 1
print("전체 토큰 길이:",word_size)

pad_x = pad_x.reshape(12,5,1)
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Conv1D

model = Sequential()
model.add(Conv1D(10,2,input_shape=(5,1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print(acc)