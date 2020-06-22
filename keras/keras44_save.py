# keras44_save.py
#keras40_lstm_split.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

#2.  모델
model = Sequential()
model.add(LSTM(10, input_shape=(4, 1)))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(100))

model.summary()

#model.save(".//keras//save_44.h5") #h5 확장자 사용
model.save("./model/save_44.h5") #h5 확장자 사용
#model.save(".\model\save_44.h5") #h5 확장자 사용

print("저장 잘됨.")