import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import keras
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

voice_path = './data/dacon/voice/'

labels=os.listdir(voice_path)
# wav 파일로부터 데이터를 불러오는 함수, 파일 경로를 리스트 형태로 입력
    no_of_recordings=[]
    for label in labels:
        waves = [f for f in os.listdir(voice_path + '/'+ label) if f.endswith('.wav')]
        no_of_recordings.append(len(waves))
        print(no_of_recordings)

# Wav 파일로부터 Feature를 만듭니다.
x_data = glob(train_path + 'train/train_00000.wav') # 파일을 리스트 형식으로 반환 
x_data = data_loader(x_data)
x_data = x_data[:, ::8] # 매 8번째 데이터만 사용
print(x_data)
x_data = x_data / 30000 # 최대값 30,000 을 나누어 데이터 정규화(Minmax)
print(x_data)
x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1) # CNN 모델에 넣기 위한 데이터 shape 변경
# 정답 값을 불러옵니다
y_data = pd.read_csv(train_path + 'train_answer.csv', index_col=0)
y_data = y_data.values

# Feature, Label Shape을 확인합니다.
x_data.shape, y_data.shape

# 모델을 만듭니다.
# model = Sequential()
# model.add(Conv1D(16, 32, activation='relu', input_shape=(x_data.shape[1], x_data.shape[2])))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(30, activation='softmax'))
# model.compile(loss=tf.keras.losses.KLDivergence(), optimizer='adam')

# # 모델 폴더를 생성합니다.
# model_path = 'model/'
# if not os.path.exists(model_path):
#   os.mkdir(model_path)

# # Validation 점수가 가장 좋은 모델만 저장합니다.
# model_file_path = model_path + 'Epoch_{epoch:03d}_Val_{val_loss:.3f}.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_file_path, monitor='val_loss', verbose=1, save_best_only=True)

# # 10회 간 Validation 점수가 좋아지지 않으면 중지합니다.
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# # 모델을 학습시킵니다.
# history = model.fit(
#     x_data, y_data, 
#     epochs=100, batch_size=256, validation_split=0.2, shuffle=True,
#     callbacks=[checkpoint, early_stopping]
# )

# # 훈련 결과를 확인합니다.
# plt.plot(history.epoch, history.history['loss'], '-o', label='training_loss')
# plt.plot(history.epoch, history.history['val_loss'], '-o', label='validation_loss')
# plt.legend()
# plt.xlim(left=0)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# # 검증 wav 파일로부터 Feature를 만듭니다.
# x_test = glob('data/test/*.wav')
# x_test = data_loader(x_test)
# x_test = x_test / 30000
# x_test = x_test[:, ::8]
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# # 가장 좋은 모델의 weight를 불러옵니다.
# weigth_file = glob('model/*.hdf5')[-1]
# print(weigth_file)
# model.load_weights(weigth_file)

# # 예측 수행
# y_pred = model.predict(x_test)

# # 예측 결과로 제출 파일을 생성합니다.
# submission = pd.read_csv('data/submission.csv', index_col=0)
# submission.loc[:, :] = y_pred
# submission.to_csv('submission.csv')
