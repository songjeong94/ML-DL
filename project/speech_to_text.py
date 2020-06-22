import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import random

warnings.filterwarnings("ignore")

#데이터 탐색 및 시각화를 통하여 사전처리 단계 이해
train_audio_path = './project/data/train/audio/'
samples, sample_rate = librosa.load(train_audio_path+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + './project/data/train/audio/yes/0a7c2a8d_nohash_0.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

#오디오 신호의 샘플링 속도 살펴보기
ipd.Audio(samples, rate=sample_rate)
print(sample_rate)


#위의 신호의 샘플링 속도는 16,000Hz 이지만 대부분의 음성 주파수가 8000Hz에 있으므로 8000으로 샘플링
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)

#각 음성 명령에 대한 녹음횟수를 이해해보는단계
labels=os.listdir(train_audio_path)

#각 라벨의 개수와 플롯 막대 그래프를 찾습니다.
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
print(no_of_recordings)
#plot
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()

labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

#녹음시간에 대해서 살표보기
# duration_of_recordings=[]
# for label in labels:
#     waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
#     for wav in waves:
#         sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
#         duration_of_recordings.append(float(len(samples)/sample_rate))
    
# plt.hist(np.array(duration_of_recordings))


#이전 데이터 탐색에서 몇번의 레코딩 지속시간이 1초 미만이고 샘플링 속도가 너무 높은것으로 나타났습니다.
#따라서 오디오 파를 읽고 아래의 전처리 단계를 사용하여이를 처리하겠습니다.
#다음은 2가지 단계 입니다.
#리샘플링 , 1초 미만의 짧은 명령 제거
#아래의 코드로 전처리 단계 정의
train_audio_path = './project/data/train/audio'

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)
print

print("all_wave:", all_wave)
print("all_label:", all_label)

#출력 레이블을 정수로 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)
print("classes:", classes)
#이제 다중 인코딩 문제이므로 정수로 인코딩된 레이블을 원핫인코딩을 통해 벡터로 변환
from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))

#conv1d에 입력은 3D배열이어야하므로 차원 재구성
all_wave = np.array(all_wave).reshape(-1,8000,1)
print(all_wave.shape)

#트레인 테스트 데이터 분리
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y), stratify=y, test_size = 0.2, random_state=1, shuffle=True)

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras import backend as K
K.clear_session()
# 모델 구성
inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.2)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.2)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.2)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.2)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.2)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.2)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs = inputs, outputs = outputs)
model.summary()


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 

modelpath = ('./STUDY/project/data/{epoch:02d} - {val_loss:.4f}.hdf5')
mc = ModelCheckpoint(filepath= modelpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit(x_tr, y_tr ,epochs=50, callbacks=[es, mc], batch_size=8, validation_data=(x_val,y_val))


#모델 성능 시각화
from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend()
pyplot.show()

from keras.models import load_model
#model=load_model('./project/data/best_model.hdf5')

#주어진 오디오파일의 텍스트를 예측하는 함수 정의
def predict(audio):
    prob=model.predict(audio.reshape(-1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

#예측시간 val 데이터 예측
import random
index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:",predict(samples))

import sounddevice as sd
import soundfile as sf

samplerate = 16000  
duration = 2 # seconds
filename = 'predict.wav'
print("녹음시작")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
print("녹음끝")
sd.wait()
sf.write(filename, mydata, samplerate)

os.listdir('D:\Study')
filepath = 'D:\Study'

#저장된 음성 명령을 읽고 텍스트로 변환하겠습니다.
samples, sample_rate = librosa.load(filepath +'/predict.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)
print("'송정현님' 께서 말씀하신 단어는 :",predict(samples))
print('비트캠프 송정현')