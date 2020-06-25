import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import random

#한가지 데이터 파형 확인
voice_path = './data/dacon/voice/'
samples, sample_rate = librosa.load(voice_path + 'train/train_00000.wav', sr = 16000 )
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('raw wave of ' + './data/dacon/voice/train/train_00000.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
plt.show()

labels=os.listdir(voice_path)

no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(voice_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
print(no_of_recordings)

labels = [""]

