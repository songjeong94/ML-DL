import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import random

train_path = './data/dacon/voice/'
samples, sample_rate = librosa.load(train_path + 'train/train_00000.wav', sr = 8000 )
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + './data/dacon/voice/train/train_00000.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
plt.show()
ipd.Audio(samples, rate =sample_rate)
print(sample_rate)
