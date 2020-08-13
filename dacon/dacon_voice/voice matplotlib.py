import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import wavfile
from glob import glob
from tqdm import tqdm

%matplotlib inline

sns.set_style('darkgrid')

def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)    
    out = np.array(out)
    return out

%%time

# 데이터 불러오기
x_data = glob('./rsc/train/*.wav')
x_data = data_loader(x_data)

%%time

fs, data = wavfile.read('./rsc/train/train_00303.wav')
data = np.array(data)

plt.plot(data)

# !sudo pip install numba==0.43.0
# !sudo pip install llvmlite==0.32.1
# !sudo pip install librosa

import librosa.display, librosa

sig, sr = librosa.load('./rsc/train/train_00303.wav')

plt.figure()
librosa.display.waveplot(sig, sr, alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

# Fourier -> Spectrum

fft = np.fft.fft(sig)

magnitude = np.abs(fft) 

f = np.linspace(0,sr,len(magnitude))

left_spectrum = magnitude[:int(len(magnitude) / 2)]
left_f = f[:int(len(magnitude) / 2)]

plt.figure()
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")


# STFT -> spectrogram

hop_length = 256
n_fft = 1024

hop_length_duration = float(hop_length) / sr
n_fft_duration = float(n_fft) / sr

stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

magnitude = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(magnitude)

plt.figure()
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")

#wavfile의 data와 librosa의 waveform 이 살짝 다른데 어떤 차이가 있는건가요?
#[librosa.core.load]를 보시면 librosa 를 이용하여 load 한 audio 의 경우 자동으로 resample (given sampling rate=22050) 이 됩니다.
#  하지만 scipy.io.wavfile.read 의 경우 자동으로 resample 을 하지 않는다는 차이점이 있습니다. :) 