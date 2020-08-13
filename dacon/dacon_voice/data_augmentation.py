import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython.display
# import torch

def noising(data, noise_factor):
    """
    원본 데이터에 노이즈를 추가합니다.
    noise factor를 통해 조절합니다.
    """
    noise = np.random.randh(len(data))
    augmented_data = data + noise_dactor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shifting(data, sampling_rate, shift_max, shift_direction):
    """
    원본 데이터를 좌우로 이동시킵니다.
    shift_max를 통해 최대 얼마까지 이동시킬지 조절합니다.
    """
    shift = np.random.randint(sampling_rate * shift_max+1)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shitf)
    
    #Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shitf:] = 0
    return augmented_data

def change_pitch(data, sampling_rate, pitch_factor):
    """
    원본 데이터의 피치를 조절합니다
    """

# 원본 데이터
data, sr = librosa.load("D:\study\data\dacon/voice/train/train_01000.wav")
IPython.display.Audio(data=data, rate=sr)

mfcc_data = librosa.feature.melspectrogram(data, sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)
print(mfcc_data.shape)
plt.pcolor(mfcc_data)
plt.show()

# 노이즈 추가
noising_data = noising(data, np.random.uniform(0, 0, 5))
IPython.display.Audio(data=noising_data, rate=sr)

mfcc_data = librosa.feature.melspectrogram(noising_data, sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)
print(mfcc_data.shape)
plt.pcolor(mfcc_data)
plt.show()

# 좌우 이동
shifting_data = shifting(data, sr, 0.3, 'both')
IPython.display.Audio(data=shifting_data, rate=sr)

mfcc_data = librosa.feature.melspectrogram(shifting_data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80)
print(mfcc_data.shape)
plt.pcolor(mfcc_data)
plt.show()

# 피치 조절
pitch_data = change_pitch(data, sr, np.random.randit(-5, 5))
IPython.display.Audio(data=pitch_data, rate=sr)

mfcc_data = librosa.feature.melspectrogram(pitch_data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80)
print(mfcc_data.shape)
plt.pcolor(mfcc_data)
plt.show()

## 증강 데이터 저장
train_files = os.listdir("D:\study\data\dacon/voice/train")
try:
    os.mkdir('aug_data')
    os.mkdir('aug_data/train')
    os.mkdir('aug_data/test')
except:
    pass

for file in tqdm(train_files):
    data,sr = librosa.load('D:\study\data\dacon/voice/train/'+file)
    mfcc_data = torch.tensor(librosa.feature.melspectrogram(data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80))
    torch.save(mfcc_data,'aug_data/train/'+file.split('_')[1].split(".")[0]+'_0.pt')
    
    noise_data = noising(data,np.random.uniform(0,0.5))
    mfcc_data = torch.tensor(librosa.feature.melspectrogram(noise_data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80))
    torch.save(mfcc_data,'aug_data/train/'+file.split('_')[1].split(".")[0]+'_1.pt')
    
    shift_data = shifting(data,sr,np.random.uniform(0,0.5),'both')
    mfcc_data = torch.tensor(librosa.feature.melspectrogram(shift_data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80))
    torch.save(mfcc_data,'aug_data/train/'+file.split('_')[1].split(".")[0]+'_2.pt')
    
    pitch_data = change_pitch(data,sr,np.random.randint(-5,5))
    mfcc_data = torch.tensor(librosa.feature.melspectrogram(pitch_data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80))
    torch.save(mfcc_data,'aug_data/train/'+file.split('_')[1].split(".")[0]+'_3.pt')

test_files = os.listdir('data/test')

for file in test_files:
    data,sr = librosa.load('D:\study\data\dacon/voice/test/'+file)
    mfcc_data = torch.tensor(librosa.feature.melspectrogram(data,sr,n_fft=512,win_length=400,hop_length=160,n_mels=80))
    torch.save(mfcc_data,'aug_data/test/'+file.split('_')[1].split(".")[0]+'.pt')