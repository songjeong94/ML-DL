# 앞뒤가 똑같은 오토인코더 = x가 인풋이면 x도 아웃풋
# 대한민국 축구대표팀. 좋은 경험이었다? --> 그 과정에서 특징을 추출함
# 특징을 추출한다? 어디서 배웠지? PCA(차원축소, 압축)

# 예시) 기태 사진에 잉크를 뿌렸다. 오토인코더에 넣으면 기태 얼굴이 나옴. 하지만 흐릿하게 나옴
# 잉크가 특징이 아니고 기태 얼굴이 특징이기 때문에

# 더 깔끔하게 만들고싶다? 그래서 나온게 GAN

import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()
print(x_train.shape)        # (60000, 28, 28)
print(x_test.shape)         # (10000, 28, 28)
print(y_train.shape)        # (60000,)
print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리 . OneHotEncoding
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)
# print(y_test.shape)         # (10000, 10)


#2. 데이터 전처리 . 정규화
x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255

X = np.append(x_train, x_test, axis=0)

print(X.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

best_n_components = np.argmax(cumsum >= 0.95)+1
print(cumsum>=0.95)
print(best_n_components) 