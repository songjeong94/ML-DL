import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.데이터
a =  np.array(range(1, 11)) #1~11의 범위
size = 5 

def split_x(seq, size):
    aaa = [] # 임심 메모리 리스트
    for i in range(len(seq) - size + 1):#seq - size +1  = 값 행종료값
        subset = seq[i: (i+size)] #열 지정
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=====================")
print(dataset)
