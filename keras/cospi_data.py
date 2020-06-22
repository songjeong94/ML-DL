import numpy as np
import pandas as pd

kospi = pd.read_csv("./data/csv/kospi200.csv", index_col = 0, header=0, encoding='cp949', sep=',')
print(kospi)
print(kospi.shape)
samsung = pd.read_csv("./data/csv/samsung.csv", index_col=0, header=0, encoding='cp949', sep=',')
print(samsung)
print(samsung.shape)

#kospi200의 거래량
for i in range(len(kospi.index)):
    kospi.iloc[i,2] = float(kospi.iloc[i,2].replace(',', ''))
#삼성전자의 모든 데이터
for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',', ''))

#최근날짜를 가장 아래로
kospi = kospi.sort_values(['일자'], ascending=[True])
samsung = samsung.sort_values(['일자'], ascending=[True])
print(kospi)
print(samsung)

kospi = kospi.values
samsung = samsung.values
print(type(kospi), type(samsung))
print(kospi.shape, samsung.shape)

np.save('./data/kospi200.npy', arr=kospi)
np.save('./data/samsung.npy', arr=samsung)