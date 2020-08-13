import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung.csv',
                    index_col = 0, 
                    header=0,
                    sep=',',
                    encoding='CP949') #cp949 한글처리

hite = pd.read_csv('./data/csv/hite.csv',
                    index_col = 0, 
                    header=0,
                    sep=',',
                    encoding='CP949')

# print(samsung)
# print(hite.head())

#None 제거1
samsung = samsung.dropna(axis = 0)
print(samsung)
print(samsung.shape)
hite = hite.fillna(method='bfill')
hite = hite.dropna(axis=0)

#None 제거2
#hite = hite[0:509]
#hite.iloc[0, 1:5] = [10,20,30,40] #index location
#hite.loc['2020-06-02', '고가':'거래량'] = ['10,000', '20,000', '30,000', '40,000']

print(hite)

# 삼성과 하이트의 정렬을 오름차순 변경

samsung = samsung.sort_values(['일자'], ascending=['True']) #decendig=내림차순
hite = hite.sort_values(['일자'], ascending=['True']) #decendig=내림차순

print(samsung)
print(hite)

# 콤마제거, 문자를 정수로 형변환
for i in range(len(samsung.index)):
    samsung.iloc[i, 0] = int(samsung.iloc[i, 0].replace(',',''))

print(samsung)
print(type(samsung.iloc[0,0]))

for i in range(len(hite.index)): 
    for j in range(len(hite.iloc[i])): 
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',',""))

print(hite)
print(type(hite.iloc[1,1]))

print(samsung.shape)
print(hite.shape)

samsung = samsung.values
hite = hite.values

print(type(hite))

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)