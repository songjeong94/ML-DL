import numpy as np
import pandas as pd

samsung = pd.read_csv("./data/csv/삼성전자 주가.csv", index_col = 0, header=0, encoding='cp949', sep=',')
print(samsung.shape) #700,1
hite = pd.read_csv("./data/csv/하이트 주가.csv", index_col=0, header=0, encoding='cp949', sep=',')

samsung = samsung[0:509]
#samsung = samsung.dropna(how ='all')
print(samsung) 
print(samsung.shape)
#hite = hite.dropna(how ='all')
hite = hite[0:509]

hite.fillna(0).astype(str)
print(hite)
print(hite.shape)



for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',', ''))

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        if hite.fillna():
            hite.fillna(0)
            hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',''))

#최근날짜를 가장 아래로
samsung = samsung.sort_values(['일자'], ascending=[True])
hite = hite.sort_values(['일자'], ascending=[True])
print(samsung)
print(hite)

samsung = samsung.values
hite = hite.values
print(type(hite), type(samsung))
print(hite.shape, samsung.shape)

np.save('./data/hite.npy', arr=hite)
np.save('./data/samsung.npy', arr=samsung)