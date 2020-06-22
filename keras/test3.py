#None 제거 1

samsung = samsung.dropna(axis = 0)
hite = hite.fillna(method='bfill')
hite = hite.dropna(axis=0)

#None 제거 2
hite = hite[0:509]
#hite.iloc[0, 1:5] = [10,20,30,40]
hite.loc['2020-06-02', '고가':'거래량'] = ['10', '20', '30', '40']

print(hite)