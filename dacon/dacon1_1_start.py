import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col=0, sep = ',')
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col=0)

print('train.shape:', train.shape) #(10000, 75) : x_train, test 71 , y_train , test 4
print('test.shape:', test.shape)   #(10000, 71) : x_predict
print('submission:', submission.shape) #(10000, 4) :y_predict


train = train.interpolate() #보간법 //선형보간 (값 중간중간 선을 이어서 대체함)
print(train.isnull().sum())
test = test.interpolate() #보간법 //선형보간 (값 중간중간 선을 이어서 대체함)
print(test.isnull().sum())

train = train.fillna(method='bfill')
test = test.fillna(method='bfill')

train = train.sort_values(['rho'], ascending=['True'])
print(train)
train = train.values
test = test.values
submission = submission.values

np.save('./data/dacon/comp1/train.npy', arr=train)
np.save('./data/dacon/comp1/test.npy', arr=test)
np.save('./data/dacon/comp1/submission.npy', arr=submission)

# 서브밋 파일 만든다.
# y_pred.to_csv(경로)
