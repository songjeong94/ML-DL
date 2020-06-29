import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

train = pd.read_csv('./data/dacon/comp2/train_features.csv', header = 0, index_col=[0,1])
target = pd.read_csv('./data/dacon/comp2/train_target.csv', header = 0, index_col=0)
test = pd.read_csv('./data/dacon/comp2/test_features.csv', header = 0, index_col=[0,1])
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header = 0, index_col=0)

print(train.isnull().sum())
print(target.isnull().sum())
print(test.isnull().sum())

train = train.groupby(['id']).count()
test = test.groupby(['id']).mean()

print(train.shape)
print(test.shape)

train = train.values
test = test.values
target = target.values
submission = submission.values


np.save('./data/dacon/comp2/train.npy', arr=train)
np.save('./data/dacon/comp2/test.npy', arr=test)
np.save('./data/dacon/comp2/target.npy', arr=target)
np.save('./data/dacon/comp2/submission.npy', arr=submission)