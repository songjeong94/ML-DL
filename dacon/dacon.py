import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input 
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score
import pandas as pd
train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col=0)

train = train.interpolate() #보간법 //선형보간 (값 중간중간 선을 이어서 대체함)
print(train.isnull().sum())
test = test.interpolate() #보간법 //선형보간 (값 중간중간 선을 이어서 대체함)
print(test.isnull().sum())

train = train.fillna(method='bfill')
test = test.fillna(method='bfill')

x = train.iloc[:, :36]
y = train.iloc[:, 71:]
test = test.iloc[:, :36]


stan = StandardScaler()
x = stan.fit_transform(x)

print(x.shape)
print(y.shape)
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
model = RandomForestRegressor()
acc = accuracy_score(x_test, y_test)

model.fit(x_train,y_train)
# mae = mean_absolute_error(y_test)
# print("mae:", mae)

print("acc:", acc)
