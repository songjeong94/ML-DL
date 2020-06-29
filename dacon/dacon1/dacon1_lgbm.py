import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor
from sklearn.ensemble import IsolationForest
import warnings 

import pandas as pd
train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')

x = train[:, :71]
y = train[:, 71:]
0
test = test[:, :71]

# 이상치 제거
# clf = IsolationForest(max_samples=1000, random_state=1)
# clf.fit(x)
# pred_outliers = clf.predict(x)
# out = pd.DataFrame(pred_outliers)
# out = out.rename(columns={0:"out"})
# new_train = pd.concat([x, out],1)


print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle='True', random_state=1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = LGBMRegressor(n_estimators=300,
learning_rate = 0.01,
max_depth=40,
num_iterations=2000,
feature_fraction=0.9,
colsample_bytree=0.9,
num_leaves=1000, n_jobs=-1)
# model = GridSearchCV(model, parameters, cv =5)

model = MultiOutputRegressor(model)

warnings.filterwarnings('ignore')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# print(y_pred)
# print(y_pred.shape)
# print(x_test.shape)
# print(y_test.shape)
mae = mean_absolute_error(y_test, y_pred)
# warnings.filterwarnings('ignore')
print("mae 값은:", mae)
# warnings.filterwarnings('ignore')

# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')


