import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMRegressor
import pandas as pd
train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')

x = train[:, :71]
y = train[:, 71:]

test = test[:, :71]

print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle='True', random_state=1)

model = LGBMRegressor()
model.fit(x_train,y_train)

y_pred = model.predict(test)
print(model.feature_importances_)
print(y_pred)
print(y_pred.shape)

import matplotlib.pyplot as plt
import numpy as np

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')


