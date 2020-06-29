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
import matplotlib.pyplot as plt
import warnings 

import pandas as pd
train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')

x = train[:, :71]
xfreq = np.asarray(x)

plt.figure()
plt.scatter(xfreq[:,1], xfreq[:,70])
plt.show()

y = train[:, 71:]

test = test[:, :71]

# from scipy.interpolate import splrep, splev
# spl = splrep(xfreq[:,0], xfreq[:,1])
# fs = 1000 # max 500Hz *2
# dt = 1/fs # 0.001
# spl = splrep(xfreq[:,0], xfreq[:,1])
# newt = np.arange(0, 1, dt)
# newx = splev(newt, spl)
# print(newt)
# print(newx)

# if True:
#     plt.figure()
#     plt.scatter(newt, newx)
#     plt.savefig('fft02_2.jpg')
#     plt.show()

