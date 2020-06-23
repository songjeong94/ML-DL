import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input 
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import pandas as pd
train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')

x = train[:, :71]
y = train[:, 71:]

test = test[:, :71]

print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle='True', random_state=1)

parameters = { 'rf__n_estimators' : [10, 100],
           'rf__max_depth' : [6, 8, 10, 12],
           'rf__min_samples_leaf' : [8, 12, 18],
           'rf__min_samples_split' : [8, 16, 20]}


model2 = RandomForestRegressor(output=4)

pipe = Pipeline([("scaler", MinMaxScaler()),('rf', model2)])
search = RandomizedSearchCV(pipe, parameters, cv = 5)
search.fit(x_train, y_train)

#model2.fit(x_train,y_train)

y_pred = model2.predict(test)
print(model2.feature_importances_)
print(y_pred)
print(y_pred.shape)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_x(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center') #barh = 가로 막대 그래프./ 중앙정렬
    plt.yticks(np.arange(n_features))
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_x(model2)
plt.show()

# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')


