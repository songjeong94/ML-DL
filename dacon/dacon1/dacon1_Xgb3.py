import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input 
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
import warnings 

import pandas as pd
train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')

x = train[:, :71]
y1 = train[:, 71:72]
y2 = train[:, 72:73]
y3 = train[:, 73:74]
y4 = train[:, 74:75]


test = test[:, :71]

print(x.shape)
x_train, x_test, y1_train, y1_test = train_test_split(x,y1, test_size=0.2, shuffle='True', random_state=66)
x_train, x_test, y2_train, y2_test = train_test_split(x,y2, test_size=0.2, shuffle='True', random_state=66)
x_train, x_test, y3_train, y3_test = train_test_split(x,y3, test_size=0.2, shuffle='True', random_state=66)
x_train, x_test, y4_train, y4_test = train_test_split(x,y4, test_size=0.2, shuffle='True', random_state=66)

          
# parameters=[
#     { 'n_estimators' : [6,10,30,100,300],
#     'learning_rate' : [0.01,0.5 , 1],
#     'colsample_bytree' : [0.6, 0.8, 0.9], # 0.6~0.9사용
#     'colsample_bylevel': [0.6, 0.8, 0.9],
#     'max_depth' : [6,7,8]}
# ]

# model1 = XGBRFRegressor(n_estimators= 300,learning_rate=1,colsample_bytree=0.99,colsample_bylevel=0.99,max_depth=50,nrounds=1000,scale_pos_weight=1.5)
#model2 = XGBRFRegressor(n_estimators= 400,learning_rate=1,colsample_bytree=0.99,colsample_bylevel=0.99,max_depth=50,nrounds=1000,scale_pos_weight=1.5)
model3 = XGBRFRegressor(n_estimators= 400,learning_rate=1,colsample_bytree=0.99,colsample_bylevel=0.99,max_depth=10,nrounds=1000,scale_pos_weight=1.5)
# model4 = XGBRFRegressor(n_estimators= 100,learning_rate=1,colsample_bytree=0.99,colsample_bylevel=0.99,max_depth=50,nrounds=1000,scale_pos_weight=1.5)

# model = GridSearchCV(model, parameters, cv =5)
# model = MultiOutputRegressor(model)

warnings.filterwarnings('ignore')
# model1.fit(x_train, y1_train)
#model2.fit(x_train, y2_train)
model3.fit(x_train, y3_train)
# model4.fit(x_train, y4_train)

# y1_pred = model1.predict(test)
# print(y1_pred)
# print(y1_pred.shape)

# y2_pred = model2.predict(test)
# print(y2_pred)
# print(y2_pred.shape)

# y3_pred = model3.predict(test)
# print(y3_pred)
# print(y3_pred.shape)

# y4_pred = model4.predict(test)
# print(y4_pred)
# print(y4_pred.shape)



# acc1 = model1.score(x_test, y1_test)
# acc2 = model2.score(x_test, y2_test)
acc3 = model3.score(x_test, y3_test)
# acc4 = model4.score(x_test, y4_test)

warnings.filterwarnings('ignore')
# print(acc1)
# print(acc2)
print(acc3)
# print(acc4)

# print("최적의 매개 변수 :  ", model.best_params_)
warnings.filterwarnings('ignore')
# thresholds = np.sort(model.feature_importances_)

# print(thresholds)

# for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)

#     selection_x_train = selection.transform(x_train)

#     #print(selection_x_train.shape)

#     selection_model = XGBRegressor()

#     parameters=[
#     { 'n_estimators' : [6,10,30,100,300],
#     'learning_rate' : [0.01,0.5 , 1],
#     'colsample_bytree' : [0.6, 0.8, 0.9], # 0.6~0.9사용
#     'colsample_bylevel': [0.6, 0.8, 0.9],
#     'max_depth' : [6,7,8]}
#     ]

#     selection_model = GridSearchCV(selection_model, parameters, cv=5, n_jobs= -1)
#     model = MultiOutputRegressor(selection_model) 
#     selection_model.fit(selection_x_train, y_train)

#     selection_x_test = selection.transform(x_test)
#     y_pred = selection_model.predict(selection_x_test)

#     r2 = r2_score(y_test, y_pred)
#     #print("R2:",r2)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
#                         r2*100.0))
#     print("최적의 매개변수 : ", model.best_estimator_)

# print("최적의 매개변수 : ", model.best_estimator_)
# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')


