
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from lightgbm import LGBMRegressor
import warnings 
import pandas as pd

train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')
print(train)
x1 = train[:, 1:36]
x2 = train[:, 36:71]

y1 = train[:, 71:72]
y2 = train[:, 72:73]
y3 = train[:, 73:74]
y4 = train[:, 74:75]


test = test[:, :71]




x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size=0.2, shuffle='True', random_state=66)
x2_train, x2_test, y1_train, y1_test = train_test_split(x2,y1, test_size=0.2, shuffle='True', random_state=66)

y2_train, y2_test = train_test_split(y2, test_size=0.2, shuffle='True', random_state=66)
y3_train, y3_test = train_test_split(y3, test_size=0.2, shuffle='True', random_state=66)
y4_train, y4_test = train_test_split(y4, test_size=0.2, shuffle='True', random_state=66)

scaler = MinMaxScaler()
scaler.fit(x1_train)
scaler.fit(x2_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

# parameters=[
#     { 'n_estimators' : [6,10,30,100,300],
#     'learning_rate' : [0.01,0.5 , 1],
#     'colsample_bytree' : [0.6, 0.8, 0.9], # 0.6~0.9사용
#     'colsample_bylevel': [0.6, 0.8, 0.9],
#     'max_depth' : [6,7,8]}
# ]
gpu_id=0, tree_method='gpu_hist'
# model1 = XGBRFRegressor(n_estimators= 300,learning_rate=1,colsample_bytree=1,colsample_bylevel=1,max_depth=50,subsample=0.8, n_jobs=-1)
# model2 = XGBRFRegressor(n_estimators= 400,learning_rate=1,colsample_bytree=1,colsample_bylevel=1,max_depth=50)
model2 =LGBMRegressor()
# model3 = XGBRFRegressor(n_estimators= 350,learning_rate=1,colsample_bytree=1,colsample_bylevel=1,max_depth=40,subsample=1,n_jobs=-1)
# model4 = XGBRFRegressor(n_estimators= 100,learning_rate=1,colsample_bytree=1,colsample_bylevel=0.7,max_depth=30,n_jobs=-1)

# model = GridSearchCV(model, parameters, cv =5)
# model = MultiOutputRegressor(model2)

warnings.filterwarnings('ignore')
# model1.fit(x_train, y1_train)
model2.fit([x1_train,x2_train], y2_train)
# model3.fit(x_train, y3_train)
# model4.fit(x_train, y4_train)

# y1_pred = model1.predict(x_test)
# print(y1_pred)
# print(y1_pred.shape)

y2_pred = model2.predict([x1_test,x2_test])
# print(y2_pred)
# print(y2_pred.shape)

# y3_pred = model3.predict(x_test)
# print(y3_pred)
# print(y3_pred.shape)

# y4_pred = model4.predict(x_test)
# print(y4_pred)
# print(y4_pred.shape)



# mae1 = mean_absolute_error(y1_test, y1_pred)
mae2 = mean_absolute_error(y2_test, y2_pred)
# mae3 = mean_absolute_error(y3_test, y3_pred)
# mae4 = mean_absolute_error(y4_test, y4_pred)

warnings.filterwarnings('ignore')
# print(mae1)
print(mae2)
# print(mae3)
# print(mae4)

# print("최적의 매개 변수 :  ", model.best_params_)
warnings.filterwarnings('ignore')
# thresholds = np.sort(model3.feature_importances_)

#model1
# print(thresholds)
# for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
#     selection = SelectFromModel(model1, threshold=thresh, prefit=True)

#     selection_x_train = selection.transform(x_train)

#     #print(selection_x_train.shape)

#     selection_model1 = XGBRFRegressor(n_estimators= 300, learning_rate=1, colsample_bytree=0.99, colsample_bylevel=0.99, max_depth=50, n_jobs=-1)
#     selection_model1.fit(selection_x_train, y3_train)

#     selection_x_test = selection.transform(x_test)
#     y3_pred = selection_model1.predict(selection_x_test)

#     mae = mean_absolute_error(y3_test, y3_pred)
#     #print("R2:",r2)

#     print("Thresh=%.3f, n=%d, mae: %.2f%%" %(thresh, selection_x_train.shape[1], mae))

#model3
# for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
#     selection = SelectFromModel(model3, threshold=thresh, prefit=True)

#     selection_x_train = selection.transform(x_train)

#     #print(selection_x_train.shape)

#     selection_model3 = XGBRegressor(n_estimators= 350,learning_rate=1,colsample_bytree=1,colsample_bylevel=1,max_depth=40,subsample=1,n_jobs=-1)
#     selection_model3.fit(selection_x_train, y3_train)

#     selection_x_test = selection.transform(x_test)
#     y3_pred = selection_model3.predict(selection_x_test)

#     mae = mean_absolute_error(y3_test, y3_pred)
#     #print("R2:",r2)

#     print("Thresh=%.3f, n=%d, mae: %.2f%%" %(thresh, selection_x_train.shape[1], mae))


#model4
# for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
#     selection = SelectFromModel(model4, threshold=thresh, prefit=True)

#     selection_x_train = selection.transform(x_train)

#     #print(selection_x_train.shape)

#     selection_model4 = XGBRegressor(n_estimators= 100,learning_rate=0.01,colsample_bytree=0.99,colsample_bylevel=0.99,max_depth=50,n_jobs=-1)
#     selection_model4.fit(selection_x_train, y4_train)

#     selection_x_test = selection.transform(x_test)
#     y4_pred = selection_model4.predict(selection_x_test)

#     mae = mean_absolute_error(y4_test, y4_pred)
#     #print("R2:",r2)

#     print("Thresh=%.3f, n=%d, mae: %.2f%%" %(thresh, selection_x_train.shape[1], mae))

# a = np.arange(10000,20000)
# y1_pred = pd.DataFrame(y1_pred,a)
# y1_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header='hhb',index_label='id')

# a = np.arange(10000,20000)
# y2_pred = pd.DataFrame(y2_pred,a)
# y2_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header='hbo2',index_label='id')

# a = np.arange(10000,20000)
# y3_pred = pd.DataFrame(y3_pred,a)
# y3_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header='ca',index_label='id')

# a = np.arange(10000,20000)
# y4_pred = pd.DataFrame(y4_pred,a)
# y4_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header='na',index_label='id')




# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')


