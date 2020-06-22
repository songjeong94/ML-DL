#과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from xgboost import XGBRFRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# 화귀 모델
boston = load_boston()
x = boston.data
y = boston.target

print(x.shape) # (506, 13)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, shuffle=True)

parameters=[
    { 'n_estimators' : [300,500,3300],
    'learning_rate' : [0.01,0.5 , 1],
    'colsample_bytree' : [0.6, 0.8, 0.9], # 0.6~0.9사용
    'colsample_bylevel': [0.6, 0.8, 0.9],
    'max_depth' : [6,7,8]}
]

model = GridSearchCV(XGBRFRegressor(), parameters, cv=5, n_jobs= -1) # 결측치제거 전처리 안해도된다.

model.fit(x_train, y_train)

print(model.best_estimator_)
print("==========================================")
print(model.best_params_)
print("==========================================")
score = model.score(x_test, y_test)
print('정수: ', score)

# plot_importance(model)
# plt.show()