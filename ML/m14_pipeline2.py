#RandomizedSearchCV + Pipeline
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#1.데이터

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.2)

#그리드/ 랜덤 서치에서 사용할 매개 변수
# parameters = [
#     {"svm__C": [1,10,100,1000], "svm__kernel": ['linear']},
#     {"svm__C": [1,10,100,1000], "svm__kernel": ['rbf'], 'svm__gamma':[0.001, 0.0001]},
#     {"svm__C": [1,10,100,1000], "svm__kernel": ['sigmoid'], 'svm__gamma':[0.001, 0.0001]}
# ]

parameters = [
    {"svc__C": [1,10,100,1000], "svc__kernel": ['linear']},
    {"svc__C": [1,10,100,1000], "svc__kernel": ['rbf'], 'svc__gamma':[0.001, 0.0001]},
    {"svc__C": [1,10,100,1000], "svc__kernel": ['sigmoid'], 'svc__gamma':[0.001, 0.0001]}
]

#20가지 경우의 수


#2. 모델
#model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC())
model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

#4. 평가, 예측
print("최적의 매개 변수 :  ", model.best_estimator_)
print("acc: ", acc)

import sklearn as sk
print("sklearn :", sk.__version__)