#RandomizedSearchCV + Pipeline
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn .ensemble import RandomForestClassifier

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

parameters =  { 'randomforestclassifier__n_estimators' :[10, 100],
           'randomforestclassifier__max_depth'  :[6, 8, 10, 12],
           'randomforestclassifier__min_samples_leaf'  :[8, 12, 18],
           'randomforestclassifier__min_samples_split'  :[8, 16, 20]}
#20가지 경우의 수

#2. 모델

#model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#pipe = Pipeline([("scaler", MinMaxScaler()), ('rf', RandomForestClassifier())])
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = RandomizedSearchCV(pipe, parameters, cv=5)


#3. 훈련
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print("최적의 매개 변수 :  ", model.best_estimator_) #estimator(모든 파라미터) , params(내가 지정한 파라미터)
print("acc: ", acc)

import sklearn as sk
print("sklearn :", sk.__version__)