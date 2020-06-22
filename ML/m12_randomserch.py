#RandomForest  적용 
#cifar10 적용
#Random

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer
import numpy as np

#1 데이터

breast = load_breast_cancer()

x = breast.data
y = breast.target
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=6)

parameters =  { 'n_estimators' :[10, 100],
           'max_depth'  :[6, 8, 10, 12],
           'min_samples_leaf'  :[8, 12, 18],
           'min_samples_split'  :[8, 16, 20]}

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv =kfold ,n_jobs=-1, n_iter=10)

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))
  




