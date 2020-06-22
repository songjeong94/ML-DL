import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import warnings 
from sklearn.datasets import load_boston
import numpy as np

warnings.filterwarnings('ignore')

boston = load_boston()
print(boston)
x = boston.data
y = boston.target

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=6)

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter ='regressor') # 모든 분류 모델 확인

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 = ", r2_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)

