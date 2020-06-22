import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings 

warnings.filterwarnings('ignore')
#1 데이터

iris = pd.read_csv('./data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:,4]

print(x)
print(y)

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=6)


kfold = KFold(n_splits=5, shuffle=True)


warnings.filterwarnings('ignore')
allAlgorithms =all_estimators(type_filter ='classifier') # 모든 분류 모델 확인

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    scores = cross_val_score(model, x,y, cv=kfold)
    
    print(name, "의 정답률 = ")
    print(scores)
    #model.fit(x, y)

import sklearn
print(sklearn.__version__)

