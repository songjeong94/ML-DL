#과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from xgboost import XGBRFClassifier, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 화귀 모델
iris = load_iris()
x = iris.data
y = iris.target

print(x.shape) # (506, 13)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, shuffle=True)

n_estimators = 3300    # 나무의 개수
learning_rate = 1    # 학습률
colsample_bytree = 0.92  # 0.6~0.9사용
colsample_bylevel = 0.92 # 0.6~0.9사용

max_depth = 6
n_jobs = -1

model = XGBRFClassifier(maxdepth= max_depth, learning_rate = learning_rate,
                    n_estimators = n_estimators, n_jobs=n_jobs,
                    colsample_bytree = colsample_bytree,
                    colsample_bylevel = colsample_bylevel) # 결측치제거 전처리 안해도된다.

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('정수: ', score)

plot_importance(model)
# plt.show()