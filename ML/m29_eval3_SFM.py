# xgboost evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor,XGBRFClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_iris
import pickle

## 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBRFClassifier(n_estimators = 300,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['merror', 'mlogloss'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)])
        #   early_stopping_rounds = 100)
# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

results = model.evals_result()
print("eval's result : ", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
# print("r2 Score : %.2f%%" %(r2 * 100))
print("acc : ", acc)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
   

    print(selection_x_train.shape)
    
    selection_model = XGBRFClassifier( objective="multi:softprob",n_jobs=-1)

    selection_model.fit(selection_x_train,y_train, eval_metric = ['merror', 'mlogloss'],
          eval_set = [(selection_x_train, y_train),
                      (selection_x_test, y_test)])

    y_pred = selection_model.predict(selection_x_test)

    acc = accuracy_score(y_test, y_pred)
    #print("R2:",r2)
    for i in thresholds:
        pickle.dump(model, open("./model/sample/xgb_save/iris.pickle{}.dat".format(selection_x_train.shape[1]), "wb"))
    print("Thresh=%.3f, n=%d, acc: %.2f%%" %(thresh, selection_x_train.shape[1],
                        acc*100.0))


## 그래프, 시각화
# epochs = len(results['validation_0']['logloss'])
# x_axis = range(0, epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label = 'Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label = 'Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['rmse'], label = 'Train')
# ax.plot(x_axis, results['validation_1']['rmse'], label = 'Test')
# ax.legend()
# plt.ylabel('RMSE')
# plt.title('XGBoost RMSE')
# plt.show()