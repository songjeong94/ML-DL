from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor, XGBRFClassifier

#
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

model = XGBRFClassifier(n_estimators=1000, learning_rate=1)

model.fit(x_train, y_train, verbose=True, eval_metric="error",
                    eval_set = [(x_train, y_train), (x_test, y_test)])

#rmse,mae,logloss,error,auc

results = model.evals_result()
print("eval:", results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print("acc:", acc)

import pickle
# pickle.dump(model, open("./model/sample/xgb_save/cancer.pickle.dat", "wb"))

print("저장됨.")

model2 = pickle.load(open("./model/sample/xgb_save/cancer.pickle.dat", "rb"))
print("불러오깅")

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print("acc:", acc)
