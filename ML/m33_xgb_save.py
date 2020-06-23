from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor, XGBRFClassifier

#
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

model = XGBRFClassifier(n_estimators=1000, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric="error",
                    eval_set = [(x_train, y_train), (x_test, y_test)])

#rmse,mae,logloss,error,auc

results = model.evals_result()
print("eval:", results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print("acc:", acc)

# import pickle
# pickle.dump(model, open("./model/sample/xgb_save/cancer.pickle.dat", "wb"))

# import joblib
# joblib.dump(model, "./model/sample/xgb_save/cancer.joblib.dat")

model.save_model("./model/sample/xgb_save/cancer.model")
print("저장됨.")

# model2 = joblib.load("./model/sample/xgb_save/cancer.joblib.dat")
model2 = XGBRFClassifier()
model2.load_model("./model/sample/xgb_save/cancer.model")
print("불러오깅")

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred) 
print("acc:", acc)

#save_model 및 dump_model 함수는 모델을 저장하지만 차이점은 dump_model에서 기능 이름을 저장하고 트리를 텍스트 형식으로 저장할 수 있다는 것입니다.

#피클은 파이썬에서 객체를 직렬화하는 표준 방법입니다

#피클에 대하여 
# import pickle 을 통하여 모듈 임포트가 필요하다.
# pickle 모듈을 이용하면 원하는 데이터를 자료형의 변경없이 파일로 저장하여 그대로 로드할 수 있다.
# (open(‘text.txt’, ‘w’) 방식으로 데이터를 입력하면 string 자료형으로 저장된다.)
# pickle로 데이터를 저장하거나 불러올때는 파일을 바이트형식으로 읽거나 써야한다. (wb, rb)
# wb로 데이터를 입력하는 경우는 .bin 확장자를 사용하는게 좋다.
# 모든 파이썬 데이터 객체를 저장하고 읽을 수 있다.