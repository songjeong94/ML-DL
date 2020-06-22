from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import  r2_score

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
            train_size =0.8)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("R2: ", score)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds: #중요하지 않은 컬럼들을 하나씩 지워나간다.
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    selection_x_train = selection.transform(x_train)

    #print(selection_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    r2 = r2_score(y_test, y_pred)
    #print("R2:",r2)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1],
                        r2*100.0))

# 그리드 서치까지 엮어라.

#데이콘 적용해라, 71개 컬럼

#메일 제목: 말똥이 24등
