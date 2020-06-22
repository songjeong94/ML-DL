import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd


train = np.load('./data/dacon/comp2/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp2/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp2/submission.npy', allow_pickle='True')
target = np.load('./data/dacon/comp2/target.npy', allow_pickle='True')

print(train.shape) # 2800, 4
print(target.shape)  # 2800, 4
print(test.shape) #700,4

x_train, x_test, y_train, y_test = train_test_split(
    train , target,  train_size=0.7
)

parameters ={
    'rf__n_estimators' : [100],
    'rf__max_depth' : [10],
    'rf__min_samples_leaf' : [3],
    'rf__min_samples_split' : [5]
}

pipe = Pipeline([('scaler', MinMaxScaler()), ('rf',RandomForestRegressor())])
kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(pipe, parameters, cv=kfold, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

y_pred = model.predict(test)
print(y_pred)
# a = np.arange(2800,3500)
#y_pred = pd.DataFrame(y_pred,a)
#y_pred.to_csv('./data/dacon/comp2/sample_submission.csv', index = True, header=['X','Y','M','V'],index_label='id')


submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./data/dacon/comp2/comp2_sub.csv', index = False)