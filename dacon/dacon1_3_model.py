import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input 
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')

x = train[:, :71]
y = train[:, 71:]

test = test[:, :71]
print(x.shape)
print(y.shape)
stan = StandardScaler()
x = stan.fit_transform(x)
print(x.shape)
print(y.shape)
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle='True', random_state=1)

model = Sequential()
model.add(Dense(100, input_dim=71))
model.add(Dropout(0.7))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(4, activation='relu'))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=20, batch_size=10)

loss , mae = model.evaluate(x_test, y_test)

print("mae: ", mae)

y_pred = model.predict(test)
print(y_pred)
print(y_pred.shape)

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center') #barh = 가로 막대 그래프./ 중앙정렬
    plt.yticks(np.arange(n_features), x.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()

