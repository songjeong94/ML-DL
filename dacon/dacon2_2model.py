import numpy as np
from keras.models import Sequential, Input
from keras.layers import Dense, LSTM, Dropout, Input ,Flatten, Conv1D
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from keras.utils import np_utils

train = np.load('./data/dacon/comp2/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp2/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp2/submission.npy', allow_pickle='True')
target = np.load('./data/dacon/comp2/target.npy', allow_pickle='True')


print(train.shape) # 2800, 4
print(target.shape)  # 2800, 4
print(test.shape) #700,4

train = train.reshape(2800, 4, 1)
test = test.reshape(700, 4,1)
x_train, x_test, y_train, y_test = train_test_split(
    train , target,  train_size=0.8
)

model = Sequential()
model.add(Conv1D(100,2 ,input_shape=(4,1)))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Flatten())
model.add(Dense(4))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=20, batch_size=20)

loss, mse = model.evaluate(x_test, y_test)

print("mse:", mse)

y_pred = model.predict(test)
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