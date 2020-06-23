
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import warnings 
import pandas as pd

train = np.load('./data/dacon/comp1/train.npy', allow_pickle='True')
test = np.load('./data/dacon/comp1/test.npy', allow_pickle='True')
submission = np.load('./data/dacon/comp1/submission.npy', allow_pickle='True')
print(train)
x1 = train[:, 1:36]
x2 = train[:, 36:71]

y = train[:, 71:]

test = test[:, :71]

x1_train, x1_test, y_train, y_test = train_test_split(x1,y, test_size=0.2, shuffle='True', random_state=66)
x2_train, x2_test = train_test_split(x2, test_size=0.2, shuffle='True', random_state=66)



scaler = MinMaxScaler()
scaler.fit(x1_train)
scaler.fit(x2_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

print(x1_train.shape)
print(x2_train.shape)
print(y_train.shape)
x1_train = x1_train.reshape(-1,35,1)
x1_test = x1_test.reshape(-1,35,1)
x2_train = x2_train.reshape(-1,35,1)
x2_test = x2_test.reshape(-1,35,1)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input,Flatten,Dropout

input1 = Input(shape=(35,1))
dense1_1 = Dense(100, activation='relu', name = 'hi_1')(input1)
drop1 = Dropout(0.4)(dense1_1)
dense1_2 = Dense(200, activation='relu', name = 'hi_2')(drop1)
drop2 = Dropout(0.4)(dense1_2) 
dense1_3 = Dense(100, activation='relu', name = 'hi_3')(drop2)


input2 = Input(shape=(35,1)) 
dense2_1 = Dense(40, activation='relu', name = 'hello_1')(input2) 
drop3 = Dropout(0.4)(dense2_1) 
dense2_2 = Dense(80, activation='relu', name = 'hello_2')(drop3)
drop4 = Dropout(0.4)(dense2_2) 
dense2_3 = Dense(20, activation='relu', name = 'hello_3')(drop4)


from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3])

middle1 = Dense(50)(merge1)
drop5 = Dropout(0.2)(middle1)
middle1 = Dense(50)(drop5)
######## output 모델 구성########## 

output1 = Dense(10)(middle1)
f1 = Flatten()(output1)
output1_2 = Dense(10)(f1)
output1_3 = Dense(4)(output1_2)


model = Model(inputs = [input1, input2], outputs = output1_3) 

#model.summary()

#3. 훈련
model.compile(loss = 'mae', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train ,epochs=10, batch_size=4, validation_split= 0.25) 

#4. 평가, 예측

loss = model.evaluate([x1_test, x2_test],y_test, batch_size=1) 

print("loss: ", loss)

y_pred = model.predict([x1_test, x2_test])

mae = mean_absolute_error(y_test, y_pred)
print("mae:", mae)

