#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(311,411), range(100)])
y = np.array(range(711,811)) 

print(x.shape) #  열우선, 행무시

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.8)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#model = Sequential()
#model.add(Dense(5, input_dim = 3))
#model.add(Dense(4))
#model.add(Dense(1))

input1 = Input(shape=(3,1)) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense1 = Dense(10,input_shape=(3,1), activation='relu')(input1) #꼬리에 모델명을 붙여준다.
dense2 = Dense(10, activation='relu')(dense1) 
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3) 
output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs=output1) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=(0.3), verbose=2) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("mse : ", mse)


y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
