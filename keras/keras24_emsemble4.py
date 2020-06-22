#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(301, 401)])

y1 = np.array([range(711,811), range(611,711)]) 
y2 = np.array([range(101,201), range(411,511)])

 #  열우선, 행무시

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, random_state=66, shuffle=True, train_size=0.8)

print(x1_train.shape) #(80, 2)
print(y1_test.shape) # (20, 2)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(2, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense1_1 = Dense(100, activation='relu', name = 'hi_1')(input1) #꼬리에 모델명을 붙여준다.
dense1_2 = Dense(200, activation='relu', name = 'hi_2')(dense1_1) 
dense1_3 = Dense(100, activation='relu', name = 'hi_3')(dense1_2)


######## output 모델 구성########## 

output1 = Dense(10)(dense1_3)
output1_2 = Dense(10)(output1)
output1_3 = Dense(2)(output1_2)

output2 = Dense(10)(dense1_3)
output2_2 = Dense(10)(output2)
output2_3 = Dense(2)(output2_2)

model = Model(inputs = input1, outputs = [output1_3, output2_3]) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

#model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train, [y1_train, y2_train] ,epochs=50, batch_size=1, validation_split=(0.25), verbose=1) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측

loss = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

print("loss: ", loss)

y1_predict, y2_predict = model.predict(x1_test)
print("y1_predict: ", y1_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE : ", (RMSE1 + RMSE2)/2)


# R2 구하기
from sklearn.metrics import r2_score
r2_y1 = r2_score(y1_test, y1_predict)
r2_y2 = r2_score(y2_test, y2_predict)

print("R2 :", (r2_y1 + r2_y2)/2)
