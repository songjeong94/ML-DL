#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311,411), range(411,511)])
x2 = np.array([range(711,811), range(711,811), range(511,611)]) 

y1 = np.array([range(101, 201), range(411,511), range(100)])

 #  열우선, 행무시

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, shuffle=True, train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train, x2_test = train_test_split(
    x2, random_state=66, shuffle=True, train_size=0.8)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense1_1 = Dense(100, activation='relu', name = 'hi_1')(input1) #꼬리에 모델명을 붙여준다.
dense1_2 = Dense(200, activation='relu', name = 'hi_2')(dense1_1) 
dense1_3 = Dense(100, activation='relu', name = 'hi_3')(dense1_2)


input2 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense2_1 = Dense(100, activation='relu', name = 'hello_1')(input2) #꼬리에 모델명을 붙여준다.
dense2_2 = Dense(200, activation='relu', name = 'hello_2')(dense2_1) 
dense2_3 = Dense(100, activation='relu', name = 'hello_3')(dense2_2)


from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3])

middle1 = Dense(100)(merge1)
middle1 = Dense(100)(middle1)
middle1 = Dense(100)(middle1)
middle1 = Dense(100)(middle1)

######## output 모델 구성########## 

output1 = Dense(10)(middle1)
output1_2 = Dense(10)(output1)
output1_3 = Dense(3)(output1_2)


model = Model(inputs = [input1, input2], outputs = output1_3) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

#model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train ,epochs=50, batch_size=1, validation_split=(0.25), verbose=1) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측

loss = model.evaluate([x1_test, x2_test],y1_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

print("loss: ", loss)

y1_predict = model.predict([x1_test, x2_test])
print("y1_predict: ", y1_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
RMSE =  RMSE(y1_test, y1_predict)
print("RMSE : ", RMSE)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y1_predict)

print("R2 :", r2_y_predict )

