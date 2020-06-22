#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311,411), range(100)])
y1 = np.array([range(711,811), range(711,811), range(100)]) 

x2 = np.array([range(101, 201), range(411,511), range(100, 200)])
y2 = np.array([range(501,601), range(711,811), range(100)]) 

 #  열우선, 행무시

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False, train_size=0.8)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense1_1 = Dense(10, activation='relu', name = 'hi_1')(input1) #꼬리에 모델명을 붙여준다.
dense1_2 = Dense(9, activation='relu', name = 'hi_2')(dense1_1) 
dense1_3 = Dense(8, activation='relu', name = 'hi_3')(dense1_2)
dense1_4 = Dense(7, activation='relu', name = 'hi_4')(dense1_3) 

input2 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense2_1 = Dense(20, activation='relu', name = 'hello_1')(input2) #꼬리에 모델명을 붙여준다.
dense2_2 = Dense(19, activation='relu', name = 'hello_2')(dense2_1) 
dense2_3 = Dense(18, activation='relu', name = 'hello_3')(dense2_2)
dense2_4 = Dense(17, activation='relu', name = 'hello_4')(dense2_3) 

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_4, dense2_4])

middle1 = Dense(30)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

######## output 모델 구성########## 

output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(30)(middle1)
output2_2 = Dense(5)(output2)
output2_3 = Dense(3)(output2_2)


model = Model(inputs = [input1, input2], outputs = [output1_3, output2_3]) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

#model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=(0.3), verbose=1) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측

loss = model.evaluate([x1_test. x2_test],[y1_test, y2_test], batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)


y_predict = model.predict([x1_test, x2_test])
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
