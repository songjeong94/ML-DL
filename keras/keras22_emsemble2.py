#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311,411)])
x2 = np.array([range(711,811), range(711,811)]) 

y1 = np.array([range(101, 201), range(411,511)])
y2 = np.array([range(501,601), range(711,811)]) 
y3 = np.array([range(411,511), range(611,711)]) 

 #  열우선, 행무시

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
y3_train, y3_test = train_test_split(
   y3, shuffle=False, train_size=0.8)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(2, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense1_1 = Dense(100, activation='relu', name = 'hi_1')(input1) #꼬리에 모델명을 붙여준다.
dense1_2 = Dense(100, activation='relu', name = 'hi_2')(dense1_1) 
dense1_3 = Dense(100, activation='relu', name = 'hi_3')(dense1_2)


input2 = Input(shape=(2, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다.
dense2_1 = Dense(100, activation='relu', name = 'hello_1')(input2) #꼬리에 모델명을 붙여준다.
dense2_2 = Dense(100, activation='relu', name = 'hello_2')(dense2_1) 
dense2_3 = Dense(100, activation='relu', name = 'hello_3')(dense2_2)


from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3])

middle1 = Dense(100)(merge1)
middle1 = Dense(100)(middle1)

######## output 모델 구성########## 

output1 = Dense(10)(middle1)
output1_2 = Dense(10)(output1)
output1_3 = Dense(2)(output1_2)

output2 = Dense(10)(middle1)
output2_2 = Dense(10)(output2)
output2_3 = Dense(2)(output2_2)

output3 = Dense(10)(middle1)
output3_2 = Dense(10)(output3)
output3_3 = Dense(2)(output3_2)

model = Model(inputs = [input1, input2], outputs = [output1_3, output2_3, output3_3]) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

#model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train] ,epochs=50, batch_size=1, validation_split=(0.3), verbose=1) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측

loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test, y3_test], batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.

print("loss: ", loss)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
RMSE1 =  RMSE(y1_test, y1_predict)
RMSE2 =  RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", (RMSE1 + RMSE2 + RMSE3)/3 )


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
r2_y3_predict = r2_score(y3_test, y3_predict)

print("R2_1 :", r2_y_predict )
print("R2_2 :", r2_y2_predict )
print("R2_3 :", r2_y3_predict )
print("R2 :", (r2_y_predict + r2_y2_predict + r2_y3_predict)/3 )
