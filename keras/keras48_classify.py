import numpy as np

x = np.array(range(1, 11))
y = np.array([1,0,1,0,1,0,1,0,1,0])

print(x.shape)
print(y.shape)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(10, input_dim =1, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


#3.컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=80, batch_size=1)
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)
x_pred = np.array([1,2,3,4])
x_pred = np.transpose(x_pred)
y_predict = model.predict(x_pred)

#y_predict = np.around(y_predict)
#print(y_predict)
for f in y_predict:
    if f >= 0.5:
        print("[1]")
    else:
        print("[0]")
        
print('y_pred :', y_predict)



#과제 : 하이퍼파라미터 튜닝