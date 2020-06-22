#1.데이터
import numpy as np #numpy를 import 하고 numpy의 이름을 np로 줄여서 쓴다.
x = np.array([1,2,3,4,5,6,7,8,9,10]) #정제된 데이터
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential #Sequential:순차적 모델을 만든다.
from keras.layers import Dense #Dense:1차함수 

model = Sequential() #순차적 모델의 모델명
model.add(Dense(1, input_dim = 1, activation='relu'))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #손실을 줄이기위해 mse 사용 adam=최적화 metrics=훈련상황을 모니터링
model.fit(x,y,epochs=100,batch_size = 1) #x와 y룰 훈련 /epochs:몇번훈련시킬것인가 /batch size:몇개씩 잘를것인가

#4. 평가 예측
loss, acc = model.evaluate(x,y,batch_size=1)
print("loss : ", loss)
print("acc : ", acc)